#!/usr/bin/env python3
"""Tests for ctxscope. Run with: python3 -m unittest discover -s tests -v"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ctxscope  # noqa: E402

EXAMPLES = ROOT / "examples"


def analyze(payload, encoding="estimate"):
    messages, meta = ctxscope.normalize(payload, encoding)
    meta["encoding"] = encoding
    meta["source"] = "test"
    return messages, meta, ctxscope.analyze(messages, meta)


class TokenTests(unittest.TestCase):
    def test_empty_is_zero(self):
        self.assertEqual(ctxscope.estimate_tokens(""), 0)

    def test_whitespace_only_is_cheap(self):
        self.assertLessEqual(ctxscope.estimate_tokens("   \n\n\t "), 2)

    def test_monotonic(self):
        short = ctxscope.estimate_tokens("hello world")
        long = ctxscope.estimate_tokens("hello world " * 50)
        self.assertLess(short, long)

    def test_english_guidance(self):
        sentence = "The quick brown fox jumps over the lazy dog."
        self.assertGreaterEqual(ctxscope.estimate_tokens(sentence), 8)
        self.assertLessEqual(ctxscope.estimate_tokens(sentence), 13)

    def test_json_costs_more_than_prose_per_char(self):
        prose = "a" * 200
        payload = json.dumps({f"key{i}": i for i in range(20)})
        self.assertGreater(ctxscope.estimate_tokens(payload) / len(payload), ctxscope.estimate_tokens(prose) / len(prose))

    def test_close_to_tiktoken_when_available(self):
        try:
            import tiktoken
        except Exception:
            self.skipTest("tiktoken not installed")
        encoding = tiktoken.get_encoding("o200k_base")
        text = (
            "The agent loop is a scheduler with a language model inside it. Every turn it decides "
            "whether to call a tool, and every tool result lands in the context window whether or not "
            "it was useful. Teams notice the cost long before they notice the mechanism. "
        ) * 20
        exact = len(encoding.encode(text))
        estimate = ctxscope.estimate_tokens(text)
        self.assertLess(abs(estimate - exact) / exact, 0.12, f"estimate {estimate} vs exact {exact}")

    def test_exact_matches_tiktoken(self):
        try:
            import tiktoken
        except Exception:
            self.skipTest("tiktoken not installed")
        text = "one two three four five six seven"
        self.assertEqual(ctxscope.count_tokens(text, "o200k"), len(tiktoken.get_encoding("o200k_base").encode(text)))


class NormalizeTests(unittest.TestCase):
    def test_openai_roles_and_schemas(self):
        payload = {
            "model": "gpt-5",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "search", "arguments": "{\"q\": \"x\"}"}}]},
                {"role": "tool", "name": "search", "tool_call_id": "1", "content": "result"},
            ],
            "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
        }
        messages, meta, _ = analyze(payload)
        self.assertEqual(meta["model"], "gpt-5")
        schema_messages = [m for m in messages if m.synthetic == "tool_schemas"]
        self.assertEqual(len(schema_messages), 1)
        kinds = {part.kind for message in messages for part in message.parts}
        self.assertIn("tool_call", kinds)
        self.assertIn("tool_result", kinds)
        self.assertEqual(messages[-1].role, "tool")

    def test_anthropic_tool_result_block(self):
        payload = {
            "system": "sys",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "go"}]},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "read", "input": {"path": "x"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "file contents"}]},
            ],
        }
        messages, meta, _ = analyze(payload)
        self.assertEqual(meta["format"], "anthropic")
        kinds = [part.kind for message in messages for part in message.parts]
        self.assertIn("tool_result", kinds)
        self.assertIn("tool_call", kinds)
        tool_result = [part for message in messages for part in message.parts if part.kind == "tool_result"][0]
        self.assertEqual(tool_result.text, "file contents")

    def test_image_block_is_counted(self):
        payload = {"messages": [{"role": "user", "content": [{"type": "text", "text": "what"}, {"type": "image_url", "image_url": {"url": "data:..."}}]}]}
        messages, _, data = analyze(payload)
        self.assertEqual(data["totals"]["tokens"], sum(m.tokens for m in messages))
        self.assertTrue(any(row["kind"] == "image" for row in data["kinds"]))

    def test_plain_text_input(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("just some words in a file")
            path = handle.name
        messages, meta = ctxscope.load(path)
        self.assertEqual(meta["format"], "plain-text")
        self.assertEqual(messages[-1].role, "user")

    def test_jsonl_takes_last_request(self):
        rows = [
            {"messages": [{"role": "user", "content": "one"}]},
            {"messages": [{"role": "user", "content": "two"}, {"role": "assistant", "content": "answer two"}]},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            handle.write("\n".join(json.dumps(row) for row in rows))
            path = handle.name
        messages, meta = ctxscope.load(path)
        self.assertEqual(meta["requests"], 2)
        self.assertIn("two", messages[-1].text)

    def test_empty_request_is_not_a_crash(self):
        messages, _, data = analyze({"messages": []})
        self.assertEqual(data["totals"]["tokens"], 0)
        self.assertEqual(data["score"]["value"], 100)


class AnalysisTests(unittest.TestCase):
    def test_kind_totals_match_total(self):
        _, _, data = analyze(ctxscope.build_demo_trace())
        self.assertEqual(sum(row["tokens"] for row in data["kinds"]), data["totals"]["tokens"])

    def test_percentages_are_sane(self):
        _, _, data = analyze(ctxscope.build_demo_trace())
        self.assertAlmostEqual(sum(row["pct"] for row in data["kinds"]), 100.0, delta=1.0)

    def test_duplicates_are_found_and_priced(self):
        block = "the same tool result repeated " * 40
        payload = {
            "messages": [
                {"role": "user", "content": "task"},
                {"role": "tool", "content": block},
                {"role": "tool", "content": block},
                {"role": "tool", "content": block},
            ]
        }
        _, _, data = analyze(payload)
        self.assertEqual(len(data["duplicates"]), 1)
        cluster = data["duplicates"][0]
        self.assertEqual(cluster["count"], 3)
        self.assertEqual(cluster["wasted"], cluster["tokens"] * 2)
        self.assertEqual(data["totals"]["duplicate_tokens"], cluster["wasted"])

    def test_small_repeats_are_ignored(self):
        payload = {"messages": [{"role": "tool", "content": "ok"}, {"role": "tool", "content": "ok"}]}
        _, _, data = analyze(payload)
        self.assertEqual(data["duplicates"], [])

    def test_waste_finds_pretty_json(self):
        payload = {"messages": [{"role": "tool", "content": json.dumps({"a": list(range(30))}, indent=4)}]}
        _, _, data = analyze(payload)
        self.assertTrue(any(item["id"] == "pretty-json" for item in data["waste"]))

    def test_trims_never_increase_tokens(self):
        messages, meta, data = analyze(ctxscope.build_demo_trace())
        for row in data["trims"]["rows"]:
            self.assertGreaterEqual(row["tokens"], 0, row["id"])
        self.assertLessEqual(data["trims"]["safe_tokens"], data["trims"]["baseline"])
        self.assertGreaterEqual(data["trims"]["safe_tokens"], 0)

    def test_combos_are_monotonic_in_mask_bits(self):
        messages, meta, data = analyze(ctxscope.build_demo_trace())
        combos = data["trims"]["combos"]
        bits = data["trims"]["bits"]
        for mask in range(32):
            base = combos[str(mask)]
            for name, bit in bits.items():
                if mask & (1 << bit):
                    bigger = mask & ~(1 << bit)
                    self.assertLessEqual(base, combos[str(bigger)] + 1, f"{name} in {mask} increased tokens")

    def test_health_score_bounds(self):
        for payload in (ctxscope.build_demo_trace(), ctxscope.build_demo_trace(healthy=True)):
            _, _, data = analyze(payload)
            self.assertGreaterEqual(data["score"]["value"], 0)
            self.assertLessEqual(data["score"]["value"], 100)
            self.assertIn(data["score"]["grade"], ("A", "B", "C", "D", "F"))

    def test_clean_trace_scores_higher(self):
        _, _, messy = analyze(ctxscope.build_demo_trace())
        _, _, clean = analyze(ctxscope.build_demo_trace(healthy=True))
        self.assertGreater(clean["score"]["value"], messy["score"]["value"])
        self.assertLess(messy["trims"]["safe_pct"], 100)

    def test_message_rows_are_consistent(self):
        messages, meta, data = analyze(ctxscope.build_demo_trace())
        self.assertEqual(len(data["messages"]), len(messages))
        self.assertEqual(sum(row["tokens"] for row in data["messages"]), data["totals"]["tokens"])

    def test_recommendations_are_never_empty(self):
        for payload in (ctxscope.build_demo_trace(), ctxscope.build_demo_trace(healthy=True)):
            _, _, data = analyze(payload)
            self.assertTrue(ctxscope.recommendations(data))


class RenderTests(unittest.TestCase):
    def setUp(self):
        _, _, self.data = analyze(ctxscope.build_demo_trace())

    def test_markdown_has_all_sections(self):
        report = ctxscope.render_markdown(self.data)
        for heading in ("## Composition", "## Biggest messages", "## Trim simulator", "## Recommendations"):
            self.assertIn(heading, report)

    def test_html_is_self_contained(self):
        html = ctxscope.render_html(self.data)
        self.assertGreater(len(html), 20000)
        for forbidden in ("<script src", "<link ", "fetch(", "XMLHttpRequest", "@import", "cdn."):
            self.assertNotIn(forbidden, html, forbidden)
        self.assertIn("</html>", html)
        self.assertIn('"score"', html)

    def test_html_escapes_the_source_name(self):
        self.data["meta"]["source"] = "<script>alert(1)</script>"
        html = ctxscope.render_html(self.data)
        self.assertNotIn("<script>alert(1)</script>", html)

    def test_terminal_report_has_numbers(self):
        report = ctxscope.render_terminal(self.data, color=False)
        self.assertIn("context health", report)
        self.assertIn("TRIM SIMULATOR", report)
        self.assertNotIn("\033[", report)


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "ctxscope.py"), *args],
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_demo_runs(self):
        result = self.run_cli("--demo", "--no-color")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("WHERE THE TOKENS GO", result.stdout)

    def test_writes_json_and_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "out.json"
            html_path = Path(tmp) / "out.html"
            result = self.run_cli(str(EXAMPLES / "agent-trace.json"), "--json", str(json_path), "--html", str(html_path), "--quiet")
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(json_path.read_text())
            self.assertGreater(data["totals"]["tokens"], 1000)
            self.assertIn("ctxscope", html_path.read_text())

    def test_missing_file(self):
        result = self.run_cli("does-not-exist.json")
        self.assertEqual(result.returncode, 2)
        self.assertIn("no such file", result.stderr)

    def test_stdin(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "ctxscope.py"), "-", "--no-color"],
            input=json.dumps({"messages": [{"role": "user", "content": "hello from stdin"}]}),
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("hello from stdin", result.stdout)

    def test_examples_are_valid(self):
        for name in ("agent-trace.json", "healthy-trace.json", "anthropic-messages.json"):
            result = self.run_cli(str(EXAMPLES / name), "--quiet", "--json", "/dev/null")
            self.assertEqual(result.returncode, 0, f"{name}: {result.stderr}")


if __name__ == "__main__":
    unittest.main()
