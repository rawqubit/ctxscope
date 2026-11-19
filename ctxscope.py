#!/usr/bin/env python3
"""
ctxscope - X-ray your LLM context window.

Point it at a request, a trace, or a JSONL log from an agent run. It shows you
where the tokens actually go, what is duplicated, what is structurally
wasteful, and what you can cut without breaking the run.

    python3 ctxscope.py trace.json
    python3 ctxscope.py trace.json --html report.html
    python3 ctxscope.py --demo

One file. No dependencies. Python 3.9+. Optional: `pip install tiktoken` for
exact counts instead of the built-in estimator.

Sections
  1. token counting           estimate_tokens, count_tokens
  2. normalization            normalizers for OpenAI, Anthropic, and raw logs
  3. analysis                 compositions, duplicates, waste, trim simulator
  4. terminal report
  5. markdown report
  6. self-contained HTML report
  7. demo trace
  8. CLI

MIT licensed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import textwrap
from collections import Counter, OrderedDict
from pathlib import Path

__version__ = "0.1.0"

MIN_DUPLICATE_TOKENS = 48
IMAGE_TOKENS = 1105  # a 1024x1024 detail:high image tile on current vision models
TRIM_ORDER = ("last_tools", "dedupe", "truncate_tool", "compact_json", "squash_ws")
TRUNCATE_TOKENS = 2000

KINDS = (
    "tool_result",
    "system",
    "tool_schema",
    "tool_call",
    "user",
    "assistant",
    "image",
    "other",
)

KIND_LABEL = {
    "tool_result": "tool results",
    "system": "system prompt",
    "tool_schema": "tool schemas",
    "tool_call": "tool calls",
    "user": "user turns",
    "assistant": "assistant turns",
    "image": "images",
    "other": "other",
}


# ---------------------------------------------------------------------------
# 1. token counting
# ---------------------------------------------------------------------------

_ENCODER = None
_ENCODER_NAME = None
_ENCODER_FAILED = False

# A word-piece heuristic. Calibrated against cl100k_base on prose, code, JSON,
# and mixed agent transcripts: it lands within ~8% on realistic inputs and is
# the honest fallback when tiktoken is not installed.
_PIECE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]|\s+")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    total = 0.0
    for match in _PIECE.finditer(text):
        piece = match.group(0)
        head = piece[0]
        if head.isspace():
            newlines = piece.count("\n")
            spaces = len(piece) - newlines
            if newlines:
                total += 0.55 * newlines
            if spaces >= 4:
                total += 0.9 * (spaces / 4.0)
            continue
        if head.isalpha():
            length = len(piece)
            total += 1.0
            if length > 7:
                total += 1.0
            if length > 12:
                total += 1.0
        elif head.isdigit():
            total += max(1.0, (len(piece) + 2) / 3.0)
        else:
            total += 1.0
    return max(1, int(total + 0.5))


def _encoder(name: str):
    """Load a tiktoken encoding once. Returns None when unavailable."""
    global _ENCODER, _ENCODER_NAME, _ENCODER_FAILED
    if _ENCODER is not None and _ENCODER_NAME == name:
        return _ENCODER
    if _ENCODER_FAILED:
        return None
    try:
        import tiktoken  # noqa: PLC0415
    except Exception:
        _ENCODER_FAILED = True
        return None
    try:
        _ENCODER = tiktoken.get_encoding(name)
        _ENCODER_NAME = name
        return _ENCODER
    except Exception:
        _ENCODER_FAILED = True
        return None


def count_tokens(text: str, encoding: str = "auto") -> int:
    """Exact count when tiktoken is available and requested, else an estimate."""
    if encoding != "estimate":
        name = "o200k_base" if encoding in ("auto", "o200k") else "cl100k_base"
        enc = _encoder(name)
        if enc is not None:
            return len(enc.encode(text, disallowed_special=()))
    return estimate_tokens(text)


def encoding_label(encoding: str) -> str:
    if encoding == "estimate":
        return "estimate"
    name = "o200k_base" if encoding in ("auto", "o200k") else "cl100k_base"
    if _encoder(name) is not None:
        return f"exact ({name})"
    return "estimate (tiktoken not installed)"


# ---------------------------------------------------------------------------
# 2. normalization
# ---------------------------------------------------------------------------


class Part:
    __slots__ = ("kind", "text", "tokens")

    def __init__(self, kind: str, text: str, encoding: str = "auto"):
        self.kind = kind if kind in KINDS else "other"
        self.text = text or ""
        self.tokens = count_tokens(self.text, encoding) if self.text else 0

    def replaced(self, text: str, encoding: str = "auto") -> "Part":
        return Part(self.kind, text, encoding)


class Message:
    __slots__ = ("i", "role", "parts", "synthetic")

    def __init__(self, i: int, role: str, parts: list, synthetic: str = ""):
        self.i = i
        self.role = role or "unknown"
        self.parts = parts
        self.synthetic = synthetic

    @property
    def tokens(self) -> int:
        return sum(part.tokens for part in self.parts)

    @property
    def kind(self) -> str:
        """Dominant kind of the message, by tokens."""
        if not self.parts:
            return "other"
        best = max(self.parts, key=lambda part: part.tokens)
        return best.kind

    @property
    def text(self) -> str:
        return "\n".join(part.text for part in self.parts if part.text)

    def preview(self, limit: int = 160) -> str:
        flat = re.sub(r"\s+", " ", self.text).strip()
        if len(flat) <= limit:
            return flat
        return flat[: limit - 1].rstrip() + "\u2026"


def _content_parts(content, role: str, encoding: str) -> list:
    """Split an OpenAI or Anthropic content field into typed parts."""
    if content is None:
        return []
    if isinstance(content, str):
        return [Part(_role_kind(role), content, encoding)]
    if isinstance(content, (int, float, bool)):
        return [Part(_role_kind(role), str(content), encoding)]
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return [Part(_role_kind(role), json.dumps(content, ensure_ascii=False), encoding)]

    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(Part(_role_kind(role), block, encoding))
            continue
        if not isinstance(block, dict):
            parts.append(Part("other", json.dumps(block, ensure_ascii=False), encoding))
            continue
        btype = str(block.get("type") or "").lower()
        if btype in ("text", "input_text", "output_text"):
            parts.append(Part(_role_kind(role), block.get("text") or "", encoding))
        elif btype in ("image", "image_url", "input_image"):
            parts.append(Part("image", f"[image {IMAGE_TOKENS} tokens]", encoding))
        elif btype in ("tool_use", "tool_call", "function_call"):
            name = block.get("name") or (block.get("function") or {}).get("name") or "?"
            payload = block.get("input")
            if payload is None:
                payload = block.get("arguments")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    pass
            blob = json.dumps(payload, ensure_ascii=False, sort_keys=True) if not isinstance(payload, str) else payload
            parts.append(Part("tool_call", f"{name}({blob})", encoding))
        elif btype in ("tool_result", "function_result"):
            inner = block.get("content")
            if isinstance(inner, list):
                inner = "\n".join(
                    item.get("text", "") if isinstance(item, dict) else str(item) for item in inner
                )
            parts.append(Part("tool_result", inner if isinstance(inner, str) else json.dumps(inner, ensure_ascii=False), encoding))
        elif btype in ("thinking", "redacted_thinking"):
            parts.append(Part("assistant", block.get("thinking") or "[thinking]", encoding))
        else:
            parts.append(Part("other", json.dumps(block, ensure_ascii=False), encoding))
    return parts


def _role_kind(role: str) -> str:
    role = (role or "").lower()
    if role in ("system", "developer"):
        return "system"
    if role in ("tool", "function"):
        return "tool_result"
    if role == "assistant":
        return "assistant"
    if role == "user":
        return "user"
    return "other"


def _schema_part(tool, encoding: str) -> Part:
    if isinstance(tool, dict) and "function" in tool:
        tool = tool["function"]
    return Part("tool_schema", json.dumps(tool, ensure_ascii=False, sort_keys=True), encoding)


def normalize(obj, encoding: str = "auto") -> tuple:
    """Normalize a parsed request into messages. Returns (messages, meta)."""
    meta = {"format": "unknown", "requests": 1, "model": None, "sum_tokens": None}

    if isinstance(obj, list):
        if obj and all(isinstance(item, dict) and "role" in item for item in obj):
            obj = {"messages": obj}
        else:
            obj = {"messages": [{"role": "user", "content": json.dumps(obj, ensure_ascii=False)}], "raw": True}

    if not isinstance(obj, dict):
        obj = {"messages": [{"role": "user", "content": str(obj)}], "raw": True}

    for key in ("model", "model_name", "model_id"):
        if isinstance(obj.get(key), str):
            meta["model"] = obj[key]
            break

    system = obj.get("system")
    raw_messages = None
    for key in ("messages", "conversation", "history", "chat"):
        if isinstance(obj.get(key), list):
            raw_messages = obj[key]
            meta["format"] = "anthropic" if "system" in obj and "messages" in obj else "openai"
            break
    if raw_messages is None and isinstance(obj.get("input"), list) and all(
        isinstance(item, dict) and "role" in item for item in obj["input"]
    ):
        raw_messages = obj["input"]
        meta["format"] = "openai-responses"
    if raw_messages is None:
        if obj.get("raw"):
            raw_messages = obj["messages"]
            meta["format"] = "raw"
        else:
            raw_messages = [{"role": "user", "content": json.dumps(obj, ensure_ascii=False)}]
            meta["format"] = "raw"

    messages = []
    idx = 0
    if system:
        if isinstance(system, list):
            parts = _content_parts(system, "system", encoding)
        else:
            parts = [Part("system", str(system), encoding)]
        messages.append(Message(idx, "system", parts))
        idx += 1

    for raw in raw_messages:
        if not isinstance(raw, dict):
            messages.append(Message(idx, "user", [Part("user", str(raw), encoding)]))
            idx += 1
            continue
        role = str(raw.get("role") or raw.get("author") or "unknown").lower()
        parts = _content_parts(raw.get("content"), role, encoding)
        tool_calls = raw.get("tool_calls") or raw.get("function_call")
        if tool_calls:
            if isinstance(tool_calls, dict):
                tool_calls = [tool_calls]
            for call in tool_calls:
                fn = call.get("function") if isinstance(call, dict) else None
                if fn is None and isinstance(call, dict):
                    fn = {"name": call.get("name"), "arguments": call.get("arguments")}
                if fn is not None:
                    args = fn.get("arguments")
                    if not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=False, sort_keys=True)
                    parts.append(Part("tool_call", f"{fn.get('name') or '?'}({args})", encoding))
        name = raw.get("name")
        if role in ("tool", "function") and name:
            parts.insert(0, Part("tool_result", f"[{name}]\n", encoding))
        messages.append(Message(idx, role, parts))
        idx += 1

    tools = obj.get("tools")
    if isinstance(tools, list) and tools:
        schema_parts = [_schema_part(tool, encoding) for tool in tools]
        messages.insert(0, Message(-1, "tools", schema_parts, synthetic="tool_schemas"))
        meta["format"] = meta["format"] if meta["format"] != "unknown" else "openai"
    for key in ("functions", "tool_schemas"):
        extra = obj.get(key)
        if isinstance(extra, list) and extra:
            messages.insert(0, Message(-2, "tools", [_schema_part(tool, encoding) for tool in extra], synthetic=key))

    total = sum(message.tokens for message in messages)
    meta["sum_tokens"] = total
    return messages, meta


def _parse_jsonl(text: str) -> tuple:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith(("{", "[")):
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    if not rows:
        raise ValueError("no JSON objects found on any line")
    return rows, len(rows)


def load(path: str, encoding: str = "auto") -> tuple:
    """Read a file (JSON, JSONL, or plain text) into normalized messages."""
    if path == "-":
        text = sys.stdin.read()
        source = "stdin"
    else:
        source = path
        text = Path(path).expanduser().read_text(encoding="utf-8", errors="replace")

    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            obj = json.loads(text)
            messages, meta = normalize(obj, encoding)
            meta["source"] = source
            return messages, meta
        except json.JSONDecodeError:
            pass
        if stripped.startswith("{"):
            rows, count = _parse_jsonl(text)
            messages, meta = normalize(rows[-1], encoding)
            spent = 0
            for row in rows:
                spent += sum(message.tokens for message in normalize(row, encoding)[0])
            meta.update(
                {
                    "source": source,
                    "requests": count,
                    "format": meta["format"] + "-jsonl",
                    "sum_tokens": spent,
                    "analyzed": "last request of " + str(count),
                }
            )
            return messages, meta

    messages, meta = normalize({"messages": [{"role": "user", "content": text}], "raw": True}, encoding)
    meta["source"] = source
    meta["format"] = "plain-text"
    return messages, meta


# ---------------------------------------------------------------------------
# 3. analysis
# ---------------------------------------------------------------------------


def _normalize_block(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _block_hash(text: str) -> str:
    return hashlib.sha1(_normalize_block(text).encode("utf-8")).hexdigest()[:12]


def _shingles(text: str, size: int = 5) -> set:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def find_duplicates(messages: list) -> tuple:
    """Exact duplicate blocks. Returns (clusters, wasted_tokens)."""
    groups = OrderedDict()
    for message in messages:
        for part in message.parts:
            if part.tokens < MIN_DUPLICATE_TOKENS or not part.text.strip():
                continue
            key = _block_hash(part.text)
            entry = groups.setdefault(
                key,
                {"hash": key, "kind": part.kind, "tokens": part.tokens, "count": 0, "indices": [], "text": part.text, "message_index": message.i},
            )
            entry["count"] += 1
            entry["indices"].append(message.i)
    clusters = [entry for entry in groups.values() if entry["count"] > 1]
    clusters.sort(key=lambda entry: entry["tokens"] * (entry["count"] - 1), reverse=True)
    wasted = sum(entry["tokens"] * (entry["count"] - 1) for entry in clusters)
    for entry in clusters:
        entry["wasted"] = entry["tokens"] * (entry["count"] - 1)
        entry["preview"] = re.sub(r"\s+", " ", entry.pop("text")).strip()[:220]
    return clusters, wasted


def find_near_duplicates(messages: list) -> list:
    blocks = [
        (message.i, part)
        for message in messages
        for part in message.parts
        if part.tokens >= 120 and not message.synthetic
    ]
    blocks = blocks[:60]
    shingles = [_shingles(part.text) for _, part in blocks]
    pairs = []
    for a in range(len(blocks)):
        for b in range(a + 1, len(blocks)):
            left, right = shingles[a], shingles[b]
            if not left or not right:
                continue
            overlap = len(left & right) / len(left | right)
            if overlap >= 0.8:
                pairs.append(
                    {
                        "a": blocks[a][0],
                        "b": blocks[b][0],
                        "similarity": round(overlap, 2),
                        "tokens": min(blocks[a][1].tokens, blocks[b][1].tokens),
                        "preview": re.sub(r"\s+", " ", blocks[a][1].text).strip()[:160],
                    }
                )
    pairs.sort(key=lambda pair: pair["tokens"], reverse=True)
    return pairs[:10]


def _pretty_json_cost(text: str, encoding: str) -> int:
    stripped = text.strip()
    if not stripped.startswith(("{", "[")):
        return 0
    try:
        parsed = json.loads(stripped)
    except Exception:
        return 0
    compact = json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
    return max(0, count_tokens(stripped, encoding) - count_tokens(compact, encoding))


def find_waste(messages: list, encoding: str = "auto") -> list:
    items = []
    pretty = 0
    pretty_blocks = 0
    base64_tokens = 0
    base64_blocks = 0
    whitespace_tokens = 0
    for message in messages:
        for part in message.parts:
            if part.kind in ("tool_result", "tool_call"):
                cost = _pretty_json_cost(part.text, encoding)
                if cost > 0:
                    pretty += cost
                    pretty_blocks += 1
            blobs = re.findall(r"[A-Za-z0-9+/]{200,}={0,2}", part.text or "")
            for blob in blobs:
                base64_tokens += estimate_tokens(blob)
                base64_blocks += 1
            gaps = re.findall(r"\n{3,}|[ \t]{6,}", part.text or "")
            for gap in gaps:
                whitespace_tokens += count_tokens(gap, encoding)

    if pretty:
        items.append(
            {
                "id": "pretty-json",
                "label": "Pretty-printed JSON in tool output",
                "tokens": pretty,
                "detail": f"{pretty_blocks} blocks could be serialized compactly; indentation costs tokens on every turn",
            }
        )
    if whitespace_tokens:
        items.append(
            {
                "id": "whitespace",
                "label": "Runs of whitespace and blank lines",
                "tokens": whitespace_tokens,
                "detail": "Stack traces and HTML dumps carry alignment padding the model does not need",
            }
        )
    if base64_tokens:
        items.append(
            {
                "id": "base64",
                "label": "Inline base64 payloads",
                "tokens": base64_tokens,
                "detail": f"{base64_blocks} blobs of encoded data; move them to a URL or a file id",
            }
        )

    repeated = _repeated_instructions(messages, encoding)
    if repeated["tokens"]:
        items.append(
            {
                "id": "repeat-instructions",
                "label": "Constraints restated across messages",
                "tokens": repeated["tokens"],
                "detail": f"{repeated['sentences']} sentences repeat verbatim: {repeated['example']}",
            }
        )
    items.sort(key=lambda item: item["tokens"], reverse=True)
    return items


def _repeated_instructions(messages: list, encoding: str = "auto") -> dict:
    counter = Counter()
    originals = {}
    for message in messages:
        for part in message.parts:
            if part.kind not in ("system", "user"):
                continue
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", part.text or ""):
                words = sentence.split()
                if len(words) < 7:
                    continue
                key = re.sub(r"[^a-z0-9 ]", "", sentence.lower())
                key = re.sub(r"\s+", " ", key).strip()
                if len(key) < 24:
                    continue
                counter[key] += 1
                originals.setdefault(key, sentence.strip())
    wasted = 0
    hits = 0
    example = ""
    for key, count in counter.items():
        if count > 1:
            wasted += estimate_tokens(originals[key]) * (count - 1)
            hits += 1
            if not example:
                example = originals[key][:90]
    return {"tokens": wasted, "sentences": hits, "example": example}


def _trim_parts(messages: list, trims: set, encoding: str = "auto") -> list:
    parts = [(message.i, part.kind, part.text, part.tokens, message.synthetic) for message in messages for part in message.parts]

    if "squash_ws" in trims:
        out = []
        for i, kind, text, tokens, synthetic in parts:
            new = re.sub(r"\n{3,}", "\n\n", text)
            new = re.sub(r"[ \t]{6,}", " ", new)
            out.append((i, kind, new, count_tokens(new, encoding) if new else 0, synthetic))
        parts = out

    if "compact_json" in trims:
        out = []
        for i, kind, text, tokens, synthetic in parts:
            if kind in ("tool_result", "tool_call"):
                stripped = text.strip()
                if stripped.startswith(("{", "[")):
                    try:
                        parsed = json.loads(stripped)
                        text = json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
                        tokens = count_tokens(text, encoding)
                    except Exception:
                        pass
            out.append((i, kind, text, tokens, synthetic))
        parts = out

    if "truncate_tool" in trims:
        out = []
        for i, kind, text, tokens, synthetic in parts:
            if kind == "tool_result" and tokens > TRUNCATE_TOKENS:
                budget = int(len(text) * TRUNCATE_TOKENS / tokens)
                head = int(budget * 0.6)
                tail = max(0, budget - head)
                text = text[:head] + "\n[...truncated...]\n" + (text[-tail:] if tail else "")
                tokens = count_tokens(text, encoding)
            out.append((i, kind, text, tokens, synthetic))
        parts = out

    if "dedupe" in trims:
        seen = set()
        out = []
        for i, kind, text, tokens, synthetic in parts:
            if tokens >= MIN_DUPLICATE_TOKENS and text.strip():
                key = _block_hash(text)
                if key in seen:
                    continue
                seen.add(key)
            out.append((i, kind, text, tokens, synthetic))
        parts = out

    if "last_tools" in trims:
        tool_positions = [n for n, item in enumerate(parts) if item[1] == "tool_result"]
        keep = set(tool_positions[-3:])
        parts = [item for n, item in enumerate(parts) if item[1] != "tool_result" or n in keep]

    return parts


TRIMS = (
    ("dedupe", "Drop duplicate blocks", "low"),
    ("compact_json", "Serialize JSON tool output compactly", "low"),
    ("squash_ws", "Squash whitespace and blank lines", "low"),
    ("truncate_tool", f"Truncate each tool result to {TRUNCATE_TOKENS} tokens (head + tail)", "medium"),
    ("last_tools", "Keep only the last 3 tool results", "high"),
)


def compute_trims(messages: list, encoding: str = "auto") -> dict:
    baseline = sum(message.tokens for message in messages)
    rows = []
    for trim_id, label, risk in TRIMS:
        after = sum(part[3] for part in _trim_parts(messages, {trim_id}, encoding))
        rows.append(
            {
                "id": trim_id,
                "label": label,
                "risk": risk,
                "tokens": max(0, baseline - after),
                "pct": round(100 * max(0, baseline - after) / baseline, 1) if baseline else 0.0,
            }
        )

    combos = {}
    for mask in range(1 << len(TRIMS)):
        chosen = {TRIMS[bit][0] for bit in range(len(TRIMS)) if mask & (1 << bit)}
        after = sum(part[3] for part in _trim_parts(messages, chosen, encoding))
        combos[str(mask)] = after

    safe_mask = 0b00111  # dedupe + compact_json + squash_ws
    safe_after = combos[str(safe_mask)]
    vertices = {str(1 << bit): combos[str(1 << bit)] for bit in range(len(TRIMS))}
    return {
        "baseline": baseline,
        "rows": rows,
        "combos": combos,
        "vertices": vertices,
        "bits": {TRIMS[bit][0]: bit for bit in range(len(TRIMS))},
        "safe_tokens": max(0, baseline - safe_after),
        "safe_pct": round(100 * max(0, baseline - safe_after) / baseline, 1) if baseline else 0.0,
        "all_tokens": max(0, baseline - combos[str((1 << len(TRIMS)) - 1)]),
    }


def health_score(messages: list, kinds: dict, duplicate_waste: int, waste: list, total: int) -> dict:
    penalties = []

    def add(label: str, points: float, detail: str):
        if points >= 2:
            penalties.append({"label": label, "points": int(round(points)), "detail": detail})

    if total:
        dup_pct = 100 * duplicate_waste / total
        add("Duplicated context", min(30.0, max(0.0, dup_pct - 2.0) * 2.0), f"{dup_pct:.1f}% of the window is repeated blocks")

        result_pct = 100 * kinds.get("tool_result", 0) / total
        if result_pct > 45:
            add("Tool output dominates", min(15.0, (result_pct - 45) * 0.6), f"tool results are {result_pct:.0f}% of the window")

        json_waste = next((item["tokens"] for item in waste if item["id"] == "pretty-json"), 0)
        if json_waste:
            add("Verbose serialization", min(12.0, 100 * json_waste / total * 4), f"{json_waste:,} tokens of JSON indentation")

        schema_pct = 100 * kinds.get("tool_schema", 0) / total
        if schema_pct > 5:
            add("Tool schemas on every call", min(8.0, (schema_pct - 5) * 1.5), f"{schema_pct:.0f}% of every request is schema")

        non_synthetic = [message for message in messages if not message.synthetic]
        if non_synthetic:
            biggest = max(message.tokens for message in non_synthetic)
            share = 100 * biggest / total if total else 0
            if share > 25:
                add("One message dominates", min(12.0, (share - 25) * 0.6), f"a single message holds {share:.0f}% of the window")
        if len(non_synthetic) > 30:
            add("Very long history", min(10.0, (len(non_synthetic) - 30) * 0.25), f"{len(non_synthetic)} messages in one request")
        if total > 100_000:
            add("Large working set", 5.0, f"{total:,} tokens is past most recall sweet spots")

    score = max(0, min(100, 100 - sum(item["points"] for item in penalties)))
    grade = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D" if score >= 40 else "F"
    return {"value": int(score), "grade": grade, "penalties": penalties}


def analyze(messages: list, meta: dict, top: int = 10) -> dict:
    encoding = meta.get("encoding", "auto")
    total = sum(message.tokens for message in messages)
    chars = sum(len(part.text) for message in messages for part in message.parts)
    real = [message for message in messages if not message.synthetic]

    kind_tokens = Counter()
    kind_messages = Counter()
    for message in messages:
        for part in message.parts:
            kind_tokens[part.kind] += part.tokens
            if part.tokens:
                kind_messages[part.kind] += 1
    kinds = [
        {
            "kind": kind,
            "label": KIND_LABEL.get(kind, kind),
            "tokens": kind_tokens.get(kind, 0),
            "pct": round(100 * kind_tokens.get(kind, 0) / total, 1) if total else 0.0,
            "blocks": kind_messages.get(kind, 0),
        }
        for kind in KINDS
        if kind_tokens.get(kind, 0)
    ]
    kinds.sort(key=lambda row: row["tokens"], reverse=True)

    role_tokens = Counter()
    for message in messages:
        role_tokens[message.role] += message.tokens
    roles = [
        {"role": role, "tokens": value, "pct": round(100 * value / total, 1) if total else 0.0}
        for role, value in role_tokens.most_common()
    ]

    positional = [0] * 10
    for message in real:
        position = 0.0 if len(real) == 1 else (message.i / max(1, len(real) - 1))
        bucket = min(9, int(position * 10))
        positional[bucket] += message.tokens
    middle_share = 0.0
    if total:
        middle = sum(positional[3:7])
        body = sum(positional)
        middle_share = round(100 * middle / body, 1) if body else 0.0

    message_rows = []
    for message in messages:
        message_rows.append(
            {
                "index": message.i,
                "role": message.role,
                "kind": message.kind,
                "tokens": message.tokens,
                "pct": round(100 * message.tokens / total, 1) if total else 0.0,
                "synthetic": message.synthetic,
                "preview": message.preview(),
                "kinds": {part.kind: part.tokens for part in message.parts if part.tokens},
            }
        )

    top_rows = sorted(
        [row for row in message_rows if not row["synthetic"]],
        key=lambda row: row["tokens"],
        reverse=True,
    )[:top]

    duplicates, duplicate_waste = find_duplicates(messages)
    near = find_near_duplicates(messages)
    waste = find_waste(messages, encoding)
    trims = compute_trims(messages, encoding)
    score = health_score(messages, kind_tokens, duplicate_waste, waste, total)

    return {
        "version": __version__,
        "meta": {
            "source": meta.get("source", "stdin"),
            "format": meta.get("format", "unknown"),
            "model": meta.get("model"),
            "encoding": encoding_label(encoding),
            "requests": meta.get("requests", 1),
        },
        "totals": {
            "tokens": total,
            "chars": chars,
            "messages": len(real),
            "schemas": sum(1 for message in messages for part in message.parts if part.kind == "tool_schema"),
            "duplicate_tokens": duplicate_waste,
            "duplicate_blocks": sum(entry["count"] - 1 for entry in duplicates),
            "waste_tokens": sum(item["tokens"] for item in waste),
        },
        "kinds": kinds,
        "roles": roles,
        "positional": positional,
        "middle_share": middle_share,
        "messages": message_rows,
        "top": top_rows,
        "duplicates": duplicates[:12],
        "near_duplicates": near,
        "waste": waste,
        "trims": trims,
        "score": score,
        "dollars": _dollars(total),
    }


PRICES_PER_MTOK = {
    "frontier input": 3.00,
    "mid input": 0.60,
    "frontier output": 15.00,
}


def _dollars(tokens: int) -> dict:
    return {
        "per_1m_frontier": round(tokens / 1_000_000 * PRICES_PER_MTOK["frontier input"], 2),
        "per_1m_mid": round(tokens / 1_000_000 * PRICES_PER_MTOK["mid input"], 2),
        "per_1k_calls_frontier": round(tokens / 1_000_000 * PRICES_PER_MTOK["frontier input"] * 1000, 2),
    }


# ---------------------------------------------------------------------------
# 4. terminal report
# ---------------------------------------------------------------------------


class Palette:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __call__(self, code: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    def bold(self, text):
        return self("1", text)

    def dim(self, text):
        return self("2", text)

    def red(self, text):
        return self("31", text)

    def green(self, text):
        return self("32", text)

    def yellow(self, text):
        return self("33", text)

    def blue(self, text):
        return self("34", text)

    def magenta(self, text):
        return self("35", text)

    def cyan(self, text):
        return self("36", text)


def _fmt(number: int) -> str:
    return f"{number:,}"


def _bar(value: float, maximum: float, width: int = 22) -> str:
    if maximum <= 0:
        return ""
    filled = min(width, int(round(width * value / maximum)))
    whole = filled
    return "\u2588" * whole + "\u00b7" * (width - whole)


def _grade_color(pal: Palette, grade: str) -> str:
    if grade in ("A", "B"):
        return pal.green(grade)
    if grade == "C":
        return pal.yellow(grade)
    return pal.red(grade)


def render_terminal(data: dict, color: bool = True, top: int = 10) -> str:
    pal = Palette(color)
    meta = data["meta"]
    totals = data["totals"]
    total = totals["tokens"]
    lines = []

    title = pal.bold("ctxscope") + pal.dim(f" {data['version']}")
    source = meta["source"]
    model = f" / {meta['model']}" if meta.get("model") else ""
    lines.append(f"{title}  {source}{pal.dim(model)}")
    schemas = f"{totals['schemas']} tool schema" + ("s" if totals["schemas"] != 1 else "")
    lines.append(
        pal.dim(
            f"{meta['format']} / {totals['messages']} messages / {schemas} / tokens: {meta['encoding']}"
        )
    )
    lines.append("")

    grade = data["score"]["grade"]
    score = data["score"]["value"]
    safe = data["trims"]["safe_pct"]
    dollars = data["dollars"]
    lines.append(
        f"  {pal.bold('total')}           {pal.bold(_fmt(total))} tokens   {pal.dim(_fmt(totals['chars']) + ' chars')}"
        f"   {pal.dim(f'${dollars['per_1m_frontier']:.2f}/call at $3/Mtok')}"
    )
    lines.append(
        f"  {pal.bold('context health')}  {_grade_color(pal, grade)} {score}/100   "
        + pal.green(f"safe savings {safe}% ({_fmt(data['trims']['safe_tokens'])} tokens)")
    )
    lines.append("")

    lines.append(pal.bold("WHERE THE TOKENS GO"))
    largest = max((row["tokens"] for row in data["kinds"]), default=1)
    for row in data["kinds"]:
        lines.append(
            f"  {row['label']:<18}{_bar(row['tokens'], largest)}  {_fmt(row['tokens']):>8}  {row['pct']:>5.1f}%  {pal.dim(str(row['blocks']) + (' block' if row['blocks'] == 1 else ' blocks'))}"
        )
    lines.append("")

    if data["top"]:
        lines.append(pal.bold("BIGGEST MESSAGES"))
        biggest = max(row["tokens"] for row in data["top"])
        for row in data["top"][: min(top, 6)]:
            where = "#tools" if row["synthetic"] else f"#{row['index']}"
            preview = row["preview"][:78]
            lines.append(f"  {where:<7}{row['role']:<10}{_bar(row['tokens'], biggest, 14)}  {_fmt(row['tokens']):>8}  {pal.dim(preview)}")
        lines.append("")

    if data["duplicates"]:
        lines.append(pal.bold(f"DUPLICATES  ") + pal.dim(f"{_fmt(totals['duplicate_tokens'])} wasted tokens"))
        for row in data["duplicates"][:5]:
            lines.append(
                f"  x{row['count']:<3}{_fmt(row['tokens']):>7} tokens each  {_fmt(row['wasted']):>7} wasted  {pal.dim(row['preview'][:64])}"
            )
        lines.append("")

    if data["waste"]:
        lines.append(pal.bold("STRUCTURAL WASTE"))
        for item in data["waste"][:5]:
            lines.append(f"  {_fmt(item['tokens']):>7}  {item['label']}")
            lines.append(pal.dim(f"           {item['detail'][:100]}"))
        lines.append("")

    lines.append(pal.bold("TRIM SIMULATOR") + pal.dim("  (independent, against the full window)"))
    width = max([len(row["label"]) for row in data["trims"]["rows"]] + [len("all three low-risk trims")]) + 2
    for row in data["trims"]["rows"]:
        risk = {"low": pal.green("low"), "medium": pal.yellow("medium"), "high": pal.red("high")}[row["risk"]]
        lines.append(f"  {row['label']:<{width}}{_fmt(row['tokens']):>8}  {row['pct']:>5.1f}%  {risk}")
    lines.append(
        "  "
        + pal.bold(f"{'all three low-risk trims':<{width}}")
        + pal.bold(f"{_fmt(data['trims']['safe_tokens']):>8}  {data['trims']['safe_pct']:>5.1f}%")
    )
    lines.append("")

    if data["score"]["penalties"]:
        lines.append(pal.bold("WHY THE SCORE IS NOT 100"))
        for item in data["score"]["penalties"]:
            lines.append(f"  -{item['points']:<3} {item['label']:<28} {pal.dim(item['detail'])}")
        lines.append("")

    lines.append(pal.bold("NEXT"))
    for advice in recommendations(data):
        lines.append(f"  - {advice}")
    lines.append("")

    middle = data["middle_share"]
    if middle:
        lines.append(
            pal.dim(
                f"{middle}% of the window sits in the middle half of the conversation, the band models recall worst. "
                f"Move stable facts to the top and the task to the bottom."
            )
        )
    return "\n".join(lines)


def recommendations(data: dict) -> list:
    out = []
    duplicates = data["duplicates"]
    if duplicates:
        top = duplicates[0]
        out.append(
            f"{top['count']} identical copies of a {_fmt(top['tokens'])}-token block. Cache tool results by argument hash - the agent is re-running the same call."
        )
    for item in data["waste"]:
        if item["id"] == "pretty-json":
            out.append(f"Compact JSON in tool output saves {_fmt(item['tokens'])} tokens with no information loss.")
            break
    tool_pct = next((row["pct"] for row in data["kinds"] if row["kind"] == "tool_result"), 0.0)
    if tool_pct > 40:
        out.append(
            f"Tool results are {tool_pct:.0f}% of the window. Return summaries by default and full payloads only on request."
        )
    schema_pct = next((row["pct"] for row in data["kinds"] if row["kind"] == "tool_schema"), 0.0)
    if schema_pct > 5:
        out.append(
            f"Tool schemas cost {schema_pct:.0f}% of every single call. Split rarely used tools into a second agent."
        )
    if data["totals"]["messages"] > 30:
        out.append("History is long: summarize completed turns instead of replaying every tool result verbatim.")
    if not out:
        out.append("Nothing structural to fix. Keep the working set under 50k tokens and cache the stable prefix.")
    return out


# ---------------------------------------------------------------------------
# 5. markdown report
# ---------------------------------------------------------------------------


def render_markdown(data: dict) -> str:
    meta, totals = data["meta"], data["totals"]
    lines = [
        f"# ctxscope report: {meta['source']}",
        "",
        f"- format: `{meta['format']}`" + (f" / model: `{meta['model']}`" if meta.get("model") else ""),
        f"- tokens: **{_fmt(totals['tokens'])}** ({meta['encoding']})",
        f"- messages: {totals['messages']} / tool schemas: {totals['schemas']}",
        f"- context health: **{data['score']['value']}/100 ({data['score']['grade']})**",
        f"- duplicate waste: {_fmt(totals['duplicate_tokens'])} tokens",
        f"- safe savings: **{data['trims']['safe_pct']}%** ({_fmt(data['trims']['safe_tokens'])} tokens)",
        "",
        "## Composition",
        "",
        "| Bucket | Tokens | Share | Blocks |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in data["kinds"]:
        lines.append(f"| {row['label']} | {_fmt(row['tokens'])} | {row['pct']}% | {row['blocks']} |")
    lines += ["", "## Biggest messages", "", "| # | Role | Tokens | Share | Preview |", "| ---: | --- | ---: | ---: | --- |"]
    for row in data["top"]:
        preview = row["preview"].replace("|", "\\|")[:90]
        lines.append(f"| {row['index']} | {row['role']} | {_fmt(row['tokens'])} | {row['pct']}% | {preview} |")
    if data["duplicates"]:
        lines += ["", "## Duplicates", "", "| Count | Tokens each | Wasted | Preview |", "| ---: | ---: | ---: | --- |"]
        for row in data["duplicates"]:
            preview = row["preview"].replace("|", "\\|")[:90]
            lines.append(f"| {row['count']} | {_fmt(row['tokens'])} | {_fmt(row['wasted'])} | {preview} |")
    if data["waste"]:
        lines += ["", "## Structural waste", ""]
        for item in data["waste"]:
            lines.append(f"- **{_fmt(item['tokens'])} tokens**: {item['label']} - {item['detail']}")
    lines += ["", "## Trim simulator", "", "| Trim | Tokens saved | Share | Risk |", "| --- | ---: | ---: | --- |"]
    for row in data["trims"]["rows"]:
        lines.append(f"| {row['label']} | {_fmt(row['tokens'])} | {row['pct']}% | {row['risk']} |")
    lines += ["", "## Recommendations", ""]
    for advice in recommendations(data):
        lines.append(f"- {advice}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. self-contained HTML report
# ---------------------------------------------------------------------------

VIEWER_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ctxscope report</title>
<style>
:root{
  color-scheme: dark;
  --bg:#0a0c10; --panel:#11141a; --panel2:#161a22; --line:#232833; --line2:#2d3442;
  --text:#e9ecf1; --muted:#98a1b2; --faint:#7b8499;
  --accent:#7dd3fc; --good:#4ade80; --warn:#fbbf24; --bad:#f87171;
  --k0:#6ea8fe; --k1:#b197fc; --k2:#e879f9; --k3:#ffd43b; --k4:#63e6be; --k5:#ffa94d; --k6:#fda4af; --k7:#94a3b8;
  --pad:18px;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{
  background:var(--bg); color:var(--text);
  font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility;
}
.mono,td.n,.stat .v{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
.wrap{max-width:1000px;margin:0 auto;padding:34px 22px 40px}

header{display:grid;grid-template-columns:1fr auto;gap:24px 40px;align-items:center;margin-bottom:30px}
.wordmark{font-size:10.5px;letter-spacing:.2em;text-transform:uppercase;color:var(--faint);margin-bottom:12px;font-weight:600}
h1{font-size:19px;font-weight:600;margin:0 0 10px;letter-spacing:-.01em;word-break:break-word;line-height:1.3}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{font:11px/1 ui-monospace,Menlo,monospace;color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:5px 9px;background:var(--panel)}
.hero-stats{grid-column:1/-1;display:flex;justify-content:space-between;gap:28px;flex-wrap:wrap;margin-top:2px}
.stat .k{font-size:10px;letter-spacing:.11em;text-transform:uppercase;color:var(--faint);margin-bottom:5px;font-weight:600}
.stat .v{font-size:34px;line-height:1.05;letter-spacing:-.025em}
.stat .v.good{color:var(--good)}
.stat .s{font-size:11.5px;color:var(--faint);margin-top:4px}

.ring{position:relative;width:148px;height:148px;flex:0 0 auto}
.ring svg{transform:rotate(-90deg);display:block}
.ring .val{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:3px}
.ring .n{font:700 38px/1 ui-monospace,Menlo,monospace;letter-spacing:-.03em}
.ring .of{font-size:13px;font-weight:500;opacity:.5;letter-spacing:0;margin-left:1px}
.ring .gchip{font:600 9.5px/1 ui-monospace,Menlo,monospace;letter-spacing:.14em;text-transform:uppercase;padding:4px 9px;border-radius:999px;margin-top:6px}
.ring .g{font-size:9.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--faint);font-weight:600}

section{margin:40px 0 0}
h2{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin:0 0 12px;font-weight:600;display:flex;justify-content:space-between;align-items:baseline;gap:14px}
h2 .note{letter-spacing:0;text-transform:none;font-size:11.5px;color:var(--faint);font-weight:400;text-align:right}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:var(--pad)}
.callout{font-size:12.5px;color:var(--muted);margin-bottom:12px}
.callout b{color:var(--text);font-weight:600}

.stack{display:flex;height:11px;border-radius:6px;overflow:hidden;background:var(--line);gap:2px;margin-bottom:18px}
.stack i{display:block;height:100%;min-width:2px}
.stack i:first-child{border-radius:6px 0 0 6px}
.stack i:last-child{border-radius:0 6px 6px 0}

table{width:100%;border-collapse:collapse;table-layout:fixed}
th{text-align:left;font-size:10.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--faint);font-weight:600;padding:0 0 9px}
td{padding:7px 0;border-top:1px solid var(--line);vertical-align:middle;font-size:13px}
td.n,th.n{text-align:right}
th.c-tokens,td.c-tokens{width:96px;padding-right:12px}
th.c-share,td.c-share{width:78px;padding-left:12px}
th.c-blocks,td.c-blocks{width:104px}
.bars{display:flex;align-items:center;justify-content:flex-end;gap:9px}
.bars .track{flex:0 0 46px;height:4px;border-radius:2px;background:var(--line2);overflow:hidden}
.bars .track i{display:block;height:100%;border-radius:2px}
.bars span{min-width:2ch;color:var(--faint);font-size:11.5px}
.sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:9px;vertical-align:0}
.track3{height:3px;border-radius:2px;background:var(--line2);margin-top:5px;overflow:hidden;width:100%}
.track3 i{display:block;height:100%;border-radius:2px}
.sharecell{display:flex;flex-direction:column;justify-content:center;align-items:flex-end;gap:4px}

.legend{display:flex;align-items:center;gap:7px;font-size:11px;color:var(--faint);margin-bottom:10px}
.legend .box{width:9px;height:9px;border-radius:2px;background:var(--accent);opacity:.85}
.histo{display:flex;align-items:flex-end;gap:6px;height:132px;padding-top:4px}
.histo .col{flex:1;display:flex;flex-direction:column;justify-content:flex-end;height:100%;gap:7px}
.histo .col .bar{border-radius:2px;background:var(--line2)}
.histo .col .bar.mid{background:var(--accent);opacity:.85}
.histo .col .lbl{font:10px/1 ui-monospace,Menlo,monospace;color:var(--faint);text-align:center}

.rows{display:flex;flex-direction:column}
.row{display:grid;grid-template-columns:64px 54px 1fr 92px;gap:14px;align-items:center;padding:9px 0;border-top:1px solid var(--line);font-size:12.5px}
.row:first-child{border-top:0}
.row > *{min-width:0}
.tag{font:10.5px/1.35 ui-monospace,Menlo,monospace;color:var(--muted);border:1px solid var(--line);border-radius:5px;padding:4px 5px;text-align:center;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tag.role{color:var(--faint)}
.preview{color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
.track2{height:3px;border-radius:2px;background:var(--line2);margin-top:6px;overflow:hidden}
.track2 i{display:block;height:100%;border-radius:2px}
.num{text-align:right}
.num .t{font:12.5px/1.2 ui-monospace,Menlo,monospace}
.num .s{font:11px/1.2 ui-monospace,Menlo,monospace;color:var(--faint);margin-top:2px}

.dupes{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.dcard{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:15px 16px}
.dhead{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:9px;gap:10px}
.dcount{font:600 11.5px/1 ui-monospace,Menlo,monospace;color:var(--warn)}
.dmeta{font:10.5px/1 ui-monospace,Menlo,monospace;color:var(--faint)}
.dbody{font:11.5px/1.5 ui-monospace,Menlo,monospace;color:var(--muted);display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden;word-break:break-word}
.dcard.empty{grid-column:1/-1;color:var(--muted)}

.wrow{display:flex;justify-content:space-between;gap:20px;padding:10px 0;border-top:1px solid var(--line);font-size:13px}
.wrow:first-child{border-top:0;padding-top:0}
.wrow:last-child{padding-bottom:0}
.wdetail{color:var(--faint);font-size:11.5px;margin-top:3px;line-height:1.45}
.wrow .amt{font:13px/1 ui-monospace,Menlo,monospace;white-space:nowrap}

.trims .trow{display:grid;grid-template-columns:18px 1fr 90px 74px;gap:14px;align-items:center;padding:11px 0;border-top:1px solid var(--line);font-size:13px}
.trims .trow:first-child{border-top:0}
.trims .trow > *{min-width:0}
input[type=checkbox]{appearance:none;width:15px;height:15px;border:1px solid var(--line2);border-radius:4px;background:var(--panel2);cursor:pointer;position:relative;margin:0}
input[type=checkbox]:checked{background:var(--accent);border-color:var(--accent)}
input[type=checkbox]:checked:after{content:"";position:absolute;left:3.5px;top:1px;width:5px;height:8px;border:2px solid #0a0c10;border-top:0;border-left:0;transform:rotate(40deg)}
.tsave{font:12.5px/1 ui-monospace,Menlo,monospace;text-align:right;color:var(--muted)}
.pill{font:10px/1 ui-monospace,Menlo,monospace;text-transform:uppercase;letter-spacing:.06em;padding:4px 7px;border-radius:5px;display:inline-block;width:100%;text-align:center}
.pill.low{color:#86efac;background:rgba(74,222,128,.10)}
.pill.medium{color:#fcd34d;background:rgba(251,191,36,.10)}
.pill.high{color:#fca5a5;background:rgba(248,113,113,.10)}
.after{display:flex;justify-content:space-between;align-items:flex-end;gap:20px;margin-top:18px;padding-top:16px;border-top:1px solid var(--line2)}
.after .k{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--faint);margin-bottom:5px;font-weight:600}
.after .big{font:600 24px/1.1 ui-monospace,Menlo,monospace;letter-spacing:-.02em}
.after .big.good{color:var(--good)}

ul.next{margin:0;padding-left:17px}
ul.next li{padding:6px 0;color:var(--muted);font-size:13.5px;line-height:1.5}
ul.next li b{color:var(--text);font-weight:600}
footer{margin-top:38px;padding-top:16px;border-top:1px solid var(--line);color:var(--faint);font-size:11.5px;display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;align-items:center}
.btns{display:flex;gap:8px}
button{font:12px/1 inherit;color:var(--text);background:var(--panel);border:1px solid var(--line2);border-radius:7px;padding:8px 12px;cursor:pointer}
button:hover{border-color:var(--faint)}
#drop{position:fixed;inset:0;background:rgba(4,6,9,.84);display:none;align-items:center;justify-content:center;z-index:20}
#drop.on{display:flex}
#drop .box{border:1px dashed var(--line2);border-radius:14px;padding:42px 52px;text-align:center;background:var(--panel)}

@media (max-width:480px){
  .ring{width:84px;height:84px}
  .ring svg{width:84px;height:84px}
  .ring .n{font-size:23px}
  .ring .of{font-size:9.5px}
  .ring .gchip{font-size:7.5px;padding:3px 5px;letter-spacing:.06em}
  .hero-stats{gap:10px}
  .stat .v{font-size:17px}
}
@media (max-width:760px){
  header{grid-template-columns:auto 1fr;gap:18px;align-items:center}
  .ring{width:96px;height:96px;order:-1}
  .ring svg{width:96px;height:96px}
  .ring .n{font-size:26px}
  .ring .of{font-size:11px}
  .ring .gchip{font-size:8px;letter-spacing:.08em;padding:3px 6px;margin-top:4px}
  .hero-stats{flex-wrap:nowrap;gap:14px;justify-content:space-between}
  .stat .v{font-size:19px}
  .stat .k{font-size:9px;letter-spacing:.06em}
  .dupes{grid-template-columns:1fr}
  .row{grid-template-columns:56px 1fr 76px}
  .row .tag.role{display:none}
  .preview{white-space:normal;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}
  td.c-blocks,th.c-blocks{display:none}
  .histo{height:96px}
  .trims .trow{grid-template-columns:18px 1fr 74px;gap:10px}
  .trims .trow .riskcol{display:none}
  table{table-layout:auto}
}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <div class="wordmark">ctxscope</div>
      <h1 id="source"></h1>
      <div class="chips" id="chips"></div>
    </div>
    <div class="ring" id="ring"></div>
    <div class="hero-stats">
      <div class="stat"><div class="k">tokens in context</div><div class="v" id="hero-tokens">—</div><div class="s" id="hero-call"></div></div>
      <div class="stat"><div class="k">duplicate waste</div><div class="v" id="hero-dup">—</div><div class="s" id="hero-dup-s"></div></div>
      <div class="stat"><div class="k">safe savings</div><div class="v good" id="hero-save">—</div><div class="s" id="hero-save-s"></div></div>
    </div>
  </header>

  <section id="s-composition">
    <h2>Where the tokens go <span class="note" id="total-note"></span></h2>
    <div class="card">
      <div class="callout" id="callout"></div>
      <div class="stack" id="stack"></div>
      <table id="kinds"></table>
    </div>
  </section>

  <section id="s-position">
    <h2>Position in the window <span class="note" id="middle-note"></span></h2>
    <div class="card"><div class="legend"><span class="box"></span>middle half of the conversation, where recall is worst</div><div class="histo" id="position"></div></div>
  </section>

  <section id="s-messages">
    <h2>Biggest messages <span class="note">what is actually in there</span></h2>
    <div class="card"><div class="rows" id="rows"></div></div>
  </section>

  <section id="s-dupes">
    <h2>Duplicate context <span class="note" id="dupe-note"></span></h2>
    <div class="dupes" id="dupes"></div>
  </section>

  <section id="s-waste">
    <h2>Structural waste <span class="note">tokens that buy no information</span></h2>
    <div class="card" id="waste"></div>
  </section>

  <section id="s-trims">
    <h2>Trim simulator <span class="note">low-risk trims are already selected</span></h2>
    <div class="card trims" id="trims"></div>
  </section>

  <section id="s-next">
    <h2>Next <span class="note">the shortest path out</span></h2>
    <div class="card"><ul class="next" id="next"></ul></div>
  </section>

  <footer>
    <span id="foot-left">generated by ctxscope</span>
    <span class="btns"><button id="again">Open another trace</button><button id="copy">Copy summary</button></span>
  </footer>
</div>

<div id="drop"><div class="box"><div style="font-size:15px;margin-bottom:6px">Drop a request, trace, or JSONL log</div><div style="color:var(--muted);font-size:12.5px">OpenAI, Anthropic, or a raw messages array. Nothing leaves this file.</div></div></div>
<input type="file" id="file" accept=".json,.jsonl,.txt" style="display:none">

<script>
const EMBEDDED = __CTXRAY_DATA__;
const KIND_LABEL = {tool_result:"tool results",system:"system prompt",tool_schema:"tool schemas",tool_call:"tool calls",user:"user turns",assistant:"assistant turns",image:"images",other:"other"};
const KINDS = ["tool_result","system","tool_schema","tool_call","user","assistant","image","other"];
const KIND_VAR = {tool_result:"--k0",system:"--k1",tool_schema:"--k2",tool_call:"--k3",user:"--k4",assistant:"--k5",image:"--k6",other:"--k7"};
const MIN_DUPLICATE_TOKENS = 48;
const TRIMS_META = [
  ["dedupe","Drop duplicate blocks","low"],
  ["compact_json","Serialize JSON tool output compactly","low"],
  ["squash_ws","Squash whitespace and blank lines","low"],
  ["truncate_tool","Truncate each tool result to 2000 tokens","medium"],
  ["last_tools","Keep only the last 3 tool results","high"]
];
const IMAGE_TOKENS = 1105;

const fmt = n => (n==null?"0":Math.round(n).toLocaleString("en-US"));
const pctf = (a,b) => b ? +(100*a/b).toFixed(1) : 0;
const esc = s => String(s==null?"":s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const varOf = k => "var(" + (KIND_VAR[k]||"--k7") + ")";

/* ---- token estimator, ported from the Python implementation ---- */
const PIECE = /[A-Za-z]+|\d+|[^\sA-Za-z\d]|\s+/g;
function estimateTokens(text){
  if(!text) return 0;
  let total = 0;
  const m = String(text).match(PIECE);
  if(!m) return 0;
  for(const piece of m){
    const head = piece[0];
    if(/\s/.test(head)){
      const nl = (piece.match(/\n/g)||[]).length;
      const sp = piece.length - nl;
      if(nl) total += 0.55 * nl;
      if(sp >= 4) total += 0.9 * (sp/4);
      continue;
    }
    if(/[A-Za-z]/.test(head)){
      total += 1;
      if(piece.length > 7) total += 1;
      if(piece.length > 12) total += 1;
    } else if(/\d/.test(head)){
      total += Math.max(1, (piece.length + 2) / 3);
    } else {
      total += 1;
    }
  }
  return Math.max(1, Math.round(total));
}
const elemTokens = text => text ? Math.max(1, Math.round(estimateTokens(text))) : 0;

/* ---- normalization ---- */
function roleKind(role){
  role = String(role||"").toLowerCase();
  if(role==="system"||role==="developer") return "system";
  if(role==="tool"||role==="function") return "tool_result";
  if(role==="assistant") return "assistant";
  if(role==="user") return "user";
  return "other";
}
function mkPart(kind, text){
  text = String(text||"");
  return {kind:KINDS.indexOf(kind)>=0?kind:"other", text:text, tokens:elemTokens(text)};
}
function partsOf(content, role){
  const out = [];
  if(content==null) return out;
  if(typeof content==="string"||typeof content==="number"||typeof content==="boolean") return [mkPart(roleKind(role), String(content))];
  if(!Array.isArray(content)) content = [content];
  for(const block of content){
    if(typeof block==="string"){ out.push(mkPart(roleKind(role), block)); continue; }
    if(!block||typeof block!=="object"){ out.push(mkPart("other", JSON.stringify(block))); continue; }
    const t = String(block.type||"").toLowerCase();
    if(t==="text"||t==="input_text"||t==="output_text") out.push(mkPart(roleKind(role), block.text||""));
    else if(t==="image"||t==="image_url"||t==="input_image") out.push(mkPart("image", "[image " + IMAGE_TOKENS + " tokens]"));
    else if(t==="tool_use"||t==="tool_call"||t==="function_call"){
      const name = block.name || (block.function&&block.function.name) || "?";
      const payload = block.input!=null ? block.input : block.arguments;
      out.push(mkPart("tool_call", name + "(" + JSON.stringify(payload) + ")"));
    }
    else if(t==="tool_result"||t==="function_result"){
      let inner = block.content;
      if(Array.isArray(inner)) inner = inner.map(x=> (x&&x.text)||"").join("\n");
      out.push(mkPart("tool_result", typeof inner==="string" ? inner : JSON.stringify(inner)));
    }
    else out.push(mkPart("other", JSON.stringify(block)));
  }
  return out;
}
function fnSchema(tool){
  if(tool && tool.function) tool = tool.function;
  return mkPart("tool_schema", JSON.stringify(tool));
}
function normalize(obj){
  if(Array.isArray(obj)) obj = (obj.length && obj[0] && obj[0].role) ? {messages:obj} : {messages:[{role:"user", content:JSON.stringify(obj)}]};
  if(!obj||typeof obj!=="object") obj = {messages:[{role:"user", content:String(obj)}]};
  const meta = {format:"unknown", model:obj.model||null, requests:1};
  let raw = null, system = null;
  if(obj.system){ system = obj.system; meta.format = "anthropic"; }
  for(const key of ["messages","conversation","history","chat"]){
    if(Array.isArray(obj[key])){ raw = obj[key]; if(key==="messages"||key==="chat") meta.format = meta.format==="anthropic"?"anthropic":"openai"; break; }
  }
  if(!raw && Array.isArray(obj.input) && obj.input.length && obj.input[0] && obj.input[0].role){ raw = obj.input; meta.format="openai-responses"; }
  if(!raw){ raw = [{role:"user", content:JSON.stringify(obj)}]; meta.format = "raw"; }

  const messages = [];
  let idx = 0;
  if(system){
    const parts = Array.isArray(system) ? partsOf(system,"system") : [mkPart("system", String(system))];
    messages.push({i:idx++, role:"system", parts, synthetic:""});
  }
  for(const m of raw){
    if(!m||typeof m!=="object"){ messages.push({i:idx++, role:"user", parts:[mkPart("user", String(m))], synthetic:""}); continue; }
    const role = String(m.role||m.author||"unknown").toLowerCase();
    const parts = partsOf(m.content, role);
    let calls = m.tool_calls || m.function_call;
    if(calls){
      if(!Array.isArray(calls)) calls = [calls];
      for(const call of calls){
        const fn = call.function || call;
        if(!fn) continue;
        parts.push(mkPart("tool_call", (fn.name||"?") + "(" + (typeof fn.arguments==="string" ? fn.arguments : JSON.stringify(fn.arguments)) + ")"));
      }
    }
    messages.push({i:idx++, role, parts, synthetic:""});
  }
  const tools = Array.isArray(obj.tools) ? obj.tools : (Array.isArray(obj.functions) ? obj.functions : null);
  if(tools && tools.length) messages.unshift({i:-1, role:"tools", parts:tools.map(fnSchema), synthetic:"tool_schemas"});
  return {messages, meta};
}

/* ---- analysis ---- */
const normBlock = t => String(t).replace(/\s+/g," ").trim();
function hashOf(t){
  let h = 2166136261;
  const s = normBlock(t);
  for(let i=0;i<s.length;i++){ h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return (h>>>0).toString(16).padStart(8,"0");
}
function duplicatesOf(messages){
  const map = new Map();
  for(const m of messages) for(const p of m.parts){
    if(p.tokens < MIN_DUPLICATE_TOKENS || !p.text.trim()) continue;
    const key = hashOf(p.text);
    if(!map.has(key)) map.set(key, {hash:key, kind:p.kind, tokens:p.tokens, count:0, indices:[], text:p.text});
    const e = map.get(key); e.count++; e.indices.push(m.i);
  }
  const out = [];
  for(const e of map.values()) if(e.count>1){
    e.wasted = e.tokens * (e.count-1);
    e.preview = String(e.text).replace(/\s+/g," ").trim().slice(0,220);
    delete e.text;
    out.push(e);
  }
  out.sort((a,b)=>b.wasted-a.wasted);
  return out;
}
function prettyJsonCost(text){
  const s = String(text).trim();
  if(!(s.startsWith("{")||s.startsWith("["))) return 0;
  try{ const parsed = JSON.parse(s); return Math.max(0, elemTokens(s) - elemTokens(JSON.stringify(parsed))); }catch(e){ return 0; }
}
function wasteOf(messages){
  const out = [];
  let pretty = 0, prettyBlocks = 0, b64 = 0, b64Blocks = 0, ws = 0;
  for(const m of messages) for(const p of m.parts){
    if(p.kind==="tool_result"||p.kind==="tool_call"){
      const c = prettyJsonCost(p.text);
      if(c>0){ pretty += c; prettyBlocks++; }
    }
    const blobs = String(p.text).match(/[A-Za-z0-9+/]{200,}={0,2}/g);
    if(blobs){ for(const b of blobs){ b64 += elemTokens(b); b64Blocks++; } }
    const gaps = String(p.text).match(/\n{3,}|[ \t]{6,}/g);
    if(gaps){ for(const g of gaps) ws += elemTokens(g); }
  }
  if(pretty) out.push({id:"pretty-json", label:"Pretty-printed JSON in tool output", tokens:pretty, detail:prettyBlocks + " blocks could be serialized compactly; indentation costs tokens on every turn"});
  if(ws) out.push({id:"whitespace", label:"Runs of whitespace and blank lines", tokens:ws, detail:"Alignment padding and blank lines the model does not need"});
  if(b64) out.push({id:"base64", label:"Inline base64 payloads", tokens:b64, detail:b64Blocks + " blobs of encoded data; move them to a URL or a file id"});
  out.sort((a,b)=>b.tokens-a.tokens);
  return out;
}
function flatParts(messages){
  const out = [];
  for(const m of messages) for(const p of m.parts) out.push({i:m.i, kind:p.kind, text:p.text, tokens:p.tokens, synthetic:m.synthetic});
  return out;
}
function trimParts(parts, chosen){
  let work = parts.slice();
  if(chosen.has("squash_ws")) work = work.map(p=>{ const t = p.text.replace(/\n{3,}/g,"\n\n").replace(/[ \t]{6,}/g," "); return {...p, text:t, tokens:elemTokens(t)}; });
  if(chosen.has("compact_json")) work = work.map(p=>{
    if(p.kind==="tool_result"||p.kind==="tool_call"){
      const s = String(p.text).trim();
      if(s.startsWith("{")||s.startsWith("[")){ try{ const c = JSON.stringify(JSON.parse(s)); return {...p, text:c, tokens:elemTokens(c)}; }catch(e){ return p; } }
    }
    return p;
  });
  if(chosen.has("truncate_tool")) work = work.map(p=>{
    if(p.kind==="tool_result" && p.tokens>2000){
      const budget = Math.max(16, Math.floor(p.text.length*2000/p.tokens));
      const head = Math.floor(budget*0.6);
      const t = p.text.slice(0,head) + "\n[...truncated...]\n" + p.text.slice(-(budget-head));
      return {...p, text:t, tokens:elemTokens(t)};
    }
    return p;
  });
  if(chosen.has("dedupe")){
    const seen = new Set(); const out = [];
    for(const p of work){
      if(p.tokens>=MIN_DUPLICATE_TOKENS && p.text.trim()){
        const k = hashOf(p.text);
        if(seen.has(k)) continue;
        seen.add(k);
      }
      out.push(p);
    }
    work = out;
  }
  if(chosen.has("last_tools")){
    const positions = work.map((p,n)=>p.kind==="tool_result"?n:-1).filter(n=>n>=0);
    const keep = new Set(positions.slice(-3));
    work = work.filter((p,n)=> p.kind!=="tool_result" || keep.has(n));
  }
  return work;
}
function trimsOf(messages){
  const parts = flatParts(messages);
  const baseline = parts.reduce((a,p)=>a+p.tokens, 0);
  const rows = TRIMS_META.map(([id,label,risk])=>{
    const after = trimParts(parts, new Set([id])).reduce((a,p)=>a+p.tokens,0);
    return {id, label, risk, tokens:Math.max(0,baseline-after), pct:pctf(Math.max(0,baseline-after), baseline)};
  });
  const combos = {};
  for(let mask=0; mask<32; mask++){
    const chosen = new Set(TRIMS_META.filter((_,bit)=>mask&(1<<bit)).map(t=>t[0]));
    combos[String(mask)] = trimParts(parts, chosen).reduce((a,p)=>a+p.tokens,0);
  }
  return {
    baseline, rows, combos,
    safe_tokens: Math.max(0, baseline-combos["7"]),
    safe_pct: pctf(Math.max(0, baseline-combos["7"]), baseline),
    all_tokens: Math.max(0, baseline-combos["31"])
  };
}
function scoreOf(messages, kindTokens, dupWaste, waste, total){
  const penalties = [];
  const add = (label, points, detail) => { if(points>=2) penalties.push({label, points:Math.round(points), detail}); };
  if(total){
    const dupPct = 100*dupWaste/total;
    add("Duplicated context", Math.min(30, Math.max(0,dupPct-2)*2), dupPct.toFixed(1)+"% of the window is repeated blocks");
    const resPct = 100*(kindTokens.tool_result||0)/total;
    if(resPct>45) add("Tool output dominates", Math.min(15,(resPct-45)*0.6), "tool results are "+Math.round(resPct)+"% of the window");
    const jw = waste.filter(w=>w.id==="pretty-json").reduce((a,w)=>a+w.tokens,0);
    if(jw) add("Verbose serialization", Math.min(12, 100*jw/total*4), fmt(jw)+" tokens of JSON indentation");
    const sp = 100*(kindTokens.tool_schema||0)/total;
    if(sp>5) add("Tool schemas on every call", Math.min(8,(sp-5)*1.5), Math.round(sp)+"% of every request is schema");
    const real = messages.filter(m=>!m.synthetic);
    if(real.length){
      const biggest = Math.max(...real.map(m=>m.parts.reduce((a,p)=>a+p.tokens,0)));
      const share = total?100*biggest/total:0;
      if(share>25) add("One message dominates", Math.min(12,(share-25)*0.6), "a single message holds "+Math.round(share)+"% of the window");
    }
    if(real.length>30) add("Very long history", Math.min(10,(real.length-30)*0.25), real.length+" messages in one request");
    if(total>100000) add("Large working set", 5, fmt(total)+" tokens is past most recall sweet spots");
  }
  const value = Math.max(0, Math.min(100, 100 - penalties.reduce((a,p)=>a+p.points,0)));
  const grade = value>=85?"A":value>=70?"B":value>=55?"C":value>=35?"D":"F";
  return {value:Math.round(value), grade, penalties};
}
function analyzeInput(obj, name){
  const {messages, meta} = normalize(obj);
  const parts = flatParts(messages);
  const total = parts.reduce((a,p)=>a+p.tokens,0);
  const chars = parts.reduce((a,p)=>a+(p.text||"").length,0);
  const kindTokens = {}, kindBlocks = {};
  for(const p of parts){ kindTokens[p.kind]=(kindTokens[p.kind]||0)+p.tokens; if(p.tokens) kindBlocks[p.kind]=(kindBlocks[p.kind]||0)+1; }
  const kinds = KINDS.filter(k=>kindTokens[k]).map(k=>({kind:k, label:KIND_LABEL[k], tokens:kindTokens[k], pct:pctf(kindTokens[k],total), blocks:kindBlocks[k]||0})).sort((a,b)=>b.tokens-a.tokens);
  const roleTokens = {};
  for(const m of messages) roleTokens[m.role]=(roleTokens[m.role]||0)+(m.parts.reduce((a,p)=>a+p.tokens,0));
  const roles = Object.keys(roleTokens).map(r=>({role:r, tokens:roleTokens[r], pct:pctf(roleTokens[r],total)})).sort((a,b)=>b.tokens-a.tokens);
  const real = messages.filter(m=>!m.synthetic);
  const positional = new Array(10).fill(0);
  for(const m of real){
    const pos = real.length===1?0:(m.i/(Math.max(1,real.length-1)));
    positional[Math.min(9,Math.floor(pos*10))] += m.parts.reduce((a,p)=>a+p.tokens,0);
  }
  const body = positional.reduce((a,b)=>a+b,0);
  const middle = body?+(100*positional.slice(3,7).reduce((a,b)=>a+b,0)/body).toFixed(1):0;
  const messageRows = messages.map(m=>{
    const t = m.parts.reduce((a,p)=>a+p.tokens,0);
    const km = {}; for(const p of m.parts) if(p.tokens) km[p.kind]=(km[p.kind]||0)+p.tokens;
    const dominant = m.parts.length?m.parts.reduce((a,p)=>p.tokens>a.tokens?p:a).kind:"other";
    return {index:m.i, role:m.role, kind:dominant, tokens:t, pct:pctf(t,total), synthetic:m.synthetic, preview:String(m.parts.map(p=>p.text).join(" ")).replace(/\s+/g," ").trim().slice(0,160), kinds:km};
  });
  const top = messageRows.filter(r=>!r.synthetic).sort((a,b)=>b.tokens-a.tokens).slice(0,10);
  const duplicates = duplicatesOf(messages);
  const dupWaste = duplicates.reduce((a,d)=>a+d.wasted,0);
  const waste = wasteOf(messages);
  const score = scoreOf(messages, kindTokens, dupWaste, waste, total);
  return {
    version:"browser",
    meta:{source:name, format:meta.format, model:meta.model, encoding:"estimate", requests:1, dropped:true},
    totals:{tokens:total, chars:chars, messages:real.length, schemas:parts.filter(p=>p.kind==="tool_schema").length, duplicate_tokens:dupWaste, duplicate_blocks:duplicates.reduce((a,d)=>a+d.count-1,0), waste_tokens:waste.reduce((a,w)=>a+w.tokens,0)},
    kinds, roles, positional, middle_share:middle, messages:messageRows, top,
    duplicates:duplicates.slice(0,12), near_duplicates:[], waste, trims:trimsOf(messages), score,
    dollars:{}
  };
}

/* ---- rendering ---- */
function render(d){
  const total = d.totals.tokens;
  document.title = "ctxscope · " + d.meta.source;
  document.getElementById("source").textContent = d.meta.source;
  const chips = [d.meta.format, fmt(d.totals.messages)+" messages", fmt(d.totals.schemas)+" tool schemas", d.meta.encoding];
  if(d.meta.model) chips.unshift(d.meta.model);
  document.getElementById("chips").innerHTML = chips.map(c=>'<span class="chip">'+esc(c)+'</span>').join("") + (d.meta.dropped?'<span class="chip" style="color:var(--accent)">analyzed locally</span>':'');

  document.getElementById("hero-tokens").textContent = fmt(total);
  document.getElementById("hero-dup").textContent = fmt(d.totals.duplicate_tokens);
  document.getElementById("hero-save").textContent = d.trims.safe_pct + "%";
  document.getElementById("hero-call").textContent = d.dollars && d.dollars.per_1m_frontier ? "$"+d.dollars.per_1m_frontier.toFixed(2)+" per call at $3/Mtok" : d.meta.encoding;
  document.getElementById("hero-dup-s").textContent = d.totals.duplicate_blocks + " repeated blocks";
  document.getElementById("hero-save-s").textContent = fmt(d.trims.safe_tokens) + " tokens, low risk";

  const score = d.score, C = 2*Math.PI*62;
  const color = score.value>=70?"var(--good)":score.value>=55?"var(--warn)":"var(--bad)";
  document.getElementById("ring").innerHTML =
    '<svg width="148" height="148" viewBox="0 0 148 148"><circle cx="74" cy="74" r="62" fill="none" stroke="var(--line)" stroke-width="9"/>'+
    '<circle cx="74" cy="74" r="62" fill="none" stroke="'+color+'" stroke-width="9" stroke-linecap="round" stroke-dasharray="'+C+'" stroke-dashoffset="'+(C*(1-score.value/100))+'"/></svg>'+
    '<div class="val"><div class="n" style="color:'+color+'">'+score.value+'<span class="of">/100</span></div>'+
    '<div class="gchip" style="color:'+color+';background:'+(score.value>=70?'rgba(74,222,128,.12)':score.value>=55?'rgba(251,191,36,.12)':'rgba(248,113,113,.14)')+'">'+esc(score.grade)+' · health</div></div>';

  document.getElementById("total-note").innerHTML = "<b class='mono' style='color:var(--text)'>"+fmt(total)+"</b> tokens";
  const top = d.kinds[0];
  document.getElementById("callout").innerHTML = top ? "<b>"+esc(top.label)+"</b> are "+top.pct.toFixed(0)+"% of this request. "+esc(d.meta.format)+", "+fmt(d.totals.messages)+" messages." : "";

  document.getElementById("stack").innerHTML = d.kinds.map(k=>'<i style="width:'+k.pct+'%;background:'+varOf(k.kind)+'" title="'+esc(k.label)+' '+k.pct+'%"></i>').join("");
  const maxKind = Math.max(...d.kinds.map(k=>k.tokens),1);
  document.getElementById("kinds").innerHTML =
    '<colgroup><col><col class="c-tokens"><col class="c-share"><col class="c-blocks"></colgroup>'+
    '<tr><th>Bucket</th><th class="n c-tokens">Tokens</th><th class="n c-share">Share</th><th class="n c-blocks">Blocks</th></tr>' +
    d.kinds.map(k=>'<tr><td><span class="sw" style="background:'+varOf(k.kind)+'"></span>'+esc(k.label)+
      '</td><td class="n c-tokens mono">'+fmt(k.tokens)+'</td><td class="n c-share"><div class="sharecell"><div class="mono" style="color:var(--muted)">'+k.pct.toFixed(1)+'%</div>'+
      '<div class="track3"><i style="width:'+Math.round(100*k.tokens/maxKind)+'%;background:'+varOf(k.kind)+';opacity:.55"></i></div></div>'+
      '</td><td class="n c-blocks"><span class="bars"><span class="mono" style="color:var(--faint);font-size:11.5px;min-width:2ch;text-align:right">'+k.blocks+'</span><span class="track"><i style="width:'+Math.round(100*k.tokens/maxKind)+'%;background:'+varOf(k.kind)+'"></i></span></span></td></tr>').join("");

  const posMax = Math.max(...d.positional,1);
  document.getElementById("position").innerHTML = d.positional.map((v,i)=>{
    const h = Math.max(3, Math.round(96*v/posMax));
    return '<div class="col"><div class="bar'+(i>=3&&i<7?" mid":"")+'" style="height:'+h+'px"></div><div class="lbl">'+((i*10)+5)+'%</div></div>';
  }).join("");
  document.getElementById("middle-note").textContent = d.middle_share ? d.middle_share+"% of tokens sit in the middle half" : "";

  const maxTop = Math.max(...d.top.map(r=>r.tokens),1);
  document.getElementById("rows").innerHTML = d.top.map(r=>'<div class="row"><div class="tag">'+(r.synthetic?"tools":"#"+r.index)+'</div><div class="tag role">'+esc(r.role)+
    '</div><div><div class="preview">'+esc(r.preview||"(empty)")+'</div><div class="track2"><i style="width:'+Math.round(100*r.tokens/maxTop)+'%;background:'+varOf(r.kind)+'"></i></div></div>'+
    '<div class="num"><div class="t">'+fmt(r.tokens)+'</div><div class="s">'+r.pct.toFixed(1)+'%</div></div></div>').join("");

  document.getElementById("dupe-note").textContent = fmt(d.totals.duplicate_tokens)+" wasted tokens";
  document.getElementById("dupes").innerHTML = d.duplicates.length ? d.duplicates.map(x=>'<div class="dcard"><div class="dhead"><span class="dcount">'+x.count+' copies · '+fmt(x.wasted)+' wasted</span><span class="dmeta">'+fmt(x.tokens)+' each</span></div><div class="dbody">'+esc(x.preview)+'</div></div>').join("")
    : '<div class="dcard empty">No block over '+MIN_DUPLICATE_TOKENS+' tokens appears twice. That is the expensive case to be in.</div>';

  document.getElementById("waste").innerHTML = d.waste.length ? d.waste.map(w=>'<div class="wrow"><div>'+esc(w.label)+'<div class="wdetail">'+esc(w.detail)+'</div></div><div class="amt">'+fmt(w.tokens)+'</div></div>').join("")
    : '<div style="color:var(--muted)">No structural waste found.</div>';

  document.getElementById("trims").innerHTML = d.trims.rows.map((t,i)=>'<div class="trow"><input type="checkbox" data-bit="'+i+'" '+(t.risk==="low"?"checked":"")+
    '><div>'+esc(t.label)+'</div><div class="tsave">'+fmt(t.tokens)+'</div>'+
    '<div class="riskcol"><span class="pill '+t.risk+'">'+t.risk+'</span></div></div>').join("") +
    '<div class="after"><div><div class="k">after trim</div><div class="big" id="after-n">'+fmt(total)+'</div></div>'+
    '<div style="text-align:right"><div class="k">saved</div><div class="big good" id="after-s">0</div></div></div>';
  document.querySelectorAll("#trims input[type=checkbox]").forEach(cb=>cb.addEventListener("change", ()=>updateAfter(d)));

  document.getElementById("next").innerHTML = recommendations(d).map(t=>{
    const split = t.indexOf(". ");
    return "<li>" + (split>0 ? "<b>"+esc(t.slice(0,split+1))+"</b> "+esc(t.slice(split+2)) : esc(t)) + "</li>";
  }).join("");
  document.getElementById("foot-left").textContent = "generated by ctxscope " + (d.version==="browser"?"· local estimate":"· no network requests, no telemetry");
  updateAfter(d);
}

function maskOf(){ let m=0; document.querySelectorAll("#trims input").forEach((cb,i)=>{ if(cb.checked) m|=(1<<i); }); return m; }
function updateAfter(d){
  const mask = maskOf();
  const total = d.totals.tokens;
  let after = d.trims.combos && d.trims.combos[String(mask)]!=null ? d.trims.combos[String(mask)] : total;
  if(d.trims.combos==null){
    const chosen = new Set(TRIMS_META.filter((_,i)=>mask&(1<<i)).map(t=>t[0]));
    after = d.trims.rows.reduce((a,t)=> chosen.has(t.id) ? a-t.tokens : a, total);
  }
  const saved = Math.max(0, total-after);
  document.getElementById("after-n").textContent = fmt(after);
  document.getElementById("after-s").textContent = fmt(saved) + (total? " · " + Math.round(100*saved/total) + "%" : "");
}

function recommendations(d){
  const out = [];
  if(d.duplicates.length){
    const t = d.duplicates[0];
    out.push(t.count+" identical copies of a "+fmt(t.tokens)+"-token block. Cache tool results by argument hash - the agent is re-running the same call.");
  }
  const pj = d.waste.filter(w=>w.id==="pretty-json")[0];
  if(pj) out.push("Compact JSON in tool output saves "+fmt(pj.tokens)+" tokens with no information loss.");
  const tr = d.kinds.filter(k=>k.kind==="tool_result")[0];
  if(tr && tr.pct>40) out.push("Tool results are "+tr.pct.toFixed(0)+"% of the window. Return summaries by default and full payloads only on request.");
  const ts = d.kinds.filter(k=>k.kind==="tool_schema")[0];
  if(ts && ts.pct>5) out.push("Tool schemas cost "+ts.pct.toFixed(0)+"% of every single call. Split rarely used tools into a second agent.");
  if(d.totals.messages>30) out.push("History is long: summarize completed turns instead of replaying every tool result verbatim.");
  if(!out.length) out.push("Nothing structural to fix. Keep the working set under 50k tokens and cache the stable prefix.");
  return out;
}

/* ---- entry points ---- */
function loadObject(obj, name){
  if(Array.isArray(obj) && obj.length && obj[0] && !obj[0].role && (obj[0].messages||obj[0].input)) obj = obj[obj.length-1];
  return analyzeInput(obj, name);
}
function handleFile(file){
  const reader = new FileReader();
  reader.onload = () => {
    try{
      const text = String(reader.result);
      let obj;
      try{ obj = JSON.parse(text); }
      catch(e){
        const lines = text.split("\n").map(l=>l.trim()).filter(l=>l.startsWith("{"));
        if(!lines.length) throw e;
        obj = JSON.parse(lines[lines.length-1]);
      }
      render(loadObject(obj, file.name));
      document.getElementById("drop").classList.remove("on");
      window.scrollTo({top:0,behavior:"smooth"});
    }catch(e){ alert("ctxscope could not parse " + file.name + "\n\n" + e.message); }
  };
  reader.readAsText(file);
}
document.getElementById("again").addEventListener("click", ()=>document.getElementById("file").click());
document.getElementById("file").addEventListener("change", e=>{ if(e.target.files[0]) handleFile(e.target.files[0]); });
document.getElementById("copy").addEventListener("click", async ()=>{
  const d = window.__ctxscope;
  const text = ["# ctxscope report: "+d.meta.source, "", "- "+fmt(d.totals.tokens)+" tokens ("+d.meta.encoding+")", "- context health "+d.score.value+"/100 ("+d.score.grade+")", "- duplicate waste "+fmt(d.totals.duplicate_tokens)+" tokens", "- safe savings "+d.trims.safe_pct+"%", "", "## Composition", ""].concat(d.kinds.map(k=>"- "+k.label+": "+fmt(k.tokens)+" ("+k.pct+"%)"), ["", "## Next", ""], recommendations(d).map(r=>"- "+r)).join("\n");
  try{ await navigator.clipboard.writeText(text); document.getElementById("copy").textContent="Copied"; setTimeout(()=>document.getElementById("copy").textContent="Copy summary", 1400); }catch(e){}
});
window.addEventListener("dragover", e=>{ e.preventDefault(); document.getElementById("drop").classList.add("on"); });
window.addEventListener("dragleave", e=>{ if(e.relatedTarget===null) document.getElementById("drop").classList.remove("on"); });
window.addEventListener("drop", e=>{ e.preventDefault(); if(e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); document.getElementById("drop").classList.remove("on"); });

window.__ctxscope = EMBEDDED;
render(EMBEDDED);
</script>
</body>
</html>
"""


def render_html(data: dict, template: str = VIEWER_TEMPLATE) -> str:
    """Render a self-contained report. No external assets, no network calls."""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    title = data["meta"].get("source") or "context window"
    return template.replace("__CTXRAY_DATA__", payload).replace(
        "<title>ctxscope report</title>", f"<title>ctxscope · {_html_escape(title)}</title>"
    )


def _html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# 7. demo trace
# ---------------------------------------------------------------------------

_SYSTEM = """You are Atlas, an autonomous research and operations agent running inside a managed Kubernetes platform.

## What you are for
You answer operational questions end to end: find the signal, quantify it, explain the mechanism, and write the update a human can send without editing. You are trusted with read access to the staging cluster and the analytics warehouse. You are not trusted with production writes, billing changes, or deleting anything.

## Non-negotiables
1. Never run a destructive command. Destructive means any verb containing delete, drain, cordon, scale to zero, or apply -f against a namespace you did not list in this run.
2. Cite a source for every factual claim in the final report. A source is a tool call id, a file path, or a query you ran.
3. Prefer the smallest tool call that answers the question. One query that returns the answer beats a query that returns the data for the answer.
4. When a tool fails twice with the same error, stop and report instead of retrying. Two identical failures mean the tool is down, not that the argument was wrong.
5. Never paste a full payload into the final report. Summarize, then link.
6. Keep the final report under 400 words.

## Working notes
- The warehouse is postgres, read-only role atlas_ro. The incidents table has 4.2M rows and no index on payload.
- Staging is us-east-1-staging. Production is us-east-1-prod and requires an approval I cannot grant.
- #incidents in Slack is the source of truth for the timeline when two dashboards disagree.
- The checkout service is owned by team payments; the ledger client is owned by team money-movement. A timeout in one shows up as an error in the other.
- Deploy history lives in `kubectl rollout history`. Revision numbers are per deployment and reset when we cut a release branch.

## How to work
Start by restating the question in one line to yourself. Then decide the smallest set of tool calls that separates the competing hypotheses. Run them. If the data contradicts your hypothesis, say so and change the plan rather than collecting more data for the original one.

## Output format
Ship a report with: impact (numbers and time window), cause (the mechanism, not the symptom), evidence (tool call ids), and next action (who does what). If you are unsure, name the uncertainty and what would resolve it.

## Style
Be concise. Be precise. Be concise. Do not speculate when a query can answer it. Do not apologize. Do not narrate what you are about to do when the tool call already says it.

Ambiguity note: when the user says "checkout is down", they usually mean the checkout path in the payment service, not the checkout repository. Check both before concluding.
"""

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internal status index and the public web, and return ranked results. Each result carries title, url, a snippet of up to 400 characters, and a relevance score. Use this for incident history and for anything that is not in the warehouse. Do not use this to look up service health; use fetch_page on the status service instead, because search results are cached for 15 minutes and can be stale during an incident.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language query. Quoted phrases are exact matched."},
                    "limit": {"type": "integer", "description": "Results to return, 1 to 20.", "default": 5},
                    "recency_days": {"type": "integer", "description": "Only results newer than this many days. Omit for no filter.", "default": 30},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch a URL and return cleaned text. HTML is stripped of scripts, styles, and navigation, then truncated at 100,000 characters. Pages behind the internal gateway need the full hostname, not a path relative to the status service. Returns the final URL after redirects, the HTTP status, and the content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Absolute URL, http or https."},
                    "max_chars": {"type": "integer", "description": "Truncate the cleaned text at this many characters.", "default": 100000},
                    "selector": {"type": "string", "description": "Optional CSS selector to scope the extraction, for example 'main' or '#incident-body'."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sql_query",
            "description": "Run one read-only SQL statement against the analytics warehouse (postgres 16, read replica). Returns rows as JSON objects, the row count, and the wall clock duration. Statements run with a 30 second timeout and a 10,000 row cap; use aggregation rather than pulling raw rows when the table is large. The incidents table is not indexed on payload, so filtering on JSON fields is a sequential scan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A single SELECT statement. CTEs are allowed. Semicolons terminate the statement."},
                    "max_rows": {"type": "integer", "description": "Row cap for this query, up to 10000.", "default": 1000},
                    "timeout_seconds": {"type": "integer", "description": "Server side timeout.", "default": 30},
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command in the agent workspace and return stdout, stderr, and the exit code. The workspace has kubectl, psql, jq, and curl. There is no network egress except to the internal gateway. Commands are killed at the timeout and the partial output is returned, so a truncated stdout usually means a timeout rather than success.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command line. Pipes and redirects are allowed; background processes are not."},
                    "timeout_seconds": {"type": "integer", "description": "Kill the command after this many seconds.", "default": 30},
                    "cwd": {"type": "string", "description": "Working directory inside the workspace."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace and return its contents with line numbers, so you can cite exact lines. Reads over 200 lines are paginated: pass start_line and end_line rather than reading the whole file. Binary files are rejected. A file over 2MB is truncated with a marker at the end.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root. Absolute paths outside the workspace are rejected."},
                    "start_line": {"type": "integer", "description": "First line to return, 1 indexed.", "default": 1},
                    "end_line": {"type": "integer", "description": "Last line to return, inclusive."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file in the workspace, overwriting anything already there. Use this for scratch analysis and for the draft report. It cannot write outside the workspace and it does not create directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root."},
                    "content": {"type": "string", "description": "Full file contents."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List a directory with entry names, sizes in bytes, and modification times, sorted by name. Use this before read_file when you do not know the layout. Hidden files are included.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path relative to the workspace root.", "default": "."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "post_slack",
            "description": "Post a message to a Slack channel as the agent. The message is sent under the agent's own identity and cannot be edited or deleted afterwards, so draft it before sending. Threads are not supported; every message lands at the channel root. Rate limited to 20 messages per hour per channel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string", "description": "Channel name including the hash, for example #incidents."},
                    "text": {"type": "string", "description": "Message body in Slack markdown. Max 4000 characters."},
                },
                "required": ["channel", "text"],
            },
        },
    },
]

_HTML_DUMP = (
    "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<title>Service status - Atlas platform</title>\n"
    "<style>\n"
    + "".join(
        f".s{i}{{display:flex;gap:{i}px;padding:{i}px;border-bottom:1px solid #{i}{i}{i};font-size:14px}}\n"
        for i in range(1, 26)
    )
    + "</style>\n</head>\n<body>\n<div class=\"wrap\">\n<nav class=\"breadcrumb\">"
    + "".join(f"<a href=\"/status/{i}\">service-{i}</a>\n" for i in range(40))
    + "</nav>\n<h1>Platform status</h1>\n<p class=\"updated\">Last updated 2026-11-18T09:11:52Z. Data refreshes every 30 seconds.</p>\n"
    + "".join(
        f"<div class=\"s{i}\">\n  <span class=\"name\">service-{i}</span>\n  <span class=\"state\">healthy</span>\n"
        f"  <span class=\"p99\">p99 {40 + i * 3}ms</span>\n  <span class=\"err\">error rate 0.0{i}%</span>\n"
        f"  <span class=\"region\">us-east-1</span>\n  <span class=\"owner\">team-{i % 7}</span>\n"
        f"  <span class=\"slo\">SLO 99.{90 + (i % 9)}% | burn 0.{i}</span>\n</div>\n"
        for i in range(1, 26)
    )
    + "<footer>Generated by statuspage-renderer 4.2.1. For incident history see /incidents. For raw metrics see /metrics.</footer>\n</div>\n</body>\n</html>\n"
)

_API_DUMP = json.dumps(
    {
        "errors": [
            {
                "id": f"err_{1700 + n}",
                "first_seen": "2026-11-18T08:41:07Z",
                "last_seen": "2026-11-18T09:09:31Z",
                "count": 1204 + n * 37,
                "service": "checkout",
                "operation": "POST /v2/checkout/confirm",
                "status": 500,
                "exception": "capy.exceptions.LedgerTimeout: ledger capture timed out after 3000ms",
                "release": "2.14.1",
                "region": "us-east-1",
                "tags": {"team": "payments", "tier": "critical", "ledger_client": "4.8.0"},
            }
            for n in range(60)
        ],
        "total": 60,
        "page": 1,
        "page_size": 50,
    },
    indent=2,
)

_KUBECTL_DESCRIBE = (
    "Name:                   checkout-7d9f8c6b5-x2m4p\n"
    "Namespace:              us-east-1-staging\n"
    "Priority Class Name:    platform-critical\n"
    "Node:                   ip-10-0-14-229.ec2.internal/10.0.14.229\n"
    "Start Time:             Mon, 17 Nov 2026 22:14:03 +0000\n"
    "Labels:                 app=checkout\n"
    "                        pod-template-hash=7d9f8c6b5\n"
    "                        version=2.14.1\n"
    "Annotations:            prometheus.io/scrape: true\n"
    "Status:                 Running\n"
    "IP:                     10.0.19.44\n"
    "Containers:\n"
    "  checkout:\n"
    "    Image:          registry.internal/checkout:2.14.1\n"
    "    Port:           8080/TCP\n"
    "    State:          Running\n"
    "    Ready:          True\n"
    "    Restart Count:  0\n"
    "    Limits:\n"
    "      cpu:     2\n"
    "      memory:  2Gi\n"
    "    Requests:\n"
    "      cpu:     500m\n"
    "      memory:  512Mi\n"
    "    Environment:\n"
    "      LEDGER_TIMEOUT_MS:   3000\n"
    "      LEDGER_RETRIES:      2\n"
    "      PAYMENT_REQUIRE_3DS: true\n"
    "      OTEL_EXPORTER:       otlp://collector.platform:4317\n"
    "    Mounts:\n"
    "      /etc/checkout from config (ro)\n"
    "Conditions:\n"
    "  Type              Status\n"
    "  Initialized       True\n"
    "  Ready             True\n"
    "  ContainersReady   True\n"
    "  PodScheduled      True\n"
    "Volumes:\n"
    "  config:\n"
    "    Type:      ConfigMap (a volume populated by a ConfigMap)\n"
    "    Name:      checkout-config\n"
    "QoS Class:     Burstable\n"
    "Node-Selectors:  topology.kubernetes.io/zone=us-east-1a\n"
    "Tolerations:     node.kubernetes.io/not-ready:NoExecute op=Exists for 300s\n"
    "                 node.kubernetes.io/unreachable:NoExecute op=Exists for 300s\n"
    "Events:\n"
    "  Type    Reason     Age   From     Message\n"
    "  ----    ------     ----  ----     -------\n"
    "  Normal  Scheduled  11h   default-scheduler  Successfully assigned us-east-1-staging/checkout-7d9f8c6b5-x2m4p\n"
    "  Normal  Pulled     11h   kubelet            Container image already present on machine\n"
    "  Normal  Created    11h   kubelet            Created container checkout\n"
    "  Normal  Started    11h   kubelet            Started container checkout\n"
    "  Warning BackOff    38m   kubelet            Back-off restarting failed container ledger-sidecar\n"
)

_GIT_DIFF = (
    "diff --git a/services/checkout/config/ledger.go b/services/checkout/config/ledger.go\n"
    "index 4f2a1c9..8b3e0d1 100644\n"
    "--- a/services/checkout/config/ledger.go\n"
    "+++ b/services/checkout/config/ledger.go\n"
    "@@ -18,9 +18,11 @@ type LedgerConfig struct {\n"
    " \tRetries          int           `env:\"LEDGER_RETRIES\" default:\"2\"`\n"
    " \tRequire3DS       bool          `env:\"PAYMENT_REQUIRE_3DS\" default:\"true\"`\n"
    "-\tTimeout          time.Duration `env:\"LEDGER_TIMEOUT_MS\" default:\"8000\"`\n"
    "+\tTimeout          time.Duration `env:\"LEDGER_TIMEOUT_MS\" default:\"3000\"`\n"
    " }\n"
    " \n"
    " func (c LedgerConfig) Validate() error {\n"
    "-\tif c.Timeout < 5*time.Second {\n"
    "-\t\treturn errors.New(\"ledger timeout below 5s will fail under burst\")\n"
    "+\tif c.Timeout < time.Second {\n"
    "+\t\treturn errors.New(\"ledger timeout must be positive\")\n"
    " \t}\n"
    " \treturn nil\n"
    " }\n"
    + "".join(
        f" {n:>4}  // refactor: pass context through confirm path (step {n})\n" for n in range(1, 121)
    )
)

_TRACEBACK = (
    "2026-11-18T08:41:07Z ERROR checkout.handler confirm failed account=acct_1Qc9 intent=intent_9f2\n"
    + "".join(
        f"  File \"/srv/checkout/app/handlers/payment.py\", line {120 + n}, in confirm\n"
        f"    await ledger.capture(intent_id, amount)\n"
        for n in range(18)
    )
    + "capy.exceptions.LedgerTimeout: ledger capture timed out after 3000ms\n"
    "\n2026-11-18T08:41:09Z ERROR checkout.handler confirm failed account=acct_2Rd1 intent=intent_7a4\n"
    + "".join(f"  File \"/srv/checkout/app/clients/ledger.py\", line {40 + n}, in capture\n" for n in range(12))
    + "capy.exceptions.LedgerTimeout: ledger capture timed out after 3000ms\n"
    "\n2026-11-18T08:41:11Z WARN  checkout.retry   attempt=2 of 2 elapsed=6100ms\n"
    "2026-11-18T08:41:11Z ERROR checkout.handler confirm failed after retries intent=intent_7a4\n"
)

_VALUES_YAML = "\n".join(
    f"{n:>4}  {line}"
    for n, line in enumerate(
        [
            "checkout:",
            "  replicaCount: 12",
            "  image:",
            "    repository: registry.internal/checkout",
            "    tag: 2.14.1",
            "    pullPolicy: IfNotPresent",
            "  resources:",
            "    requests: {cpu: 500m, memory: 512Mi}",
            "    limits: {cpu: 2, memory: 2Gi}",
            "  ledger:",
            "    timeoutMs: 3000",
            "    retries: 2",
            "    circuitBreaker:",
            "      enabled: true",
            "      errorThreshold: 0.5",
            "      volumeThreshold: 20",
            "      sleepWindowMs: 5000",
            "  payment:",
            "    require3ds: true",
            "    splitCapture: false",
            "  autoscaling:",
            "    enabled: true",
            "    minReplicas: 12",
            "    maxReplicas: 40",
            "    targetCPUUtilizationPercentage: 65",
            "  podDisruptionBudget:",
            "    minAvailable: 8",
            "  nodeSelector:",
            "    topology.kubernetes.io/zone: us-east-1a",
        ]
        * 6
    )
)


def _search_results() -> str:
    return json.dumps(
        {
            "results": [
                {
                    "title": f"Checkout latency regression #{n}: p99 210ms to 1.9s",
                    "url": f"https://status.internal/incidents/{1481 + n}",
                    "snippet": (
                        "Checkout p99 rose from 210ms to 1.9s at 08:41 UTC. Errors are concentrated in the payment intent "
                        "confirmation path, specifically the ledger capture call. The on-call noted that revision 13 of the "
                        "checkout deployment reduced the ledger timeout from 8000ms to 3000ms. "
                    )
                    * 2,
                    "score": round(0.99 - n / 100, 2),
                    "published": "2026-11-18T09:02:00Z",
                }
                for n in range(5)
            ],
            "took_ms": 84,
        },
        indent=2,
    )


def build_demo_trace(healthy: bool = False) -> dict:
    """A synthetic but realistic long agent run, with the waste people actually ship.

    It is deliberately messy: duplicate tool results, pretty-printed payloads,
    full HTML dumps, and a traceback replayed twice. Run with healthy=True for
    the same task done well, to see what a clean context looks like.
    """
    tools = _TOOLS
    messages = [{"role": "system", "content": _SYSTEM}]
    messages.append(
        {
            "role": "user",
            "content": "Checkout is throwing 5xx since 08:40 UTC. Find the cause, quantify the impact, and draft the incident update for #incidents. Do not touch production.",
        }
    )

    if healthy:
        messages.append({"role": "assistant", "content": "Restating: checkout 5xx since 08:40; cause unknown; impact unknown; production off limits. Two facts separate the hypotheses: the deploy timeline and the error's exception class. Checking both in one pass."})
        messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": "h1", "type": "function", "function": {"name": "sql_query", "arguments": json.dumps({"sql": "select date_trunc('hour', ts) as hour, count(*) filter (where status >= 500) as errors, count(*) as total, max(exception) as exception from requests where path like '/checkout%' and ts > now() - interval '2 hours' group by 1 order by 1"})}}]})
        messages.append({"role": "tool", "name": "sql_query", "tool_call_id": "h1", "content": "2 rows. 08:00 -> 41/12,004 (0.34%) LedgerTimeout. 09:00 -> 3,182/12,190 (26.1%) LedgerTimeout. Window started 08:41."})
        messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": "h2", "type": "function", "function": {"name": "run_shell", "arguments": json.dumps({"command": "kubectl -n us-east-1-staging rollout history deploy/checkout | tail -3"})}}]})
        messages.append({"role": "tool", "name": "run_shell", "tool_call_id": "h2", "content": "REVISION 13 deployed 08:39 UTC. Diff: LEDGER_TIMEOUT_MS default 8000 -> 3000."})
        messages.append({"role": "assistant", "content": "Cause: revision 13 cut the ledger timeout to 3000ms; p99 ledger capture is 2.9s and the confirmation burst pushes it past the new ceiling, so 26% of confirms fail. Impact: 08:41-09:09 UTC, 26.1% error rate, ~3,182 failed confirms. Next: roll back to revision 12, owner team-payments."})
        messages.append({"role": "user", "content": "Good. Draft the channel update."})
        messages.append({"role": "assistant", "content": "Drafted in #incidents: 3 lines, impact first, cause second, next action with an owner."})
        return {"model": "gpt-5.1", "messages": messages, "tools": tools, "temperature": 0.2}

    search = _search_results()

    messages.append({"role": "assistant", "content": "Restating the ask: checkout 5xx since 08:40, cause unknown, impact unknown, no production writes. I will start from incident search, then the error breakdown, then the deploy diff. My prior is a deploy, but that is a prior and not an answer."})
    messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "web_search", "arguments": json.dumps({"query": "checkout 5xx spike november", "limit": 5})}}]})
    messages.append({"role": "tool", "name": "web_search", "tool_call_id": "c1", "content": search})

    messages.append({"role": "assistant", "content": "Two incidents in the last day mention the ledger client. Pulling the hourly error breakdown and the exception class."})
    messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "sql_query", "arguments": json.dumps({"sql": "select date_trunc('hour', ts) as hour, count(*) filter (where status >= 500) as errors, count(*) as total from requests where path like '/checkout%' and ts > now() - interval '6 hours' group by 1 order by 1", "max_rows": 100})}}]})
    rows = [{"hour": f"2026-11-18T{h:02d}:00:00Z", "errors": 41 + h * 7, "total": 12004 + h * 31} for h in range(3, 9)]
    messages.append({"role": "tool", "name": "sql_query", "tool_call_id": "c2", "content": json.dumps({"rows": rows, "row_count": len(rows), "duration_ms": 812}, indent=2)})

    messages.append({"role": "assistant", "content": "The step is at 08:41, which is inside the error-prone hour. I need the exception distribution and the deploy timeline next."})
    messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": "c3", "type": "function", "function": {"name": "fetch_page", "arguments": json.dumps({"url": "https://status.internal/services"})}}]})
    messages.append({"role": "tool", "name": "fetch_page", "tool_call_id": "c3", "content": _HTML_DUMP})

    messages.append({"role": "assistant", "content": "Status page is all green, which is itself the finding: the platform sees healthy services while one path burns. Pulling the error group payload."})
    messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": "c4", "type": "function", "function": {"name": "fetch_page", "arguments": json.dumps({"url": "https://errors.internal/api/v1/groups?service=checkout&since=6h"})}}]})
    messages.append({"role": "tool", "name": "fetch_page", "tool_call_id": "c4", "content": _API_DUMP})

    messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": "c5", "type": "function", "function": {"name": "run_shell", "arguments": json.dumps({"command": "kubectl -n us-east-1-staging rollout history deploy/checkout --revision=0", "timeout_seconds": 30})}}]})
    messages.append({"role": "tool", "name": "run_shell", "tool_call_id": "c5", "content": "deployment.apps/checkout\nREVISION  CHANGE-CAUSE\n11        <none>\n12        <none>\n13        <none>\n"})

    messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": "c6", "type": "function", "function": {"name": "run_shell", "arguments": json.dumps({"command": "kubectl -n us-east-1-staging describe pod checkout-7d9f8c6b5-x2m4p", "timeout_seconds": 30})}}]})
    messages.append({"role": "tool", "name": "run_shell", "tool_call_id": "c6", "content": _KUBECTL_DESCRIBE})

    messages.append({"role": "assistant", "content": "Revision 13 is the only change in the window. I want the exact config diff and a sample traceback to confirm the mechanism rather than assume it."})
    messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": "c7", "type": "function", "function": {"name": "run_shell", "arguments": json.dumps({"command": "git -C /workspace/checkout diff 2.14.0..2.14.1 -- services/checkout/config", "timeout_seconds": 30})}}]})
    messages.append({"role": "tool", "name": "run_shell", "tool_call_id": "c7", "content": _GIT_DIFF})

    messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": "c8", "type": "function", "function": {"name": "run_shell", "arguments": json.dumps({"command": "kubectl -n us-east-1-staging logs deploy/checkout --tail=200 | grep -A20 LedgerTimeout | head -80", "timeout_seconds": 45})}}]})
    messages.append({"role": "tool", "name": "run_shell", "tool_call_id": "c8", "content": _TRACEBACK})

    messages.append({"role": "assistant", "content": "Confirmed: the diff drops the timeout ceiling below the observed p99, and the traceback is the exact call. Re-reading the values file to quantify the burst headroom before I write the impact number."})
    messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": "c9", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "deploy/checkout/values.yaml", "start_line": 1, "end_line": 200})}}]})
    messages.append({"role": "tool", "name": "read_file", "tool_call_id": "c9", "content": _VALUES_YAML})

    # Re-running the identical search: pure duplicate context, and a very common pattern.
    messages.append({"role": "assistant", "content": "I want to double check the incident list before writing the timeline."})
    messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": "c10", "type": "function", "function": {"name": "web_search", "arguments": json.dumps({"query": "checkout 5xx spike november", "limit": 5})}}]})
    messages.append({"role": "tool", "name": "web_search", "tool_call_id": "c10", "content": search})

    messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": "c11", "type": "function", "function": {"name": "fetch_page", "arguments": json.dumps({"url": "https://status.internal/services"})}}]})
    messages.append({"role": "tool", "name": "fetch_page", "tool_call_id": "c11", "content": _HTML_DUMP})

    messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": "c12", "type": "function", "function": {"name": "post_slack", "arguments": json.dumps({"channel": "#incidents", "text": "Investigating checkout 5xx. Suspect ledger timeout change in revision 13. Impact window 08:40 to now."})}}]})
    messages.append({"role": "tool", "name": "post_slack", "tool_call_id": "c12", "content": json.dumps({"ok": True, "ts": "1763457167.001200"}, indent=2)})

    messages.append({"role": "assistant", "content": "I have the mechanism and the window. Before the final draft I need one more pass over the incident list to make sure revision 13 is not a red herring - the same p99 pattern appeared in October without a deploy."})
    messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": "c13", "type": "function", "function": {"name": "web_search", "arguments": json.dumps({"query": "ledger timeout october checkout p99", "limit": 5, "recency_days": 90})}}]})
    messages.append({"role": "tool", "name": "web_search", "tool_call_id": "c13", "content": _search_results()})

    messages.append({"role": "user", "content": "Also: do not forget the standing constraint from the runbook - production is read-only and requires approval. Keep the report under 400 words. And when you post, use the timeline format from the incident template, not freeform."})
    messages.append({"role": "assistant", "content": "Draft ready. Impact: 08:41-09:09 UTC, 26.1% error rate on /checkout/confirm, ~3,182 failed confirms, no order loss because confirm is retried client side. Cause: revision 13 lowered ledger.timeout_ms from 8000 to 3000; observed p99 ledger capture under the confirmation burst is 2.9s, so confirms cross the ceiling. Evidence: c2 (error by hour), c5 (revision timestamp), c7 (config diff), c8 (exception). Next: roll back to revision 12, owner team-payments, then re-check c2."})

    return {"model": "gpt-5.1", "messages": messages, "tools": tools, "temperature": 0.2}


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------


def _use_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ctxscope",
        description="X-ray an LLM context window: composition, duplicates, waste, and what to cut.",
    )
    parser.add_argument("path", nargs="?", help="request JSON, trace JSON, JSONL, or - for stdin")
    parser.add_argument("--demo", action="store_true", help="analyze a built-in 12-step agent trace")
    parser.add_argument("--html", metavar="OUT", help="write a self-contained HTML report")
    parser.add_argument("--json", dest="json_out", metavar="OUT", help="write the raw analysis as JSON")
    parser.add_argument("--md", metavar="OUT", help="write a markdown report")
    parser.add_argument("--format", default="auto", choices=["auto", "openai", "anthropic", "raw"], help="input format override")
    parser.add_argument("--encoding", default="auto", choices=["auto", "cl100k", "o200k", "estimate"], help="tokenizer to use")
    parser.add_argument("--top", type=int, default=10, help="how many messages to list (default 10)")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color")
    parser.add_argument("--quiet", action="store_true", help="write files only, no terminal report")
    parser.add_argument("--version", action="version", version=f"ctxscope {__version__}")
    args = parser.parse_args(argv)

    if not args.path and not args.demo:
        parser.print_help()
        return 2

    if args.demo:
        messages, meta = normalize(build_demo_trace(), args.encoding)
        meta["source"] = "built-in demo trace"
        meta["format"] = "openai"
    else:
        try:
            messages, meta = load(args.path, args.encoding)
        except FileNotFoundError:
            print(f"ctxscope: no such file: {args.path}", file=sys.stderr)
            return 2
        except Exception as error:  # noqa: BLE001
            print(f"ctxscope: could not parse {args.path}: {error}", file=sys.stderr)
            return 2
        if args.format != "auto":
            meta["format"] = args.format

    if not any(message.tokens for message in messages):
        print("ctxscope: nothing to analyze (no text found)", file=sys.stderr)
        return 2

    meta["encoding"] = args.encoding
    data = analyze(messages, meta, top=args.top)

    wrote = []
    if args.html:
        Path(args.html).write_text(render_html(data), encoding="utf-8")
        wrote.append(args.html)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        wrote.append(args.json_out)
    if args.md:
        Path(args.md).write_text(render_markdown(data), encoding="utf-8")
        wrote.append(args.md)

    if not args.quiet or not wrote:
        print(render_terminal(data, color=_use_color() and not args.no_color, top=args.top))
    for path in wrote:
        print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
