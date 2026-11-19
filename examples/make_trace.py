#!/usr/bin/env python3
"""Regenerate the example traces committed in this directory.

    python3 examples/make_trace.py

The traces are synthetic but built to look like real agent runs: tool schemas,
a long HTML dump, a replayed search, a pretty-printed API payload, a traceback.
Edit build_demo_trace() in ctxscope.py to change them.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import ctxscope  # noqa: E402


def main() -> int:
    messy = ctxscope.build_demo_trace()
    healthy = ctxscope.build_demo_trace(healthy=True)
    (HERE / "agent-trace.json").write_text(json.dumps(messy, indent=2) + "\n", encoding="utf-8")
    (HERE / "healthy-trace.json").write_text(json.dumps(healthy, indent=2) + "\n", encoding="utf-8")
    for name, payload in (("messy", messy), ("healthy", healthy)):
        messages, _ = ctxscope.normalize(payload, "estimate")
        print(f"{name:8} {sum(m.tokens for m in messages):>8,} tokens  {len(messages)} messages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
