#!/usr/bin/env python3
"""Thin inference adapter: Self-Harness owns prompt construction and parsing."""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True, type=Path)
    p.add_argument("--response", required=True, type=Path)
    p.add_argument("--api-base", default=os.environ.get("PROPOSER_API_BASE", "http://unites4.ib:30017/v1"))
    p.add_argument("--model", default=os.environ.get("PROPOSER_MODEL", "qwen3-8b"))
    p.add_argument("--max-tokens", type=int, default=8192)
    args = p.parse_args()
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "Return only valid JSON matching the requested proposer schema."},
            {"role": "user", "content": args.prompt.read_text()},
        ],
        "temperature": 0.2,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        args.api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=2400) as response:
        result = json.load(response)
    content = result["choices"][0]["message"]["content"]
    args.response.parent.mkdir(parents=True, exist_ok=True)
    args.response.write_text(content.rstrip() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
