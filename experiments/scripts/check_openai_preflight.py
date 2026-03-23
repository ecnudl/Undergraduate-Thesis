#!/usr/bin/env python3
"""Minimal OpenAI-compatible API preflight for experiment launchers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import requests


def build_proxy_snapshot() -> Dict[str, str]:
    snapshot = {}
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        value = os.environ.get(key)
        if value:
            snapshot[key] = value
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Check API connectivity before running experiments")
    parser.add_argument("--api-base", required=True, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", required=True, help="Model id for the smoke request")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print(json.dumps({"ok": False, "stage": "env", "error": "OPENAI_API_KEY is missing"}, ensure_ascii=False, indent=2))
        return 2

    api_base = args.api_base.rstrip("/")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    result = {
        "ok": False,
        "api_base": api_base,
        "model": args.model,
        "proxy_env": build_proxy_snapshot(),
    }

    try:
        models_resp = requests.get(f"{api_base}/models", headers=headers, timeout=args.timeout)
        result["models_status"] = models_resp.status_code
        result["models_body_preview"] = models_resp.text[:200]
        models_resp.raise_for_status()
    except Exception as exc:
        result["stage"] = "models"
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 3

    try:
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with OK only."}],
            "stream": True,
            "max_tokens": 8,
        }
        with requests.post(
            f"{api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=max(args.timeout, 30),
            stream=True,
        ) as resp:
            result["chat_status"] = resp.status_code
            resp.raise_for_status()
            first_chunk = None
            for line in resp.iter_lines(decode_unicode=True):
                if line:
                    first_chunk = line[:200]
                    break
            result["chat_first_chunk"] = first_chunk
    except Exception as exc:
        result["stage"] = "chat"
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 4

    result["ok"] = True
    result["stage"] = "done"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
