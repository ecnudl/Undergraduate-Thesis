#!/usr/bin/env python3
"""Download the official LongMemEval benchmark files.

By default this script downloads the cleaned `S` split and the oracle split,
which is the smallest useful setup for validating a long-context baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


DATASET_ROOT = Path(__file__).resolve().parent / "data" / "longmemeval"
BASE_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"

FILE_MAP = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _validate_json(path: Path) -> tuple[int, str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a JSON list")
    return len(data), _sha256(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download LongMemEval benchmark files")
    parser.add_argument(
        "--output-dir",
        default=str(DATASET_ROOT),
        help="Destination directory. Defaults to data/longmemeval",
    )
    parser.add_argument(
        "--include-m",
        action="store_true",
        help="Also download LongMemEval_M (much larger / longer context).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    variants = ["oracle", "s"]
    if args.include_m:
        variants.append("m")

    print("=" * 60)
    print("LongMemEval download")
    print("=" * 60)
    print(f"Output dir: {output_dir}")
    print(f"Variants:   {', '.join(variants)}")
    print("")

    failures: list[str] = []

    for variant in variants:
        filename = FILE_MAP[variant]
        url = f"{BASE_URL}/{filename}"
        destination = output_dir / filename

        try:
            if destination.exists() and not args.force:
                count, checksum = _validate_json(destination)
                print(f"[skip] {filename} already exists ({count} items, sha256={checksum[:12]}...)")
                continue

            print(f"[download] {filename}")
            _download(url, destination)
            count, checksum = _validate_json(destination)
            print(f"[ok] {filename}: {count} items, sha256={checksum[:12]}...")
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            failures.append(f"{filename}: network error: {exc}")
        except Exception as exc:
            failures.append(f"{filename}: {exc}")

    print("")
    if failures:
        print("[ERROR] Some downloads failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("[DONE] LongMemEval data is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
