#!/usr/bin/env python3
"""Build a fixed balanced LongMemEval subset manifest."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_COUNTS = {
    "multi-session": 44,
    "temporal-reasoning": 44,
    "knowledge-update": 26,
    "single-session-user": 23,
    "single-session-assistant": 18,
    "single-session-preference": 10,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a fixed balanced LongMemEval subset manifest")
    parser.add_argument("--infile", required=True, help="Path to LongMemEval JSON file")
    parser.add_argument("--outfile", required=True, help="Output manifest JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    args = parser.parse_args()

    infile = Path(args.infile).resolve()
    outfile = Path(args.outfile).resolve()

    with infile.open("r", encoding="utf-8") as f:
        data = json.load(f)

    by_type = defaultdict(list)
    for idx, item in enumerate(data, start=1):
        by_type[str(item.get("question_type", "unknown"))].append(
            {
                "global_index": idx,
                "question_id": item.get("question_id"),
                "question_type": item.get("question_type"),
                "question": item.get("question"),
            }
        )

    rng = random.Random(args.seed)
    selected = []
    counts = {}
    for question_type, quota in DEFAULT_COUNTS.items():
        pool = list(by_type[question_type])
        if len(pool) < quota:
            raise SystemExit(
                f"question_type={question_type} only has {len(pool)} items, quota={quota}"
            )
        rng.shuffle(pool)
        chosen = sorted(pool[:quota], key=lambda x: x["global_index"])
        selected.extend(chosen)
        counts[question_type] = len(chosen)

    selected.sort(key=lambda x: x["global_index"])
    manifest = {
        "dataset": "longmemeval",
        "split": "s_cleaned",
        "seed": args.seed,
        "total_selected": len(selected),
        "counts": counts,
        "selected_indices": [item["global_index"] for item in selected],
        "selected_question_ids": [item["question_id"] for item in selected],
        "items": selected,
    }

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps({"outfile": str(outfile), "total_selected": len(selected), "counts": counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
