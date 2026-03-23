#!/usr/bin/env python
"""Aggregate a completed experiment run into CSV files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from common.aggregation import aggregate_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate a unified experiment run")
    parser.add_argument("--run-dir", required=True, help="Run directory under experiments/results")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    result = aggregate_run(run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
