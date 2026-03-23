#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_utils import generate_unified_report  # noqa: E402


DEFAULT_RUN_DIR = REPO_ROOT / "experiments" / "results" / "gaia_workflow_semantic_eval_20260320_022030"
DEFAULT_INFILE = REPO_ROOT / "data" / "gaia" / "validation" / "metadata.jsonl"
DEFAULT_TASK_SELECTION = "[level1]1-26 [level2]1-43 [level3]1-13"
DEFAULT_MODEL = "gpt-5"
DEFAULT_API_BASE = "https://api-vip.codex-for.me/v1"
DEFAULT_RUNNER_PYTHON = Path("/home/admin123/anaconda3/envs/evolvelab/bin/python")


@dataclass(frozen=True)
class RunSpec:
    name: str
    memory_provider: str | None
    extra_env: Dict[str, str]


RUN_SPECS: Sequence[RunSpec] = (
    RunSpec(
        name="gaia_no_memory_baseline",
        memory_provider=None,
        extra_env={},
    ),
    RunSpec(
        name="gaia_workflow_shortcut_hybrid_semantic_json_full",
        memory_provider="modular",
        extra_env={
            "MODULAR_ENABLED_PROMPTS": "workflow,shortcut",
            "MODULAR_STORAGE_TYPE": "hybrid",
            "MODULAR_RETRIEVER_TYPE": "semantic",
            "MODULAR_MANAGEMENT_ENABLED": "true",
            "MODULAR_MANAGEMENT_PRESET": "json_full",
        },
    ),
)


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_simple_indices(indices_str: str) -> List[int]:
    values: List[int] = []
    for raw_part in indices_str.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            if start > end:
                raise ValueError(f"Invalid range: {part}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    seen = set()
    ordered_unique: List[int] = []
    for value in values:
        if value not in seen:
            ordered_unique.append(value)
            seen.add(value)
    return ordered_unique


def parse_task_selection(task_selection: str, data: Sequence[Dict]) -> List[int]:
    task_selection = task_selection.strip()
    if not task_selection:
        raise ValueError("task_selection cannot be empty")
    if "[level" not in task_selection.lower():
        return parse_simple_indices(task_selection)

    matches = list(re.finditer(r"\[level(\d+)\]", task_selection, re.IGNORECASE))
    if not matches:
        raise ValueError(f"Unsupported task selection syntax: {task_selection}")

    selected: List[int] = []
    seen = set()
    for idx, match in enumerate(matches):
        level = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(task_selection)
        spec = task_selection[start:end].strip()

        level_rows = [row for row in data if str(row.get("Level", "")).strip() == level]
        if not spec:
            chosen = [row["_global_index"] for row in level_rows]
        else:
            level_relative = parse_simple_indices(spec.replace(" ", ""))
            chosen = []
            for rel_idx in level_relative:
                array_idx = rel_idx - 1
                if 0 <= array_idx < len(level_rows):
                    chosen.append(level_rows[array_idx]["_global_index"])
        for value in chosen:
            if value not in seen:
                selected.append(value)
                seen.add(value)
    return selected


def parse_optional_indices(indices_str: str | None) -> List[int]:
    if not indices_str:
        return []
    return parse_simple_indices(indices_str)


def load_gaia_data(infile: Path) -> List[Dict]:
    rows = read_jsonl(infile)
    data: List[Dict] = []
    for idx, row in enumerate(rows, start=1):
        item = dict(row)
        item["_global_index"] = idx
        data.append(item)
    return data


def load_completed_indices(tasks_dir: Path) -> List[int]:
    completed: List[int] = []
    if not tasks_dir.exists():
        return completed
    for path in sorted(tasks_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        item_index = payload.get("item_index")
        if isinstance(item_index, int):
            completed.append(item_index)
    return completed


def build_runner_command(
    run_dir: Path,
    task_indices: List[int],
    spec: RunSpec,
    args: argparse.Namespace,
) -> List[str]:
    results_path = run_dir / "results.resume.jsonl"
    command = [
        args.runner_python,
        str(REPO_ROOT / "run_flash_searcher_mm_gaia.py"),
        "--infile",
        str(args.infile),
        "--outfile",
        str(results_path),
        "--task_indices",
        ",".join(str(i) for i in task_indices),
        "--seed",
        str(args.seed),
        "--token_budget",
        str(args.token_budget),
        "--judge_model",
        args.model,
        "--concurrency",
        str(args.concurrency),
        "--summary_interval",
        str(args.summary_interval),
        "--prompts_type",
        args.prompts_type,
        "--max_steps",
        str(args.max_steps),
        "--direct_output_dir",
        str(run_dir / "tasks"),
    ]
    if args.model:
        command.extend(["--model", args.model])
    if spec.memory_provider:
        command.extend([
            "--memory_provider",
            spec.memory_provider,
            "--enable_memory_evolution",
            "--shared_memory_provider",
        ])
    return command


def build_runner_env(run_root: Path, spec: RunSpec, args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("DEFAULT_MODEL", args.model)
    env.setdefault("DEFAULT_JUDGE_MODEL", args.model)
    env.setdefault("OPENAI_API_BASE", args.api_base)
    env.setdefault("JUDGE_API_BASE", args.api_base)
    env.setdefault("OPENAI_BASE_URL", args.api_base)
    env.setdefault("FORCE_STREAM", "1")
    env.setdefault("JUDGE_API_KEY", env.get("OPENAI_API_KEY", ""))
    env.setdefault("MEMORY_EMBEDDING_DEVICE", args.memory_embedding_device)
    for key, value in spec.extra_env.items():
        env[key] = value
    if spec.name == "gaia_workflow_shortcut_hybrid_semantic_json_full":
        env["MODULAR_STORAGE_DIR"] = str(run_root / "storage_gaia_workflow_shortcut_hybrid_semantic_json_full")
    return env


def run_remaining_tasks(run_root: Path, spec: RunSpec, task_indices: List[int], args: argparse.Namespace) -> int:
    run_dir = run_root / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    env = build_runner_env(run_root, spec, args)
    command = build_runner_command(run_dir, task_indices, spec, args)

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"[resume] remaining tasks: {task_indices}\n")
        log_file.write(f"[resume] command: {' '.join(command)}\n")
        log_file.write("=" * 80 + "\n")
        log_file.flush()
        process = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return process.returncode


def collect_task_results(tasks_dir: Path) -> List[Dict]:
    results: List[Dict] = []
    for path in sorted(tasks_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.name):
        with path.open("r", encoding="utf-8") as f:
            results.append(json.load(f))
    results.sort(key=lambda row: (row.get("item_index") is None, row.get("item_index", 0)))
    return results


def rebuild_run_outputs(run_root: Path, spec: RunSpec, selected_indices: Sequence[int]) -> Dict[str, int | float]:
    run_dir = run_root / spec.name
    tasks_dir = run_dir / "tasks"
    results = collect_task_results(tasks_dir)
    selected_set = set(selected_indices)
    results = [row for row in results if row.get("item_index") in selected_set]
    write_jsonl(run_dir / "results.jsonl", results)
    generate_unified_report(
        results,
        str(run_dir / "report.txt"),
        dataset_name="GAIA",
        has_levels=True,
        level_key="level",
    )

    total = len(results)
    correct = sum(1 for row in results if str(row.get("judgement", "")).strip().lower() == "correct")
    elapsed_seconds = sum(float(row.get("metrics", {}).get("elapsed_time", 0.0) or 0.0) for row in results)
    has_error = any(str(row.get("status", "")).strip().lower() == "error" for row in results)
    return {
        "config": spec.name,
        "tasks": total,
        "correct": correct,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "exit_code": 1 if has_error else 0,
    }


def write_root_summary(run_root: Path, rows: Sequence[Dict[str, int | float]]) -> None:
    summary_path = run_root / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["config", "tasks", "correct", "elapsed_seconds", "exit_code"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume unfinished GAIA workflow-vs-no-memory evaluation.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="Interrupted run directory.")
    parser.add_argument("--infile", type=Path, default=DEFAULT_INFILE, help="GAIA metadata jsonl file.")
    parser.add_argument("--task-selection", type=str, default=DEFAULT_TASK_SELECTION, help="Original GAIA task selection string.")
    parser.add_argument("--model", type=str, default=os.environ.get("DEFAULT_MODEL", DEFAULT_MODEL), help="Model id.")
    parser.add_argument("--api-base", type=str, default=os.environ.get("OPENAI_API_BASE", DEFAULT_API_BASE), help="API base URL.")
    parser.add_argument(
        "--runner-python",
        type=str,
        default=str(DEFAULT_RUNNER_PYTHON if DEFAULT_RUNNER_PYTHON.exists() else Path(sys.executable)),
        help="Python interpreter used to launch run_flash_searcher_mm_gaia.py.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Runner seed.")
    parser.add_argument("--token-budget", type=int, default=8192, help="Max completion tokens.")
    parser.add_argument("--concurrency", type=int, default=1, help="Runner concurrency.")
    parser.add_argument("--summary-interval", type=int, default=8, help="Agent summary interval.")
    parser.add_argument("--prompts-type", type=str, default="default", help="Prompt type.")
    parser.add_argument("--max-steps", type=int, default=40, help="Maximum steps.")
    parser.add_argument("--memory-embedding-device", type=str, default=os.environ.get("MEMORY_EMBEDDING_DEVICE", "cpu"), help="Embedding device for modular resume.")
    parser.add_argument(
        "--force-task-indices",
        type=str,
        default=None,
        help="Comma-separated global item indices to rerun even if task JSON already exists. Existing task files will be overwritten.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print remaining tasks without executing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env", override=False)

    data = load_gaia_data(args.infile)
    selected_indices = parse_task_selection(args.task_selection, data)
    forced_indices = [idx for idx in parse_optional_indices(args.force_task_indices) if idx in set(selected_indices)]
    if not selected_indices:
        raise SystemExit("No selected indices computed from task selection.")

    print(f"Run dir: {args.run_dir}")
    print(f"Original selection: {args.task_selection}")
    print(f"Selected tasks: {len(selected_indices)}")

    summary_rows: List[Dict[str, int | float]] = []
    nonzero_exit = 0

    for spec in RUN_SPECS:
        run_dir = args.run_dir / spec.name
        tasks_dir = run_dir / "tasks"
        completed = load_completed_indices(tasks_dir)
        completed_set = set(completed)
        remaining = [idx for idx in selected_indices if idx not in completed_set or idx in forced_indices]

        print(f"[{spec.name}] completed={len(completed_set)} remaining={len(remaining)}")
        if remaining:
            print(f"[{spec.name}] remaining indices: {remaining}")

        if remaining and not args.dry_run:
            exit_code = run_remaining_tasks(args.run_dir, spec, remaining, args)
            print(f"[{spec.name}] resume exit_code={exit_code}")
            if exit_code != 0:
                nonzero_exit = exit_code

        if run_dir.exists() and not args.dry_run:
            summary_rows.append(rebuild_run_outputs(args.run_dir, spec, selected_indices))

    if summary_rows and not args.dry_run:
        write_root_summary(args.run_dir, summary_rows)
        print(f"Rebuilt summary: {args.run_dir / 'summary.csv'}")

    return nonzero_exit


if __name__ == "__main__":
    raise SystemExit(main())
