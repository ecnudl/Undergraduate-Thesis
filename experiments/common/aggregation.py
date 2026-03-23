"""Aggregate experiment outputs into CSV tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


TASK_COLUMNS = [
    "run_name",
    "dataset",
    "split",
    "agent",
    "model",
    "seed",
    "task_id",
    "item_index",
    "status",
    "judgement",
    "task_score",
    "success",
    "latency",
    "token_usage",
    "prompt_tokens",
    "completion_tokens",
    "api_calls",
    "num_memory_units",
    "num_inserted",
    "num_deduped",
    "num_retrieved",
    "retrieval_calls",
    "management_ops_triggered",
    "graph_nodes",
    "graph_edges",
]


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_run_results(run_dir: Path) -> List[Dict]:
    """Load results from the canonical results.jsonl or task JSONs."""
    results_path = run_dir / "results.jsonl"
    if results_path.exists():
        return _read_jsonl(results_path)

    task_dir = run_dir / "tasks"
    rows: List[Dict] = []
    for path in sorted(task_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def _load_manifest(run_dir: Path) -> Dict:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _task_score(result: Dict) -> float:
    if result.get("task_score") is not None:
        return float(result["task_score"])
    if result.get("score") is not None:
        return float(result["score"])
    if str(result.get("judgement", "")).strip().lower() == "correct":
        return 1.0
    return 0.0


def _success(result: Dict) -> bool:
    if result.get("success") is not None:
        return bool(result["success"])
    return str(result.get("judgement", "")).strip().lower() == "correct"


def _flatten_result(result: Dict, manifest: Dict) -> Dict:
    config = manifest.get("config", {})
    metrics = result.get("metrics", {})
    memory_metrics = result.get("memory_metrics", {})
    return {
        "run_name": manifest.get("run_name", run_dir_name(manifest)),
        "dataset": config.get("dataset"),
        "split": config.get("split"),
        "agent": config.get("agent"),
        "model": config.get("model"),
        "seed": config.get("seed"),
        "task_id": result.get("task_id"),
        "item_index": result.get("item_index"),
        "status": result.get("status"),
        "judgement": result.get("judgement"),
        "task_score": _task_score(result),
        "success": _success(result),
        "latency": metrics.get("elapsed_time", 0.0),
        "token_usage": metrics.get("total_tokens", 0),
        "prompt_tokens": metrics.get("prompt_tokens", 0),
        "completion_tokens": metrics.get("completion_tokens", 0),
        "api_calls": metrics.get("api_calls", 0),
        "num_memory_units": memory_metrics.get("num_memory_units", 0),
        "num_inserted": memory_metrics.get("num_inserted", 0),
        "num_deduped": memory_metrics.get("num_deduped", 0),
        "num_retrieved": memory_metrics.get("num_retrieved", 0),
        "retrieval_calls": memory_metrics.get("retrieval_calls", 0),
        "management_ops_triggered": memory_metrics.get("management_ops_triggered", 0),
        "graph_nodes": memory_metrics.get("graph_nodes"),
        "graph_edges": memory_metrics.get("graph_edges"),
    }


def run_dir_name(manifest: Dict) -> str:
    return manifest.get("run_name") or manifest.get("run_dir", "")


def _write_csv(path: Path, rows: Iterable[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_run(run_dir: Path) -> Dict:
    """Write per-task and summary CSV files for a run."""
    manifest = _load_manifest(run_dir)
    results = load_run_results(run_dir)
    rows = [_flatten_result(result, manifest) for result in results]
    task_csv = run_dir / "task_metrics.csv"
    _write_csv(task_csv, rows, TASK_COLUMNS)

    total = len(rows)
    success_count = sum(1 for row in rows if row["success"])
    summary = {
        "run_name": manifest.get("run_name", run_dir.name),
        "dataset": manifest.get("config", {}).get("dataset"),
        "split": manifest.get("config", {}).get("split"),
        "agent": manifest.get("config", {}).get("agent"),
        "model": manifest.get("config", {}).get("model"),
        "seed": manifest.get("config", {}).get("seed"),
        "tasks": total,
        "successes": success_count,
        "success_rate": (success_count / total) if total else 0.0,
        "avg_task_score": (sum(row["task_score"] for row in rows) / total) if total else 0.0,
        "total_tokens": sum(row["token_usage"] for row in rows),
        "avg_latency": (sum(row["latency"] for row in rows) / total) if total else 0.0,
        "final_num_memory_units": rows[-1]["num_memory_units"] if rows else 0,
        "total_inserted": sum(row["num_inserted"] for row in rows),
        "total_deduped": sum(row["num_deduped"] for row in rows),
        "total_retrieved": sum(row["num_retrieved"] for row in rows),
        "total_retrieval_calls": sum(row["retrieval_calls"] for row in rows),
        "total_management_ops_triggered": sum(row["management_ops_triggered"] for row in rows),
        "max_graph_nodes": max((row["graph_nodes"] or 0 for row in rows), default=0),
        "max_graph_edges": max((row["graph_edges"] or 0 for row in rows), default=0),
    }
    summary_csv = run_dir / "summary.csv"
    _write_csv(summary_csv, [summary], list(summary.keys()))
    return {"task_csv": str(task_csv), "summary_csv": str(summary_csv), "summary": summary}
