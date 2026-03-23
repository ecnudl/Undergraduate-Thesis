#!/usr/bin/env python
"""
run_ablation.py — Ablation Study for 4-Module Memory System

Quantifies the marginal contribution of each memory module
(Extraction / Storage / Retrieval / Management) by running
one-factor-at-a-time ablation experiments.

Usage:
  # Dry run — print plan + cost estimate, no execution
  python experiments/scripts/run_ablation.py \
    --config experiments/configs/ablation/ablation_gaia_gpt5.json --dry-run

  # Sanity — 1 task per config smoke test
  python experiments/scripts/run_ablation.py \
    --config experiments/configs/ablation/ablation_gaia_gpt5.json --sanity-only

  # Full run with resume support
  python experiments/scripts/run_ablation.py \
    --config experiments/configs/ablation/ablation_gaia_gpt5.json --resume

  # Run only specific groups
  python experiments/scripts/run_ablation.py \
    --config experiments/configs/ablation/ablation_gaia_gpt5.json \
    --resume --groups extraction storage

  # Override task indices
  python experiments/scripts/run_ablation.py \
    --config experiments/configs/ablation/ablation_gaia_gpt5.json \
    --override-task-indices "[level1]1-5"

  # Parallel: 4 groups on 4 GPUs (~11h instead of ~37.5h)
  python experiments/scripts/run_ablation.py \
    --config experiments/configs/ablation/ablation_gaia_gpt5.json \
    --parallel --num-gpus 4

  # Parallel with sanity check first
  python experiments/scripts/run_ablation.py \
    --config experiments/configs/ablation/ablation_gaia_gpt5.json \
    --parallel --num-gpus 4 --sanity-only
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import logging
import os
import signal
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup: ensure we can import from the architecture search script
# and the common experiment modules.
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _EXPERIMENTS_DIR.parent

# Add experiments/ to sys.path so 'common.*' is importable
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))
# Add scripts/ so we can import from search_memory_architecture
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from search_memory_architecture import (
    ArchConfig,
    ExperimentResult,
    ExperimentRunner,
    RunRegistry,
    EXTRACTION_PRESETS,
    RESULT_COLUMNS,
    _write_results_csv,
    check_compatibility,
    is_config_valid,
)

logger = logging.getLogger("ablation")


# ======================================================================
# LockedRunRegistry — file-lock wrapper for concurrent access
# ======================================================================

class LockedRunRegistry(RunRegistry):
    """RunRegistry with file locking for safe concurrent access.

    Multiple GPU workers can share a single registry file. Reads and
    writes acquire an exclusive flock to prevent data corruption.
    """

    def __init__(self, path: Path):
        self.path = path
        self._data: Dict[str, Dict] = {}
        self._reload()

    def _reload(self) -> None:
        """Reload data from disk (call under lock or at init)."""
        if self.path.exists():
            with self.path.open("r") as f:
                try:
                    self._data = json.load(f)
                except json.JSONDecodeError:
                    self._data = {}

    def has(self, config_id: str) -> bool:
        self._reload()
        entry = self._data.get(config_id)
        return entry is not None and entry.get("status") == "completed"

    def get(self, config_id: str) -> Optional[ExperimentResult]:
        self._reload()
        entry = self._data.get(config_id)
        if entry is None:
            return None
        return ExperimentResult.from_dict(entry)

    def put(self, result: "ExperimentResult", key: str = "") -> None:
        lock_path = self.path.parent / (self.path.name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                self._reload()
                k = key or result.config.config_id
                self._data[k] = result.to_dict()
                self._save()
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        tmp.rename(self.path)

    def all_results(self) -> List["ExperimentResult"]:
        self._reload()
        return [ExperimentResult.from_dict(v) for v in self._data.values()]


# ======================================================================
# Constants
# ======================================================================

# Ablation CSV has extra columns beyond RESULT_COLUMNS
ABLATION_COLUMNS = [
    "config_id",
    "varied_value",
    "extraction",
    "storage",
    "retrieval",
    "management",
    "task_score",
    "total_tokens",
    "avg_latency",
    "memory_size",
    "inserted_units",
    "dedup_count",
    "retrieved_count",
    "graph_nodes",
    "graph_edges",
    "status",
    "num_tasks",
    "is_graph_sub",
]

# Estimates per task (based on gpt-5 empirical data)
EST_TOKENS_PER_TASK = 200_000
EST_SECONDS_PER_TASK = 450


# ======================================================================
# Config loading
# ======================================================================

def load_ablation_config(path: Path) -> Dict[str, Any]:
    """Load and validate ablation config JSON."""
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    required = ["ablation_name", "dataset", "runner", "anchor", "groups"]
    for key in required:
        if key not in config:
            raise ValueError(f"Missing required key '{key}' in ablation config")

    # Validate anchor
    anchor = config["anchor"]
    for mod in ("extraction", "storage", "retrieval", "management"):
        if mod not in anchor:
            raise ValueError(f"anchor missing '{mod}'")

    ok, reason = check_compatibility(
        anchor["storage"], anchor["retrieval"], anchor["management"]
    )
    if not ok:
        raise ValueError(f"Anchor config is incompatible: {reason}")

    return config


# ======================================================================
# Generate ablation configs per group
# ======================================================================

def generate_ablation_configs(
    config: Dict[str, Any],
) -> Dict[str, List[ArchConfig]]:
    """Generate ArchConfig lists for each ablation group.

    For each group, the 'vary' list specifies values for the varied module;
    all other modules are pinned to 'fixed' overrides (if any) or the anchor.
    Incompatible combinations are filtered out.
    """
    anchor = config["anchor"]
    groups_def = config.get("groups", {})
    result: Dict[str, List[ArchConfig]] = {}

    for group_name, group_spec in groups_def.items():
        vary_values = group_spec.get("vary", [])
        fixed = group_spec.get("fixed", {})

        # Determine which module is being varied based on group name
        varied_module = _group_to_module(group_name)

        configs: List[ArchConfig] = []
        seen_ids: set = set()

        for val in vary_values:
            # Start from anchor, apply fixed overrides, then apply varied value
            arch_dict = dict(anchor)
            arch_dict.update(fixed)
            arch_dict[varied_module] = val

            cfg = ArchConfig(**arch_dict)

            # Compatibility check
            ok, reason = is_config_valid(cfg)
            if not ok:
                logger.warning(
                    f"  [skip] {group_name}/{val}: {reason}"
                )
                continue

            # Dedup (anchor value appears in vary list for every group)
            if cfg.config_id not in seen_ids:
                seen_ids.add(cfg.config_id)
                configs.append(cfg)

        result[group_name] = configs

    return result


def _group_to_module(group_name: str) -> str:
    """Map group name to the module being varied."""
    mapping = {
        "extraction": "extraction",
        "storage": "storage",
        "retrieval": "retrieval",
        "management": "management",
        "retrieval_graph": "retrieval",
        "management_graph": "management",
    }
    if group_name not in mapping:
        raise ValueError(
            f"Unknown group '{group_name}'. Expected one of: {sorted(mapping)}"
        )
    return mapping[group_name]


# ======================================================================
# Materialize per-config JSON files (for reproducibility)
# ======================================================================

def materialize_experiment_configs(
    config: Dict[str, Any],
    groups: Dict[str, List[ArchConfig]],
    output_dir: Path,
) -> Dict[str, List[Path]]:
    """Write individual JSON config files for each experiment config.

    Returns mapping of group_name -> list of written file paths.
    """
    generated: Dict[str, List[Path]] = {}

    for group_name, configs in groups.items():
        group_dir = output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []

        for cfg in configs:
            file_data = {
                "ablation_name": config["ablation_name"],
                "group": group_name,
                "varied_module": _group_to_module(group_name),
                "varied_value": getattr(cfg, _group_to_module(group_name)),
                "config_id": cfg.config_id,
                "short_name": cfg.short_name,
                "arch_config": cfg.to_dict(),
                "dataset": config["dataset"],
                "runner": config["runner"],
            }
            path = group_dir / f"{cfg.short_name}.json"
            with path.open("w", encoding="utf-8") as f:
                json.dump(file_data, f, indent=2, ensure_ascii=False)
            paths.append(path)

        generated[group_name] = paths

    return generated


# ======================================================================
# Run ablation group
# ======================================================================

def run_ablation_group(
    runner: ExperimentRunner,
    group_name: str,
    configs: List[ArchConfig],
    task_indices: str,
    ablation_name: str,
) -> List[Tuple[ArchConfig, ExperimentResult]]:
    """Execute all configs in a group sequentially. Returns (config, result) pairs."""
    results: List[Tuple[ArchConfig, ExperimentResult]] = []

    logger.info(f"=== Group: {group_name} ({len(configs)} configs) ===")

    for i, cfg in enumerate(configs, 1):
        logger.info(
            f"  [{i}/{len(configs)}] {cfg.short_name}"
        )
        run_label = f"{ablation_name}/ablation/{group_name}"
        result = runner.run(
            cfg,
            run_label=run_label,
            task_indices=task_indices,
            id_prefix=f"abl_{group_name}_",
        )
        results.append((cfg, result))

    return results


# ======================================================================
# CSV output
# ======================================================================

def write_ablation_csv(
    path: Path,
    results: List[Tuple[str, ArchConfig, ExperimentResult, bool]],
    group_name: str,
) -> None:
    """Write ablation CSV for a group.

    Each entry is (varied_value, config, result, is_graph_sub).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ABLATION_COLUMNS)
        writer.writeheader()
        for varied_val, cfg, result, is_graph in results:
            row = result.to_dict()
            row["varied_value"] = varied_val
            row["is_graph_sub"] = is_graph
            writer.writerow({k: row.get(k) for k in ABLATION_COLUMNS})


def merge_csv_outputs(
    group_results: Dict[str, List[Tuple[str, ArchConfig, ExperimentResult, bool]]],
    groups_def: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Path]:
    """Write ablation CSVs, merging graph sub-experiments into their csv_target.

    Returns mapping of csv_name -> output path.
    """
    # Collect rows by target CSV name
    csv_buckets: Dict[str, List[Tuple[str, ArchConfig, ExperimentResult, bool]]] = {}

    for group_name, rows in group_results.items():
        csv_target = groups_def.get(group_name, {}).get("csv_target", group_name)
        if csv_target not in csv_buckets:
            csv_buckets[csv_target] = []
        csv_buckets[csv_target].extend(rows)

    written: Dict[str, Path] = {}
    for csv_name, rows in csv_buckets.items():
        path = output_dir / f"ablation_{csv_name}.csv"
        write_ablation_csv(path, rows, csv_name)
        logger.info(f"  Wrote {path} ({len(rows)} rows)")
        written[csv_name] = path

    return written


# ======================================================================
# Summary JSON
# ======================================================================

def write_ablation_summary(
    output_dir: Path,
    config: Dict[str, Any],
    group_results: Dict[str, List[Tuple[str, ArchConfig, ExperimentResult, bool]]],
    baseline_result: Optional[ExperimentResult],
    elapsed_seconds: float,
) -> Path:
    """Write ablation_summary.json with high-level stats."""
    summary: Dict[str, Any] = {
        "ablation_name": config["ablation_name"],
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 1),
        "anchor": config["anchor"],
        "dataset": config["dataset"],
        "baseline": baseline_result.to_dict() if baseline_result else None,
        "groups": {},
    }

    for group_name, rows in group_results.items():
        group_summary = []
        for varied_val, cfg, result, is_graph in rows:
            group_summary.append({
                "varied_value": varied_val,
                "config_id": cfg.config_id,
                "short_name": cfg.short_name,
                "task_score": result.task_score,
                "total_tokens": result.total_tokens,
                "avg_latency": result.avg_latency,
                "status": result.status,
                "num_tasks": result.num_tasks,
                "is_graph_sub": is_graph,
            })
        summary["groups"][group_name] = group_summary

    path = output_dir / "ablation_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return path


# ======================================================================
# Cost estimation
# ======================================================================

def estimate_cost(
    config: Dict[str, Any],
    groups: Dict[str, List[ArchConfig]],
    task_indices: str,
    include_baseline: bool = True,
) -> Dict[str, Any]:
    """Estimate time and token cost for the ablation study."""
    num_tasks = _count_tasks(task_indices)

    # Deduplicate configs across groups (anchor is shared)
    unique_ids: set = set()
    unique_count = 0
    per_group: Dict[str, int] = {}

    for group_name, configs in groups.items():
        group_unique = 0
        for cfg in configs:
            if cfg.config_id not in unique_ids:
                unique_ids.add(cfg.config_id)
                unique_count += 1
                group_unique += 1
        per_group[group_name] = group_unique

    if include_baseline:
        unique_count += 1  # no_memory baseline

    total_runs = unique_count
    total_tasks = total_runs * num_tasks
    total_tokens = total_tasks * EST_TOKENS_PER_TASK
    total_seconds = total_tasks * EST_SECONDS_PER_TASK
    total_hours = total_seconds / 3600

    return {
        "num_tasks_per_config": num_tasks,
        "unique_configs": unique_count,
        "per_group_unique": per_group,
        "include_baseline": include_baseline,
        "total_task_runs": total_tasks,
        "est_total_tokens": total_tokens,
        "est_total_hours": round(total_hours, 1),
        "est_seconds_per_task": EST_SECONDS_PER_TASK,
        "est_tokens_per_task": EST_TOKENS_PER_TASK,
    }


def _count_tasks(task_indices: str) -> int:
    """Rough estimate of task count from index string."""
    count = 0
    for part in task_indices.strip().split():
        # Skip level markers like [level1]
        if part.startswith("[") and "]" in part:
            bracket_end = part.index("]")
            part = part[bracket_end + 1:]
        if not part:
            continue
        for segment in part.split(","):
            if "-" in segment:
                start, end = segment.split("-", 1)
                try:
                    count += int(end) - int(start) + 1
                except ValueError:
                    count += 1
            else:
                try:
                    int(segment)
                    count += 1
                except ValueError:
                    pass
    return max(count, 1)


# ======================================================================
# Plan printing (dry-run)
# ======================================================================

def print_plan(
    config: Dict[str, Any],
    groups: Dict[str, List[ArchConfig]],
    estimates: Dict[str, Any],
    baseline: bool = True,
) -> None:
    """Pretty-print the execution plan."""
    print("\n" + "=" * 70)
    print(f"  ABLATION STUDY PLAN: {config['ablation_name']}")
    print("=" * 70)

    print(f"\n  Dataset: {config['dataset']['name']}")
    print(f"  Task indices: {config['dataset'].get('task_indices', 'N/A')}")
    print(f"  Model: {config['runner'].get('model', 'N/A')}")
    print(f"\n  Anchor: {config['anchor']}")

    if baseline:
        print(f"\n  Baseline: no_memory (1 config)")

    print(f"\n  Groups:")
    all_configs: Dict[str, ArchConfig] = {}
    for group_name, configs in groups.items():
        csv_target = config.get("groups", {}).get(group_name, {}).get("csv_target")
        suffix = f" (→ csv: {csv_target})" if csv_target else ""
        print(f"    {group_name}: {len(configs)} configs{suffix}")
        for cfg in configs:
            varied = _group_to_module(group_name)
            val = getattr(cfg, varied)
            is_anchor = (cfg.to_dict() == config["anchor"])
            anchor_mark = " ⚓" if is_anchor else ""
            dup = " (shared)" if cfg.config_id in all_configs else ""
            print(f"      - {val}{anchor_mark}{dup}: {cfg.short_name}")
            all_configs[cfg.config_id] = cfg

    print(f"\n  Estimates:")
    print(f"    Unique configs: {estimates['unique_configs']}")
    print(f"    Tasks per config: {estimates['num_tasks_per_config']}")
    print(f"    Total task runs: {estimates['total_task_runs']}")
    print(f"    Est. total tokens: {estimates['est_total_tokens']:,}")
    print(f"    Est. total time: {estimates['est_total_hours']:.1f}h")
    print()


# ======================================================================
# Main orchestrator
# ======================================================================

def run_ablation(
    config: Dict[str, Any],
    dry_run: bool = False,
    resume: bool = False,
    sanity_only: bool = False,
    selected_groups: Optional[List[str]] = None,
    override_task_indices: Optional[str] = None,
    gpu: Optional[int] = None,
) -> None:
    """Main entry point for the ablation study."""
    ablation_name = config["ablation_name"]
    output_root = Path(config.get("output_root", "./experiments/results/ablation"))
    if not output_root.is_absolute():
        output_root = _REPO_ROOT / output_root
    output_dir = output_root / ablation_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # GPU assignment
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        logger.info(f"GPU: CUDA_VISIBLE_DEVICES={gpu}")

    # Determine task indices
    task_indices = override_task_indices or config["dataset"].get(
        "task_indices", "[level1]1-20 [level2]1-10"
    )
    if sanity_only:
        # Use only 1 task for smoke testing
        task_indices = _make_sanity_indices(task_indices)
        logger.info(f"Sanity mode: using task_indices={task_indices}")

    # Generate configs
    all_groups = generate_ablation_configs(config)
    if selected_groups is not None:
        all_groups = {
            k: v for k, v in all_groups.items() if k in selected_groups
        }
        if selected_groups:
            missing = set(selected_groups) - set(all_groups)
            if missing:
                logger.warning(f"Requested groups not found: {missing}")

    include_baseline = config.get("run_no_memory_baseline", True)
    if selected_groups is not None and len(selected_groups) > 0:
        # When running specific groups, skip baseline (parallel mode runs it separately)
        include_baseline = False
    elif selected_groups is not None and len(selected_groups) == 0:
        # --baseline-only mode: run only baseline, no groups
        include_baseline = True

    # Cost estimate
    estimates = estimate_cost(config, all_groups, task_indices, include_baseline)

    # Print plan
    print_plan(config, all_groups, estimates, include_baseline)

    if dry_run:
        # Materialize configs for inspection
        gen_dir = _EXPERIMENTS_DIR / "configs" / "ablation" / "generated"
        materialized = materialize_experiment_configs(config, all_groups, gen_dir)
        for gname, paths in materialized.items():
            print(f"  Generated {len(paths)} config(s) in {paths[0].parent}")

        print("\n  [dry-run] Generating simulated results...")

    # Save config snapshot
    with (output_dir / "ablation_config.json").open("w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Set up runner (use LockedRunRegistry for concurrent safety)
    registry = LockedRunRegistry(output_dir / "ablation_registry.json")
    runner = ExperimentRunner(
        search_config=config,
        repo_root=_REPO_ROOT,
        output_root=output_dir,
        registry=registry,
        dry_run=dry_run,
    )

    # Materialize per-config JSONs
    gen_dir = _EXPERIMENTS_DIR / "configs" / "ablation" / "generated"
    materialize_experiment_configs(config, all_groups, gen_dir)

    start_time = time.time()
    group_results: Dict[str, List[Tuple[str, ArchConfig, ExperimentResult, bool]]] = {}

    # --- Baseline ---
    baseline_result: Optional[ExperimentResult] = None
    if include_baseline:
        logger.info("=== Baseline: no_memory ===")
        baseline_result = runner.run_baseline(task_indices=task_indices)

    # --- Ablation groups ---
    # Track anchor result to avoid re-running
    anchor_cfg = ArchConfig(**config["anchor"])
    anchor_result: Optional[ExperimentResult] = None

    for group_name, configs in all_groups.items():
        is_graph_sub = "csv_target" in config.get("groups", {}).get(group_name, {})
        varied_module = _group_to_module(group_name)

        raw_results = run_ablation_group(
            runner, group_name, configs, task_indices, ablation_name
        )

        # Package results with metadata
        rows: List[Tuple[str, ArchConfig, ExperimentResult, bool]] = []
        for cfg, result in raw_results:
            varied_val = getattr(cfg, varied_module)
            rows.append((varied_val, cfg, result, is_graph_sub))

            # Track anchor
            if cfg.config_id == anchor_cfg.config_id and anchor_result is None:
                anchor_result = result

        group_results[group_name] = rows

    elapsed = time.time() - start_time

    # --- Write CSVs ---
    logger.info("=== Writing results ===")

    # Add baseline to all non-graph main groups
    if baseline_result is not None:
        for group_name in list(group_results.keys()):
            if "graph" not in group_name:
                group_results[group_name].insert(
                    0,
                    ("no_memory", baseline_result.config, baseline_result, False),
                )

    csv_paths = merge_csv_outputs(
        group_results, config.get("groups", {}), output_dir
    )

    # --- Write summary ---
    summary_path = write_ablation_summary(
        output_dir, config, group_results, baseline_result, elapsed
    )
    logger.info(f"  Summary: {summary_path}")

    # --- Final report ---
    print("\n" + "=" * 70)
    print(f"  ABLATION COMPLETE: {ablation_name}")
    print("=" * 70)
    print(f"  Elapsed: {elapsed / 3600:.1f}h ({elapsed:.0f}s)")
    print(f"  Output: {output_dir}")
    for csv_name, csv_path in csv_paths.items():
        print(f"  CSV: {csv_path}")
    print(f"  Summary: {summary_path}")

    if baseline_result:
        print(f"\n  Baseline score: {baseline_result.task_score:.3f}")
    if anchor_result:
        print(f"  Anchor score:   {anchor_result.task_score:.3f}")

    # Print per-group best
    print(f"\n  Per-group results:")
    for group_name, rows in group_results.items():
        scored = [(r.task_score, varied, r) for varied, _, r, _ in rows if r.status in ("completed", "dry_run")]
        if scored:
            scored.sort(key=lambda x: -x[0])
            best_score, best_val, best_r = scored[0]
            print(f"    {group_name}: best={best_val} ({best_score:.3f})")

    print()


# ======================================================================
# Parallel orchestration
# ======================================================================

def _assign_groups_to_gpus(
    groups: Dict[str, List[ArchConfig]],
    num_gpus: int,
) -> List[List[str]]:
    """Distribute groups across GPUs, balancing by config count.

    Returns list-of-lists: gpu_assignments[gpu_id] = [group_names].
    """
    # Sort groups by config count (descending) for greedy balancing
    sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))
    gpu_loads = [0] * num_gpus
    gpu_assignments: List[List[str]] = [[] for _ in range(num_gpus)]

    for group_name, configs in sorted_groups:
        # Assign to least-loaded GPU
        min_gpu = gpu_loads.index(min(gpu_loads))
        gpu_assignments[min_gpu].append(group_name)
        gpu_loads[min_gpu] += len(configs)

    return gpu_assignments


def run_parallel(
    config: Dict[str, Any],
    config_path: Path,
    num_gpus: int = 4,
    gpu_ids: Optional[List[int]] = None,
    dry_run: bool = False,
    sanity_only: bool = False,
    selected_groups: Optional[List[str]] = None,
    override_task_indices: Optional[str] = None,
) -> None:
    """Parallel orchestrator: baseline first, then groups on separate GPUs.

    Args:
        gpu_ids: Explicit GPU device IDs (e.g. [4,5,6,7]). If None, uses 0..num_gpus-1.
    """
    if gpu_ids is None:
        gpu_ids = list(range(num_gpus))
    else:
        num_gpus = len(gpu_ids)
    ablation_name = config["ablation_name"]
    output_root = Path(config.get("output_root", "./experiments/results/ablation"))
    if not output_root.is_absolute():
        output_root = _REPO_ROOT / output_root
    output_dir = output_root / ablation_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine task indices
    task_indices = override_task_indices or config["dataset"].get(
        "task_indices", "[level1]1-20 [level2]1-10"
    )
    if sanity_only:
        task_indices = _make_sanity_indices(task_indices)

    # Generate configs for plan display
    all_groups = generate_ablation_configs(config)
    if selected_groups:
        all_groups = {k: v for k, v in all_groups.items() if k in selected_groups}

    include_baseline = config.get("run_no_memory_baseline", True)

    # Assign groups to GPUs
    gpu_assignments = _assign_groups_to_gpus(all_groups, num_gpus)

    # Print parallel plan
    estimates = estimate_cost(config, all_groups, task_indices, include_baseline)
    print_plan(config, all_groups, estimates, include_baseline)

    print("  PARALLEL MODE:")
    print(f"    GPUs: {gpu_ids}")
    max_per_gpu = 0
    for slot, group_names in enumerate(gpu_assignments):
        if not group_names:
            continue
        n_configs = sum(len(all_groups[g]) for g in group_names)
        n_tasks = n_configs * _count_tasks(task_indices)
        gpu_hours = n_tasks * EST_SECONDS_PER_TASK / 3600
        max_per_gpu = max(max_per_gpu, gpu_hours)
        print(f"    GPU {gpu_ids[slot]}: {group_names} ({n_configs} configs, ~{gpu_hours:.1f}h)")

    baseline_hours = _count_tasks(task_indices) * EST_SECONDS_PER_TASK / 3600 if include_baseline else 0
    print(f"    Est. wall time: ~{baseline_hours + max_per_gpu:.1f}h "
          f"(baseline {baseline_hours:.1f}h + parallel {max_per_gpu:.1f}h)")
    print()

    if dry_run:
        print("  [dry-run] Would spawn parallel workers. Exiting.\n")
        return

    start_time = time.time()

    # Phase 1: Run baseline on first GPU (sequential)
    if include_baseline:
        logger.info(f"=== Phase 1: Baseline (GPU {gpu_ids[0]}) ===")
        baseline_cmd = _build_worker_cmd(
            config_path, groups=[], gpu=gpu_ids[0], sanity_only=sanity_only,
            override_task_indices=override_task_indices,
            run_baseline_only=True,
        )
        _run_subprocess(baseline_cmd, "baseline", output_dir / "worker_baseline.log")

    # Phase 2: Parallel group execution
    logger.info("=== Phase 2: Parallel groups ===")
    workers: List[Tuple[str, subprocess.Popen, Path]] = []

    for slot, group_names in enumerate(gpu_assignments):
        if not group_names:
            continue
        device_id = gpu_ids[slot]
        label = f"gpu{device_id}_{'_'.join(group_names)}"
        log_path = output_dir / f"worker_{label}.log"
        cmd = _build_worker_cmd(
            config_path, groups=group_names, gpu=device_id,
            sanity_only=sanity_only,
            override_task_indices=override_task_indices,
        )
        logger.info(f"  Launching worker: GPU {device_id} -> {group_names}")
        log_f = log_path.open("w")
        proc = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT,
            cwd=str(_REPO_ROOT),
        )
        workers.append((label, proc, log_path, log_f))

    # Wait for all workers
    failed = []
    for label, proc, log_path, log_f in workers:
        proc.wait()
        log_f.close()
        if proc.returncode != 0:
            failed.append(label)
            logger.error(f"  Worker {label} failed (exit {proc.returncode}). See {log_path}")
        else:
            logger.info(f"  Worker {label} completed.")

    if failed:
        logger.warning(f"  {len(failed)} worker(s) failed: {failed}")
        logger.warning(f"  Re-run with --parallel --resume to retry failed groups.")

    elapsed = time.time() - start_time

    # Phase 3: Merge results from shared registry
    logger.info("=== Phase 3: Merging results ===")
    _merge_parallel_results(config, output_dir, all_groups, include_baseline, elapsed)

    print(f"\n  Parallel run completed in {elapsed / 3600:.1f}h ({elapsed:.0f}s)")
    if failed:
        print(f"  WARNING: {len(failed)} worker(s) failed. Check logs in {output_dir}")
    print()


def _build_worker_cmd(
    config_path: Path,
    groups: List[str],
    gpu: int,
    sanity_only: bool = False,
    override_task_indices: Optional[str] = None,
    run_baseline_only: bool = False,
) -> List[str]:
    """Build command to launch a worker subprocess."""
    cmd = [
        sys.executable, str(_SCRIPT_DIR / "run_ablation.py"),
        "--config", str(config_path),
        "--gpu", str(gpu),
    ]
    if run_baseline_only:
        cmd.append("--baseline-only")
    elif groups:
        cmd.extend(["--groups"] + groups)
    if sanity_only:
        cmd.append("--sanity-only")
    if override_task_indices:
        cmd.extend(["--override-task-indices", override_task_indices])
    return cmd


def _run_subprocess(cmd: List[str], label: str, log_path: Path) -> int:
    """Run a subprocess and wait for it to complete."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_f:
        proc = subprocess.run(
            cmd, stdout=log_f, stderr=subprocess.STDOUT,
            cwd=str(_REPO_ROOT),
        )
    if proc.returncode != 0:
        logger.error(f"  Worker {label} failed (exit {proc.returncode}). See {log_path}")
    else:
        logger.info(f"  Worker {label} completed.")
    return proc.returncode


def _merge_parallel_results(
    config: Dict[str, Any],
    output_dir: Path,
    all_groups: Dict[str, List[ArchConfig]],
    include_baseline: bool,
    elapsed: float,
) -> None:
    """Read the shared registry and produce merged CSVs + summary."""
    registry = LockedRunRegistry(output_dir / "ablation_registry.json")

    # Reconstruct group_results from registry
    group_results: Dict[str, List[Tuple[str, ArchConfig, ExperimentResult, bool]]] = {}
    anchor_result: Optional[ExperimentResult] = None
    anchor_cfg = ArchConfig(**config["anchor"])

    for group_name, configs in all_groups.items():
        is_graph_sub = "csv_target" in config.get("groups", {}).get(group_name, {})
        varied_module = _group_to_module(group_name)
        rows: List[Tuple[str, ArchConfig, ExperimentResult, bool]] = []

        for cfg in configs:
            cache_id = f"abl_{group_name}_{cfg.config_id}"
            result = registry.get(cache_id)
            if result is None:
                result = ExperimentResult(config=cfg, status="missing")
            varied_val = getattr(cfg, varied_module)
            rows.append((varied_val, cfg, result, is_graph_sub))

            if cfg.config_id == anchor_cfg.config_id and anchor_result is None:
                anchor_result = result

        group_results[group_name] = rows

    # Baseline
    baseline_result: Optional[ExperimentResult] = None
    if include_baseline:
        baseline_cfg = ArchConfig(
            extraction="no_memory", storage="none",
            retrieval="none", management="none",
        )
        baseline_id = "baseline_" + baseline_cfg.config_id
        baseline_result = registry.get(baseline_id)
        if baseline_result is not None:
            for group_name in list(group_results.keys()):
                if "graph" not in group_name:
                    group_results[group_name].insert(
                        0,
                        ("no_memory", baseline_result.config, baseline_result, False),
                    )

    csv_paths = merge_csv_outputs(
        group_results, config.get("groups", {}), output_dir
    )
    summary_path = write_ablation_summary(
        output_dir, config, group_results, baseline_result, elapsed
    )

    # Final report
    print("\n" + "=" * 70)
    print(f"  ABLATION COMPLETE (PARALLEL): {config['ablation_name']}")
    print("=" * 70)
    print(f"  Elapsed: {elapsed / 3600:.1f}h ({elapsed:.0f}s)")
    print(f"  Output: {output_dir}")
    for csv_name, csv_path in csv_paths.items():
        print(f"  CSV: {csv_path}")
    print(f"  Summary: {summary_path}")

    if baseline_result:
        print(f"\n  Baseline score: {baseline_result.task_score:.3f}")
    if anchor_result and anchor_result.status != "missing":
        print(f"  Anchor score:   {anchor_result.task_score:.3f}")

    print(f"\n  Per-group results:")
    for group_name, rows in group_results.items():
        scored = [
            (r.task_score, varied, r) for varied, _, r, _ in rows
            if r.status in ("completed", "dry_run")
        ]
        if scored:
            scored.sort(key=lambda x: -x[0])
            best_score, best_val, _ = scored[0]
            print(f"    {group_name}: best={best_val} ({best_score:.3f})")


def _make_sanity_indices(full_indices: str) -> str:
    """Extract just 1 task from the full indices for sanity testing."""
    parts = full_indices.strip().split()
    # Find first concrete index
    for i, part in enumerate(parts):
        if part.startswith("[") and "]" in part:
            bracket_end = part.index("]")
            remainder = part[bracket_end + 1:]
            prefix = part[:bracket_end + 1]
            if remainder:
                # e.g. [level1]1-20 -> [level1]1
                first = remainder.split(",")[0].split("-")[0]
                return f"{prefix}{first}"
            # Check next part for numbers
            if i + 1 < len(parts) and not parts[i + 1].startswith("["):
                first = parts[i + 1].split(",")[0].split("-")[0]
                return f"{prefix}{first}"
        elif not part.startswith("["):
            first = part.split(",")[0].split("-")[0]
            return first
    return "1"


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study for memory system modules"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to ablation config JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan and simulate results without executing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip completed configs)",
    )
    parser.add_argument(
        "--sanity-only",
        action="store_true",
        help="Run only 1 task per config for smoke testing",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="Only run specified groups (e.g., --groups extraction storage)",
    )
    parser.add_argument(
        "--override-task-indices",
        type=str,
        default=None,
        help="Override task indices from config",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run groups in parallel across multiple GPUs",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="Number of GPUs for parallel mode (default: 4). Ignored if --gpu-ids is set.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU device IDs (e.g., '4,5,6,7'). Overrides --num-gpus.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU ID to use (sets CUDA_VISIBLE_DEVICES). Used internally by parallel workers.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only run the no-memory baseline (used internally by parallel mode)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _REPO_ROOT / config_path

    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    config = load_ablation_config(config_path)

    if args.parallel:
        # Parse gpu_ids
        gpu_ids = None
        if args.gpu_ids:
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
        # Parallel mode: orchestrate workers
        run_parallel(
            config=config,
            config_path=config_path,
            num_gpus=args.num_gpus,
            gpu_ids=gpu_ids,
            dry_run=args.dry_run,
            sanity_only=args.sanity_only,
            selected_groups=args.groups,
            override_task_indices=args.override_task_indices,
        )
    elif args.baseline_only:
        # Internal: worker mode for baseline only
        run_ablation(
            config=config,
            dry_run=args.dry_run,
            resume=True,
            sanity_only=args.sanity_only,
            selected_groups=[],  # empty → no groups, baseline logic below
            override_task_indices=args.override_task_indices,
            gpu=args.gpu,
        )
    else:
        # Sequential mode (single group or all groups)
        run_ablation(
            config=config,
            dry_run=args.dry_run,
            resume=args.resume,
            sanity_only=args.sanity_only,
            selected_groups=args.groups,
            override_task_indices=args.override_task_indices,
            gpu=args.gpu,
        )


if __name__ == "__main__":
    main()
