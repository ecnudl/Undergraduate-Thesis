#!/usr/bin/env python
"""
search_memory_architecture.py — Hierarchical Evolutionary Architecture Search

Explores the optimal memory architecture composition across 4 modules:
  Extraction × Storage × Retrieval × Management

Phases:
  A: Module Screening     — one-factor-at-a-time to narrow search space
  B: Reduced Space Search  — exhaustive search over retained candidates
  C: Evolutionary Refinement — mutation/crossover around top elites
  D: Final Validation      — held-out evaluation of best architectures

Usage:
  # Dry run (no experiments executed)
  python experiments/scripts/search_memory_architecture.py --dry-run

  # Smoke test (1-task proxy)
  python experiments/scripts/search_memory_architecture.py \
    --config experiments/configs/architecture_search/default_search.json \
    --override-proxy-indices "[level1]1"

  # Full search
  python experiments/scripts/search_memory_architecture.py \
    --config experiments/configs/architecture_search/default_search.json

  # Resume interrupted search
  python experiments/scripts/search_memory_architecture.py \
    --config experiments/configs/architecture_search/default_search.json \
    --resume
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import logging
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("arch_search")

# ======================================================================
# Constants: Search Space
# ======================================================================

EXTRACTION_PRESETS: Dict[str, List[str]] = {
    "insight_tip": ["tip", "insight"],
    "trajectory_only": ["trajectory"],
    "workflow_shortcut": ["workflow", "shortcut"],
    "mixed_all": ["tip", "insight", "trajectory", "workflow", "shortcut"],
}

STORAGE_CANDIDATES = ["json", "vector", "hybrid", "graph", "llm_graph"]
RETRIEVAL_CANDIDATES = [
    "semantic", "keyword", "hybrid", "graph", "contrastive", "hybrid_graph",
]
MANAGEMENT_CANDIDATES = ["none", "lightweight", "json_basic", "json_full", "graph_full"]

GRAPH_STORAGE_TYPES = {"graph", "llm_graph"}
GRAPH_RETRIEVER_TYPES = {"graph", "hybrid_graph"}
GRAPH_MANAGEMENT_PRESETS = {"graph_full"}

DEFAULT_ANCHOR = {
    "extraction": "mixed_all",
    "storage": "hybrid",
    "retrieval": "hybrid",
    "management": "json_full",
}


# ======================================================================
# ArchConfig — a single architecture configuration
# ======================================================================

@dataclass
class ArchConfig:
    extraction: str
    storage: str
    retrieval: str
    management: str

    @property
    def config_id(self) -> str:
        raw = f"{self.extraction}|{self.storage}|{self.retrieval}|{self.management}"
        return hashlib.md5(raw.encode()).hexdigest()[:10]

    @property
    def short_name(self) -> str:
        return f"{self.extraction}__{self.storage}__{self.retrieval}__{self.management}"

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "ArchConfig":
        return ArchConfig(
            extraction=d["extraction"],
            storage=d["storage"],
            retrieval=d["retrieval"],
            management=d["management"],
        )


# ======================================================================
# Compatibility checking
# ======================================================================

def check_compatibility(
    storage: str, retrieval: str, management: str,
) -> Tuple[bool, str]:
    """Return (is_valid, reason) for a storage-retrieval-management combo."""
    if retrieval in GRAPH_RETRIEVER_TYPES and storage not in GRAPH_STORAGE_TYPES:
        return False, f"Retriever '{retrieval}' requires graph storage, got '{storage}'"
    if management in GRAPH_MANAGEMENT_PRESETS and storage not in GRAPH_STORAGE_TYPES:
        return False, f"Management '{management}' requires graph storage, got '{storage}'"
    return True, ""


def is_config_valid(cfg: ArchConfig) -> Tuple[bool, str]:
    return check_compatibility(cfg.storage, cfg.retrieval, cfg.management)


# ======================================================================
# Experiment result
# ======================================================================

@dataclass
class ExperimentResult:
    config: ArchConfig
    task_score: float = 0.0
    total_tokens: int = 0
    avg_latency: float = 0.0
    memory_size: int = 0
    inserted_units: int = 0
    dedup_count: int = 0
    retrieved_count: int = 0
    graph_nodes: Optional[int] = None
    graph_edges: Optional[int] = None
    run_dir: str = ""
    status: str = "pending"
    error: Optional[str] = None
    num_tasks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "config_id": self.config.config_id,
            **self.config.to_dict(),
            "task_score": self.task_score,
            "total_tokens": self.total_tokens,
            "avg_latency": self.avg_latency,
            "memory_size": self.memory_size,
            "inserted_units": self.inserted_units,
            "dedup_count": self.dedup_count,
            "retrieved_count": self.retrieved_count,
            "graph_nodes": self.graph_nodes,
            "graph_edges": self.graph_edges,
            "run_dir": self.run_dir,
            "status": self.status,
            "error": self.error,
            "num_tasks": self.num_tasks,
        }
        return d

    @staticmethod
    def from_dict(d: Dict) -> "ExperimentResult":
        cfg = ArchConfig.from_dict(d)
        return ExperimentResult(
            config=cfg,
            task_score=d.get("task_score", 0.0),
            total_tokens=d.get("total_tokens", 0),
            avg_latency=d.get("avg_latency", 0.0),
            memory_size=d.get("memory_size", 0),
            inserted_units=d.get("inserted_units", 0),
            dedup_count=d.get("dedup_count", 0),
            retrieved_count=d.get("retrieved_count", 0),
            graph_nodes=d.get("graph_nodes"),
            graph_edges=d.get("graph_edges"),
            run_dir=d.get("run_dir", ""),
            status=d.get("status", "pending"),
            error=d.get("error"),
            num_tasks=d.get("num_tasks", 0),
        )


# ======================================================================
# Scoring
# ======================================================================

def compute_composite_scores(
    results: List[ExperimentResult],
    weights: Optional[Dict[str, float]] = None,
) -> List[float]:
    """Compute composite score for each result using min-max normalization."""
    if not results:
        return []

    w = weights or {}
    w_score = w.get("task_score_weight", 0.80)
    w_token = w.get("token_cost_weight", 0.10)
    w_latency = w.get("latency_weight", 0.05)
    w_memory = w.get("memory_size_weight", 0.05)

    scores = [r.task_score for r in results]
    tokens = [r.total_tokens for r in results]
    latencies = [r.avg_latency for r in results]
    mem_sizes = [r.memory_size for r in results]

    def _norm(vals: List[float]) -> List[float]:
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            return [0.5] * len(vals)
        return [(v - lo) / (hi - lo) for v in vals]

    n_scores = _norm(scores)
    n_tokens = _norm(tokens)
    n_latencies = _norm(latencies)
    n_mem = _norm(mem_sizes)

    composite = []
    for i in range(len(results)):
        f = (
            w_score * n_scores[i]
            - w_token * n_tokens[i]
            - w_latency * n_latencies[i]
            - w_memory * n_mem[i]
        )
        composite.append(f)
    return composite


# ======================================================================
# Run registry — enables resume
# ======================================================================

class RunRegistry:
    """Persists experiment results for resume support.

    Uses file locking for safe concurrent access by parallel workers.
    """

    def __init__(self, path: Path):
        self.path = path
        self._data: Dict[str, Dict] = {}
        self._reload()

    def _reload(self) -> None:
        if self.path.exists():
            try:
                with self.path.open("r") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
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

    def put(self, result: ExperimentResult, key: str = "") -> None:
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

    def all_results(self) -> List[ExperimentResult]:
        self._reload()
        return [ExperimentResult.from_dict(v) for v in self._data.values()]


# ======================================================================
# Experiment Runner
# ======================================================================

class ExperimentRunner:
    """Runs a single architecture configuration as a subprocess."""

    def __init__(
        self,
        search_config: Dict,
        repo_root: Path,
        output_root: Path,
        registry: RunRegistry,
        dry_run: bool = False,
    ):
        self.search_config = search_config
        self.repo_root = repo_root
        self.output_root = output_root
        self.registry = registry
        self.dry_run = dry_run

        self._runner_cfg = search_config.get("runner", {})
        self._dataset_cfg = search_config.get("dataset", {})

    def run(
        self,
        arch_config: ArchConfig,
        run_label: str,
        task_indices: Optional[str] = None,
        id_prefix: str = "",
    ) -> ExperimentResult:
        """Run a single experiment. Returns cached result if available."""
        cache_id = id_prefix + arch_config.config_id

        # Check registry for resume
        if self.registry.has(cache_id):
            cached = self.registry.get(cache_id)
            logger.info(
                f"  [cached] {arch_config.short_name} → score={cached.task_score:.3f}"
            )
            return cached

        run_dir = self.output_root / run_label / arch_config.short_name
        run_dir.mkdir(parents=True, exist_ok=True)
        tasks_dir = run_dir / "tasks"
        tasks_dir.mkdir(exist_ok=True)

        if self.dry_run:
            result = ExperimentResult(
                config=arch_config,
                run_dir=str(run_dir),
                status="dry_run",
                task_score=random.uniform(0.3, 0.9),
                total_tokens=random.randint(50000, 500000),
                avg_latency=random.uniform(30, 300),
                memory_size=random.randint(5, 100),
            )
            self.registry.put(result, key=cache_id)
            logger.info(
                f"  [dry-run] {arch_config.short_name} → simulated score={result.task_score:.3f}"
            )
            return result

        # Build environment variables
        env = dict(os.environ)
        env.update(self._build_env(arch_config, run_dir))

        # Build command
        indices = task_indices or self._dataset_cfg.get("proxy_indices", "[level1]1-10")
        cmd = self._build_command(run_dir, tasks_dir, indices)

        # Write manifest
        manifest = {
            "run_name": arch_config.short_name,
            "config": arch_config.to_dict(),
            "command": cmd,
            "task_indices": indices,
            "run_dir": str(run_dir),
        }
        with (run_dir / "run_manifest.json").open("w") as f:
            json.dump(manifest, f, indent=2)

        # Execute
        logger.info(f"  [running] {arch_config.short_name} ...")
        log_path = run_dir / "run.log"
        try:
            with log_path.open("w") as log_file:
                proc = subprocess.run(
                    cmd,
                    env=env,
                    cwd=str(self.repo_root),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=7200,  # 2h max per experiment
                )
            if proc.returncode != 0:
                logger.warning(
                    f"  [failed] {arch_config.short_name} exit code {proc.returncode}"
                )
        except subprocess.TimeoutExpired:
            logger.warning(f"  [timeout] {arch_config.short_name}")
        except Exception as e:
            logger.error(f"  [error] {arch_config.short_name}: {e}")

        # Parse results
        result = self._parse_results(arch_config, run_dir, tasks_dir)
        self.registry.put(result, key=cache_id)
        logger.info(
            f"  [done] {arch_config.short_name} → score={result.task_score:.3f}, "
            f"tokens={result.total_tokens}, latency={result.avg_latency:.1f}s"
        )
        return result

    def _build_env(self, cfg: ArchConfig, run_dir: Path) -> Dict[str, str]:
        prompts = EXTRACTION_PRESETS.get(cfg.extraction, ["tip", "insight"])
        env = {
            "MODULAR_STORAGE_DIR": str((run_dir / "storage").resolve()),
            "MODULAR_STORAGE_TYPE": cfg.storage,
            "MODULAR_STORAGE_CONFIG": "{}",
            "MODULAR_RETRIEVER_TYPE": cfg.retrieval,
            "MODULAR_RETRIEVER_CONFIG": "{}",
            "MODULAR_ENABLED_PROMPTS": ",".join(prompts),
            "MODULAR_PROMPT_DIR": ".",
            "MODULAR_TOP_K": "5",
        }
        if cfg.management == "none":
            env["MODULAR_MANAGEMENT_ENABLED"] = "false"
        else:
            env["MODULAR_MANAGEMENT_ENABLED"] = "true"
            env["MODULAR_MANAGEMENT_PRESET"] = cfg.management

        # Runner model config
        if self._runner_cfg.get("model"):
            env["DEFAULT_MODEL"] = self._runner_cfg["model"]
        # API base override (e.g., for gpt-5 via codex proxy)
        api_base = self._runner_cfg.get("api_base")
        if api_base:
            env["OPENAI_API_BASE"] = api_base
            env["JUDGE_API_BASE"] = api_base
        api_key = self._runner_cfg.get("api_key")
        if api_key:
            env["OPENAI_API_KEY"] = api_key
            env["JUDGE_API_KEY"] = api_key
        if self._runner_cfg.get("force_stream", False):
            env["FORCE_STREAM"] = "true"
        # GPU device assignment for embedding models
        cuda_devices = self._runner_cfg.get("cuda_visible_devices")
        if cuda_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)
        return env

    def _build_command(
        self, run_dir: Path, tasks_dir: Path, task_indices: str,
    ) -> List[str]:
        dataset_name = self._dataset_cfg.get("name", "gaia")
        script_map = {
            "gaia": "run_flash_searcher_mm_gaia.py",
            "webwalkerqa": "run_flash_searcher_webwalkerqa.py",
            "xbench": "run_flash_searcher_mm_xbench.py",
        }
        script = str(self.repo_root / script_map.get(dataset_name, script_map["gaia"]))
        infile = self._dataset_cfg.get(
            "infile", "./data/gaia/validation/metadata.jsonl"
        )
        if not Path(infile).is_absolute():
            infile = str(self.repo_root / infile)

        cmd = [
            "python", script,
            "--infile", infile,
            "--outfile", str(run_dir / "results.jsonl"),
            "--memory_provider", "modular",
            "--shared_memory_provider",
            "--enable_memory_evolution",
            "--task_indices", task_indices,
            "--max_steps", str(self._runner_cfg.get("max_steps", 40)),
            "--token_budget", str(self._runner_cfg.get("token_budget", 8192)),
            "--concurrency", str(self._runner_cfg.get("concurrency", 1)),
            "--seed", str(self._runner_cfg.get("seed", 42)),
            "--summary_interval", str(self._runner_cfg.get("summary_interval", 8)),
            "--prompts_type", str(self._runner_cfg.get("prompts_type", "default")),
            "--direct_output_dir", str(tasks_dir),
        ]

        model = self._runner_cfg.get("model")
        if model:
            cmd.extend(["--model", model])
        judge = self._runner_cfg.get("judge_model")
        if judge:
            cmd.extend(["--judge_model", judge])

        return cmd

    def _parse_results(
        self, cfg: ArchConfig, run_dir: Path, tasks_dir: Path,
    ) -> ExperimentResult:
        """Parse per-task JSONs to extract aggregate metrics."""
        task_files = sorted(tasks_dir.glob("*.json"))
        if not task_files:
            return ExperimentResult(
                config=cfg, run_dir=str(run_dir), status="no_results",
                error="No task result files found",
            )

        task_scores = []
        total_tokens = 0
        latencies = []
        memory_size = 0
        inserted = 0
        deduped = 0
        retrieved = 0
        graph_n = None
        graph_e = None

        for tf in task_files:
            try:
                with tf.open("r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            # Task score
            score = data.get("task_score")
            if score is None:
                judgement = str(data.get("judgement", "")).strip().lower()
                score = 1.0 if judgement == "correct" else 0.0
            task_scores.append(float(score))

            # Metrics
            metrics = data.get("metrics", {})
            total_tokens += metrics.get("total_tokens", 0)
            latencies.append(metrics.get("elapsed_time", 0.0))

            # Memory metrics
            mm = data.get("memory_metrics", {})
            memory_size = mm.get("num_memory_units", memory_size)
            inserted += mm.get("num_inserted", 0)
            deduped += mm.get("num_deduped", 0)
            retrieved += mm.get("num_retrieved", 0)
            if mm.get("graph_nodes") is not None:
                graph_n = mm["graph_nodes"]
            if mm.get("graph_edges") is not None:
                graph_e = mm["graph_edges"]

        n = len(task_scores)
        return ExperimentResult(
            config=cfg,
            task_score=sum(task_scores) / n if n else 0.0,
            total_tokens=total_tokens,
            avg_latency=sum(latencies) / n if n else 0.0,
            memory_size=memory_size,
            inserted_units=inserted,
            dedup_count=deduped,
            retrieved_count=retrieved,
            graph_nodes=graph_n,
            graph_edges=graph_e,
            run_dir=str(run_dir),
            status="completed",
            num_tasks=n,
        )

    def run_baseline(self, task_indices: Optional[str] = None) -> ExperimentResult:
        """Run no-memory baseline for comparison."""
        cfg = ArchConfig(
            extraction="no_memory", storage="none",
            retrieval="none", management="none",
        )
        cache_id = "baseline_" + cfg.config_id

        if self.registry.has(cache_id):
            cached = self.registry.get(cache_id)
            logger.info(f"  [cached] no_memory_baseline → score={cached.task_score:.3f}")
            return cached

        run_dir = self.output_root / "baseline" / "no_memory"
        run_dir.mkdir(parents=True, exist_ok=True)
        tasks_dir = run_dir / "tasks"
        tasks_dir.mkdir(exist_ok=True)

        if self.dry_run:
            result = ExperimentResult(
                config=cfg, run_dir=str(run_dir), status="dry_run",
                task_score=random.uniform(0.4, 0.8),
                total_tokens=random.randint(50000, 200000),
                avg_latency=random.uniform(100, 300),
                memory_size=0,
            )
            self.registry.put(result, key=cache_id)
            logger.info(f"  [dry-run] no_memory_baseline → simulated score={result.task_score:.3f}")
            return result

        # Build command WITHOUT --memory_provider
        indices = task_indices or self._dataset_cfg.get("proxy_indices", "[level1]1-10")
        dataset_name = self._dataset_cfg.get("name", "gaia")
        script_map = {
            "gaia": "run_flash_searcher_mm_gaia.py",
            "webwalkerqa": "run_flash_searcher_webwalkerqa.py",
            "xbench": "run_flash_searcher_mm_xbench.py",
        }
        script = str(self.repo_root / script_map.get(dataset_name, script_map["gaia"]))
        infile = self._dataset_cfg.get("infile", "./data/gaia/validation/metadata.jsonl")
        if not Path(infile).is_absolute():
            infile = str(self.repo_root / infile)

        cmd = [
            "python", script,
            "--infile", infile,
            "--outfile", str(run_dir / "results.jsonl"),
            "--task_indices", indices,
            "--max_steps", str(self._runner_cfg.get("max_steps", 40)),
            "--token_budget", str(self._runner_cfg.get("token_budget", 8192)),
            "--concurrency", str(self._runner_cfg.get("concurrency", 1)),
            "--seed", str(self._runner_cfg.get("seed", 42)),
            "--summary_interval", str(self._runner_cfg.get("summary_interval", 8)),
            "--prompts_type", str(self._runner_cfg.get("prompts_type", "default")),
            "--direct_output_dir", str(tasks_dir),
        ]
        model = self._runner_cfg.get("model")
        if model:
            cmd.extend(["--model", model])
        judge = self._runner_cfg.get("judge_model")
        if judge:
            cmd.extend(["--judge_model", judge])

        env = dict(os.environ)
        if model:
            env["DEFAULT_MODEL"] = model
        api_base = self._runner_cfg.get("api_base")
        if api_base:
            env["OPENAI_API_BASE"] = api_base
            env["JUDGE_API_BASE"] = api_base
        api_key = self._runner_cfg.get("api_key")
        if api_key:
            env["OPENAI_API_KEY"] = api_key
            env["JUDGE_API_KEY"] = api_key
        if self._runner_cfg.get("force_stream", False):
            env["FORCE_STREAM"] = "true"
        cuda_devices = self._runner_cfg.get("cuda_visible_devices")
        if cuda_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)

        logger.info("  [running] no_memory_baseline ...")
        log_path = run_dir / "run.log"
        try:
            with log_path.open("w") as log_file:
                proc = subprocess.run(
                    cmd, env=env, cwd=str(self.repo_root),
                    stdout=log_file, stderr=subprocess.STDOUT, timeout=7200,
                )
            if proc.returncode != 0:
                logger.warning(f"  [failed] no_memory_baseline exit code {proc.returncode}")
        except subprocess.TimeoutExpired:
            logger.warning("  [timeout] no_memory_baseline")
        except Exception as e:
            logger.error(f"  [error] no_memory_baseline: {e}")

        result = self._parse_results(cfg, run_dir, tasks_dir)
        self.registry.put(result, key=cache_id)
        logger.info(
            f"  [done] no_memory_baseline → score={result.task_score:.3f}, "
            f"tokens={result.total_tokens}"
        )
        return result


# ======================================================================
# CSV helpers
# ======================================================================

RESULT_COLUMNS = [
    "config_id", "extraction", "storage", "retrieval", "management",
    "task_score", "total_tokens", "avg_latency", "memory_size",
    "inserted_units", "dedup_count", "retrieved_count",
    "graph_nodes", "graph_edges", "status", "num_tasks",
]


def _write_results_csv(path: Path, results: List[ExperimentResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for r in results:
            row = r.to_dict()
            writer.writerow({k: row.get(k) for k in RESULT_COLUMNS})


def _write_ranked_csv(
    path: Path,
    results: List[ExperimentResult],
    composite_scores: List[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked_cols = ["rank", "composite_score"] + RESULT_COLUMNS
    pairs = sorted(zip(composite_scores, results), key=lambda x: -x[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ranked_cols)
        writer.writeheader()
        for rank, (score, r) in enumerate(pairs, 1):
            row = r.to_dict()
            row["rank"] = rank
            row["composite_score"] = f"{score:.4f}"
            writer.writerow({k: row.get(k) for k in ranked_cols})


# ======================================================================
# Checkpoint analysis — early-stop / sanity checks
# ======================================================================

def _analyze_screening(
    module_results: Dict[str, List[ExperimentResult]],
    baseline_result: Optional[ExperimentResult],
    min_spread: float,
    output_dir: Path,
) -> Dict[str, Any]:
    """Analyze Phase A results. Warn if configs are too similar or worse than baseline."""
    report: Dict[str, Any] = {"modules": {}, "warnings": [], "recommendation": "continue"}

    all_low = True
    for module_name, results in module_results.items():
        completed = [r for r in results if r.status in ("completed", "dry_run")]
        if len(completed) < 2:
            report["modules"][module_name] = {"spread": 0.0, "num_candidates": len(completed)}
            continue

        scores = [r.task_score for r in completed]
        spread = max(scores) - min(scores)
        best = max(completed, key=lambda r: r.task_score)
        worst = min(completed, key=lambda r: r.task_score)

        report["modules"][module_name] = {
            "spread": round(spread, 4),
            "best": {"name": getattr(best.config, module_name), "score": round(best.task_score, 4)},
            "worst": {"name": getattr(worst.config, module_name), "score": round(worst.task_score, 4)},
            "num_candidates": len(completed),
        }

        if spread >= min_spread:
            all_low = False

    # Compare against baseline
    if baseline_result and baseline_result.status in ("completed", "dry_run"):
        report["baseline_score"] = round(baseline_result.task_score, 4)
        all_scores = []
        for results in module_results.values():
            for r in results:
                if r.status in ("completed", "dry_run"):
                    all_scores.append(r.task_score)
        if all_scores and max(all_scores) <= baseline_result.task_score:
            report["warnings"].append(
                f"WARNING: No memory config beats the no-memory baseline "
                f"(baseline={baseline_result.task_score:.3f}, best={max(all_scores):.3f}). "
                f"Memory system may not help on this proxy set."
            )
            report["recommendation"] = "warn_no_benefit"

    if all_low:
        report["warnings"].append(
            f"WARNING: All modules show score spread < {min_spread:.0%}. "
            f"Configs may be too similar to differentiate. "
            f"Consider: (1) using more proxy tasks, (2) aborting early."
        )
        if report["recommendation"] == "continue":
            report["recommendation"] = "warn_low_spread"

    # Save report
    report_path = output_dir / "checkpoint_a_analysis.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def _analyze_exhaustive(
    results: List[ExperimentResult],
    scoring_weights: Dict,
    baseline_result: Optional[ExperimentResult],
    top_n_spread_check: int,
    min_top_spread: float,
    output_dir: Path,
) -> Dict[str, Any]:
    """Analyze Phase B results. Warn if top configs are too close."""
    report: Dict[str, Any] = {"warnings": [], "recommendation": "continue"}

    completed = [r for r in results if r.status in ("completed", "dry_run")]
    if len(completed) < 2:
        report["num_completed"] = len(completed)
        return report

    scores = compute_composite_scores(completed, scoring_weights)
    ranked = sorted(zip(scores, completed), key=lambda x: -x[0])

    # Check top-N spread
    top_n = min(top_n_spread_check, len(ranked))
    top_composites = [s for s, _ in ranked[:top_n]]
    top_spread = max(top_composites) - min(top_composites)
    top_score_spread = max(r.task_score for _, r in ranked[:top_n]) - min(r.task_score for _, r in ranked[:top_n])

    report["top_n_composite_spread"] = round(top_spread, 4)
    report["top_n_task_score_spread"] = round(top_score_spread, 4)
    report["top_configs"] = [
        {"name": r.config.short_name, "score": round(r.task_score, 4), "composite": round(s, 4)}
        for s, r in ranked[:top_n]
    ]

    if top_score_spread < min_top_spread:
        report["warnings"].append(
            f"WARNING: Top-{top_n} configs differ by only {top_score_spread:.1%} in task score. "
            f"Validation (Phase D) may not reveal meaningful differences. "
            f"Consider running Phase D with top-1 only to save budget."
        )
        report["recommendation"] = "reduce_phase_d"

    # Compare best vs baseline
    if baseline_result and baseline_result.status in ("completed", "dry_run"):
        best_score = ranked[0][1].task_score
        report["baseline_score"] = round(baseline_result.task_score, 4)
        report["best_vs_baseline"] = round(best_score - baseline_result.task_score, 4)
        if best_score <= baseline_result.task_score:
            report["warnings"].append(
                f"WARNING: Best config ({best_score:.3f}) does not beat baseline "
                f"({baseline_result.task_score:.3f}). Consider aborting."
            )
            report["recommendation"] = "abort"

    report_path = output_dir / "checkpoint_b_analysis.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


# ======================================================================
# Phase A: Module Screening
# ======================================================================

def phase_a(
    runner: ExperimentRunner,
    search_config: Dict,
    output_dir: Path,
    baseline_result: Optional[ExperimentResult] = None,
    screen_module: Optional[str] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, List[ExperimentResult]]]:
    """One-factor-at-a-time screening to narrow search space.

    If screen_module is set, only screen that one module (for parallel workers).
    """
    logger.info("=" * 60)
    logger.info("PHASE A: Module Screening")
    logger.info("=" * 60)

    anchor = search_config.get("anchor", DEFAULT_ANCHOR)
    top_k_cfg = search_config.get("phase_a", {}).get("top_k", {})
    scoring_weights = search_config.get("scoring", {})

    checkpoint_path = output_dir / "checkpoint_a.json"
    if checkpoint_path.exists():
        logger.info("Phase A checkpoint found, loading...")
        with checkpoint_path.open("r") as f:
            checkpoint = json.load(f)
        # Reconstruct module_results from checkpoint if available
        saved_mod = checkpoint.get("module_results", {})
        module_results_loaded = {
            mod: [ExperimentResult.from_dict(d) for d in items]
            for mod, items in saved_mod.items()
        }
        return checkpoint["retained"], module_results_loaded

    all_results: List[ExperimentResult] = []
    module_results: Dict[str, List[ExperimentResult]] = {
        "extraction": [], "storage": [], "retrieval": [], "management": [],
    }

    modules_to_screen = [screen_module] if screen_module else ["extraction", "storage", "retrieval", "management"]

    # --- Sub-experiment: vary extraction ---
    if "extraction" in modules_to_screen:
        logger.info("\n--- Screening: Extraction ---")
        for ext_name in EXTRACTION_PRESETS:
            cfg = ArchConfig(
                extraction=ext_name,
                storage=anchor["storage"],
                retrieval=anchor["retrieval"],
                management=anchor["management"],
            )
            valid, reason = is_config_valid(cfg)
            if not valid:
                logger.info(f"  [skip] {cfg.short_name}: {reason}")
                continue
            result = runner.run(cfg, "phase_a/extraction")
            module_results["extraction"].append(result)
            all_results.append(result)

    # --- Sub-experiment: vary storage ---
    if "storage" in modules_to_screen:
        logger.info("\n--- Screening: Storage ---")
        for st in STORAGE_CANDIDATES:
            cfg = ArchConfig(
                extraction=anchor["extraction"],
                storage=st,
                retrieval=anchor["retrieval"],
                management=anchor["management"],
            )
            valid, reason = is_config_valid(cfg)
            if not valid:
                logger.info(f"  [skip] {cfg.short_name}: {reason}")
                continue
            result = runner.run(cfg, "phase_a/storage")
            module_results["storage"].append(result)
            all_results.append(result)

    # --- Sub-experiment: vary retrieval ---
    if "retrieval" in modules_to_screen:
        logger.info("\n--- Screening: Retrieval ---")
        for ret in RETRIEVAL_CANDIDATES:
            cfg = ArchConfig(
                extraction=anchor["extraction"],
                storage=anchor["storage"],
                retrieval=ret,
                management=anchor["management"],
            )
            valid, reason = is_config_valid(cfg)
            if not valid:
                logger.info(f"  [skip] {cfg.short_name}: {reason}")
                continue
            result = runner.run(cfg, "phase_a/retrieval")
            module_results["retrieval"].append(result)
            all_results.append(result)

    # --- Sub-experiment: vary management ---
    if "management" in modules_to_screen:
        logger.info("\n--- Screening: Management ---")
        for mgmt in MANAGEMENT_CANDIDATES:
            cfg = ArchConfig(
                extraction=anchor["extraction"],
                storage=anchor["storage"],
                retrieval=anchor["retrieval"],
                management=mgmt,
            )
            valid, reason = is_config_valid(cfg)
            if not valid:
                logger.info(f"  [skip] {cfg.short_name}: {reason}")
                continue
            result = runner.run(cfg, "phase_a/management")
            module_results["management"].append(result)
            all_results.append(result)

    # --- Select top-k per module ---
    retained: Dict[str, List[str]] = {}
    for module_name, results in module_results.items():
        k = top_k_cfg.get(module_name, 2)
        if not results:
            retained[module_name] = [anchor[module_name]]
            continue

        completed = [r for r in results if r.status in ("completed", "dry_run")]
        if not completed:
            retained[module_name] = [anchor[module_name]]
            continue

        scores = compute_composite_scores(completed, scoring_weights)
        ranked = sorted(zip(scores, completed), key=lambda x: -x[0])
        field_name = module_name  # extraction, storage, retrieval, management
        top = [getattr(r.config, field_name) for _, r in ranked[:k]]
        # Ensure anchor is always included
        anchor_val = anchor[module_name]
        if anchor_val not in top:
            top.append(anchor_val)
        retained[module_name] = top

        logger.info(f"\n  Top-{k} {module_name}: {top}")
        for s, r in ranked[:k]:
            logger.info(
                f"    {getattr(r.config, field_name)}: "
                f"score={r.task_score:.3f}, composite={s:.4f}"
            )

    # Write outputs
    _write_results_csv(output_dir / "screening_results.csv", all_results)
    with (output_dir / "retained_candidates.json").open("w") as f:
        json.dump(retained, f, indent=2)

    # Save checkpoint (include module_results for analysis)
    with checkpoint_path.open("w") as f:
        json.dump({
            "retained": retained,
            "module_results": {
                mod: [r.to_dict() for r in results]
                for mod, results in module_results.items()
            },
        }, f, indent=2)

    # --- Checkpoint analysis ---
    min_spread = search_config.get("phase_a", {}).get("min_spread_threshold", 0.08)
    screening_report = _analyze_screening(
        module_results, baseline_result, min_spread, output_dir,
    )
    for w in screening_report.get("warnings", []):
        logger.warning(w)

    if screening_report["recommendation"] != "continue":
        logger.warning(
            f"\n  CHECKPOINT A recommendation: {screening_report['recommendation']}"
        )
        logger.warning("  Review checkpoint_a_analysis.json for details.")

    logger.info(f"\nPhase A complete. Retained candidates: {retained}")
    return retained, module_results


# ======================================================================
# Phase B: Reduced Space Exhaustive Search
# ======================================================================

def phase_b(
    runner: ExperimentRunner,
    retained: Dict[str, List[str]],
    search_config: Dict,
    output_dir: Path,
    baseline_result: Optional[ExperimentResult] = None,
) -> List[ExperimentResult]:
    """Exhaustive search over retained candidate combinations."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE B: Reduced Space Search")
    logger.info("=" * 60)

    scoring_weights = search_config.get("scoring", {})
    top_elites = search_config.get("phase_b", {}).get("top_elites", 6)

    checkpoint_path = output_dir / "checkpoint_b.json"
    if checkpoint_path.exists():
        logger.info("Phase B checkpoint found, loading...")
        with checkpoint_path.open("r") as f:
            checkpoint = json.load(f)
        return [ExperimentResult.from_dict(d) for d in checkpoint["elites"]]

    # Generate all valid combinations
    combos = list(itertools.product(
        retained.get("extraction", ["mixed_all"]),
        retained.get("storage", ["hybrid"]),
        retained.get("retrieval", ["hybrid"]),
        retained.get("management", ["json_full"]),
    ))

    valid_configs: List[ArchConfig] = []
    skipped: List[Dict] = []
    for ext, st, ret, mgmt in combos:
        cfg = ArchConfig(extraction=ext, storage=st, retrieval=ret, management=mgmt)
        valid, reason = is_config_valid(cfg)
        if valid:
            valid_configs.append(cfg)
        else:
            skipped.append({"config": cfg.short_name, "reason": reason})

    # Cap combinations if max_combinations is set
    max_combos = search_config.get("phase_b", {}).get("max_combinations", 0)
    if max_combos > 0 and len(valid_configs) > max_combos:
        logger.info(
            f"\n  Capping {len(valid_configs)} valid combos to max_combinations={max_combos}"
        )
        random.shuffle(valid_configs)
        valid_configs = valid_configs[:max_combos]

    logger.info(
        f"\nTotal combinations: {len(combos)}, "
        f"valid: {len(valid_configs)}, skipped: {len(skipped)}"
    )
    for s in skipped:
        logger.info(f"  [skip] {s['config']}: {s['reason']}")

    # Run all valid combinations
    all_results: List[ExperimentResult] = []
    for i, cfg in enumerate(valid_configs, 1):
        logger.info(f"\n[{i}/{len(valid_configs)}] {cfg.short_name}")
        result = runner.run(cfg, "phase_b")
        all_results.append(result)

    # Score and rank
    completed = [r for r in all_results if r.status in ("completed", "dry_run")]
    if not completed:
        logger.warning("No completed experiments in Phase B")
        return []

    scores = compute_composite_scores(completed, scoring_weights)
    ranked = sorted(zip(scores, completed), key=lambda x: -x[0])

    # Write outputs
    _write_results_csv(output_dir / "reduced_space_all_results.csv", all_results)
    _write_ranked_csv(output_dir / "reduced_space_ranked_results.csv", completed, scores)

    elites = [r for _, r in ranked[:top_elites]]
    with (output_dir / "top_elites.json").open("w") as f:
        json.dump([e.to_dict() for e in elites], f, indent=2)

    logger.info(f"\nPhase B complete. Top {top_elites} elites:")
    for rank, (s, r) in enumerate(ranked[:top_elites], 1):
        logger.info(
            f"  #{rank} {r.config.short_name}: "
            f"score={r.task_score:.3f}, composite={s:.4f}"
        )

    # Save checkpoint
    with checkpoint_path.open("w") as f:
        json.dump({"elites": [e.to_dict() for e in elites]}, f, indent=2)

    # --- Checkpoint analysis ---
    b_cfg = search_config.get("phase_b", {})
    exhaustive_report = _analyze_exhaustive(
        completed, scoring_weights, baseline_result,
        top_n_spread_check=min(5, len(ranked)),
        min_top_spread=b_cfg.get("min_top_spread", 0.05),
        output_dir=output_dir,
    )
    for w in exhaustive_report.get("warnings", []):
        logger.warning(w)

    if exhaustive_report.get("recommendation") not in ("continue", None):
        logger.warning(
            f"\n  CHECKPOINT B recommendation: {exhaustive_report['recommendation']}"
        )
        logger.warning("  Review checkpoint_b_analysis.json for details.")

    return elites


# ======================================================================
# Phase C: Evolutionary Refinement
# ======================================================================

def _mutate(cfg: ArchConfig, retained: Dict[str, List[str]]) -> ArchConfig:
    """Randomly replace one module with another retained candidate."""
    modules = ["extraction", "storage", "retrieval", "management"]
    target = random.choice(modules)
    candidates = retained.get(target, [getattr(cfg, target)])
    # Pick a different value
    current = getattr(cfg, target)
    options = [c for c in candidates if c != current]
    if not options:
        # Fall back to full candidate list
        full_list = {
            "extraction": list(EXTRACTION_PRESETS.keys()),
            "storage": STORAGE_CANDIDATES,
            "retrieval": RETRIEVAL_CANDIDATES,
            "management": MANAGEMENT_CANDIDATES,
        }[target]
        options = [c for c in full_list if c != current]
    if not options:
        return deepcopy(cfg)

    new_val = random.choice(options)
    new_cfg = deepcopy(cfg)
    setattr(new_cfg, target, new_val)
    return new_cfg


def _crossover(cfg_a: ArchConfig, cfg_b: ArchConfig) -> ArchConfig:
    """Create a child by randomly picking each module from one parent."""
    return ArchConfig(
        extraction=random.choice([cfg_a.extraction, cfg_b.extraction]),
        storage=random.choice([cfg_a.storage, cfg_b.storage]),
        retrieval=random.choice([cfg_a.retrieval, cfg_b.retrieval]),
        management=random.choice([cfg_a.management, cfg_b.management]),
    )


def _local_refine(
    cfg: ArchConfig,
    enable_op_refinement: bool,
    op_whitelist: List[str],
) -> ArchConfig:
    """
    Local refinement around management preset.

    When enable_op_refinement is False (default), this just tries swapping
    between adjacent presets (e.g., json_basic ↔ json_full).
    """
    new_cfg = deepcopy(cfg)

    # Preset adjacency: lightweight → json_basic → json_full, graph_full
    preset_neighbors = {
        "none": ["lightweight"],
        "lightweight": ["none", "json_basic"],
        "json_basic": ["lightweight", "json_full"],
        "json_full": ["json_basic"],
        "graph_full": ["json_full"],
    }
    neighbors = preset_neighbors.get(cfg.management, [])
    if neighbors:
        new_cfg.management = random.choice(neighbors)

    return new_cfg


def phase_c(
    runner: ExperimentRunner,
    elites: List[ExperimentResult],
    retained: Dict[str, List[str]],
    search_config: Dict,
    output_dir: Path,
) -> List[ExperimentResult]:
    """Evolutionary local refinement around top elites."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE C: Evolutionary Refinement")
    logger.info("=" * 60)

    c_cfg = search_config.get("phase_c", {})
    pop_size = c_cfg.get("population_size", 8)
    elite_count = c_cfg.get("elite_count", 4)
    generations = c_cfg.get("generations", 3)
    mutation_rate = c_cfg.get("mutation_rate", 0.5)
    crossover_rate = c_cfg.get("crossover_rate", 0.3)
    # refine_rate = 1 - mutation_rate - crossover_rate
    enable_op_refine = c_cfg.get("enable_op_refinement", False)
    op_whitelist = c_cfg.get("op_refinement_whitelist", [])
    scoring_weights = search_config.get("scoring", {})

    checkpoint_path = output_dir / "checkpoint_c.json"
    if checkpoint_path.exists():
        logger.info("Phase C checkpoint found, loading...")
        with checkpoint_path.open("r") as f:
            checkpoint = json.load(f)
        return [ExperimentResult.from_dict(d) for d in checkpoint["final_top"]]

    # Initialize population from elites
    population: List[ExperimentResult] = list(elites)
    trajectory_rows: List[Dict] = []
    best_per_gen: List[Dict] = []
    seen_ids: Set[str] = {r.config.config_id for r in population}

    for gen in range(1, generations + 1):
        logger.info(f"\n--- Generation {gen}/{generations} ---")

        # Generate candidates
        children: List[Tuple[ArchConfig, str, str]] = []  # (config, op_type, parent_info)

        attempts = 0
        max_attempts = pop_size * 10
        while len(children) < pop_size and attempts < max_attempts:
            attempts += 1
            rand_val = random.random()

            if rand_val < mutation_rate:
                parent = random.choice(population)
                child_cfg = _mutate(parent.config, retained)
                op_type = "mutation"
                parent_info = parent.config.short_name
            elif rand_val < mutation_rate + crossover_rate:
                if len(population) < 2:
                    continue
                p1, p2 = random.sample(population, 2)
                child_cfg = _crossover(p1.config, p2.config)
                op_type = "crossover"
                parent_info = f"{p1.config.short_name} × {p2.config.short_name}"
            else:
                parent = random.choice(population)
                child_cfg = _local_refine(parent.config, enable_op_refine, op_whitelist)
                op_type = "local_refine"
                parent_info = parent.config.short_name

            # Skip invalid or duplicate configs
            valid, reason = is_config_valid(child_cfg)
            if not valid:
                continue
            if child_cfg.config_id in seen_ids:
                continue

            seen_ids.add(child_cfg.config_id)
            children.append((child_cfg, op_type, parent_info))

        # Evaluate children
        gen_results: List[ExperimentResult] = []
        for i, (child_cfg, op_type, parent_info) in enumerate(children, 1):
            logger.info(
                f"  [{i}/{len(children)}] {op_type}: {child_cfg.short_name}"
            )
            result = runner.run(child_cfg, f"phase_c/gen{gen}")
            gen_results.append(result)

            trajectory_rows.append({
                "generation": gen,
                "operation": op_type,
                "parent": parent_info,
                "child": child_cfg.short_name,
                "config_id": child_cfg.config_id,
                "task_score": result.task_score,
                "status": result.status,
            })

        # Merge population + children, select top elite_count
        all_gen = population + gen_results
        completed = [r for r in all_gen if r.status in ("completed", "dry_run")]
        if not completed:
            continue

        scores = compute_composite_scores(completed, scoring_weights)
        ranked = sorted(zip(scores, completed), key=lambda x: -x[0])
        population = [r for _, r in ranked[:elite_count]]

        best = ranked[0]
        best_per_gen.append({
            "generation": gen,
            "config": best[1].config.short_name,
            "config_id": best[1].config.config_id,
            "task_score": best[1].task_score,
            "composite_score": best[0],
        })

        logger.info(
            f"\n  Generation {gen} best: {best[1].config.short_name} "
            f"(score={best[1].task_score:.3f}, composite={best[0]:.4f})"
        )
        for rank, (s, r) in enumerate(ranked[:elite_count], 1):
            logger.info(f"    #{rank} {r.config.short_name}: composite={s:.4f}")

    # Write outputs
    traj_path = output_dir / "evolution_trajectory.csv"
    if trajectory_rows:
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        traj_cols = list(trajectory_rows[0].keys())
        with traj_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=traj_cols)
            writer.writeheader()
            writer.writerows(trajectory_rows)

    bpg_path = output_dir / "evolution_best_per_generation.csv"
    if best_per_gen:
        bpg_cols = list(best_per_gen[0].keys())
        with bpg_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=bpg_cols)
            writer.writeheader()
            writer.writerows(best_per_gen)

    final_top = population[:3]
    with (output_dir / "final_top3_configs.json").open("w") as f:
        json.dump([r.to_dict() for r in final_top], f, indent=2)

    # Save checkpoint
    with checkpoint_path.open("w") as f:
        json.dump({"final_top": [r.to_dict() for r in final_top]}, f, indent=2)

    logger.info(f"\nPhase C complete. Final top-3:")
    for i, r in enumerate(final_top, 1):
        logger.info(f"  #{i} {r.config.short_name}: score={r.task_score:.3f}")

    return final_top


# ======================================================================
# Phase D: Final Validation
# ======================================================================

def phase_d(
    runner: ExperimentRunner,
    final_configs: List[ExperimentResult],
    search_config: Dict,
    output_dir: Path,
) -> None:
    """Validate top architectures on held-out data."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE D: Final Validation")
    logger.info("=" * 60)

    d_cfg = search_config.get("phase_d", {})
    top_n = d_cfg.get("top_n", 3)
    datasets = d_cfg.get("datasets", ["gaia"])
    holdout_indices = search_config.get("dataset", {}).get(
        "holdout_indices", "[level1]11-53"
    )

    configs_to_eval = final_configs[:top_n]

    # Per-dataset evaluation
    all_eval_results: Dict[str, List[ExperimentResult]] = {}
    for ds_name in datasets:
        logger.info(f"\n--- Evaluating on {ds_name} ---")

        if ds_name == "locomo":
            logger.info("  [info] LoCoMo dataset not yet integrated, skipping.")
            continue

        ds_results: List[ExperimentResult] = []
        for cfg_result in configs_to_eval:
            cfg = cfg_result.config
            # Use a separate registry for held-out to avoid cache conflicts
            holdout_id_suffix = f"_holdout_{ds_name}"
            holdout_cfg = ArchConfig(
                extraction=cfg.extraction,
                storage=cfg.storage,
                retrieval=cfg.retrieval,
                management=cfg.management,
            )
            # Override config_id to differentiate from proxy runs
            original_id = holdout_cfg.config_id

            logger.info(f"  Running {cfg.short_name} on {ds_name} held-out set...")
            result = runner.run(
                holdout_cfg,
                f"phase_d/{ds_name}",
                task_indices=holdout_indices if ds_name == "gaia" else None,
                id_prefix=f"holdout_{ds_name}_",
            )
            ds_results.append(result)

        all_eval_results[ds_name] = ds_results

    # Write main evaluation table
    main_table_rows = []
    for ds_name, results in all_eval_results.items():
        for r in results:
            row = r.to_dict()
            row["dataset"] = ds_name
            main_table_rows.append(row)

    if main_table_rows:
        main_cols = ["dataset"] + RESULT_COLUMNS
        main_path = output_dir / "final_eval_main_table.csv"
        with main_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=main_cols)
            writer.writeheader()
            for row in main_table_rows:
                writer.writerow({k: row.get(k) for k in main_cols})

    # Write cost table
    cost_cols = [
        "dataset", "config_id", "extraction", "storage", "retrieval", "management",
        "total_tokens", "avg_latency", "memory_size", "inserted_units",
    ]
    cost_path = output_dir / "final_eval_cost_table.csv"
    if main_table_rows:
        with cost_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cost_cols)
            writer.writeheader()
            for row in main_table_rows:
                writer.writerow({k: row.get(k) for k in cost_cols})

    # Transfer matrix (only meaningful with multiple datasets)
    if len(all_eval_results) > 1:
        transfer_rows = []
        ds_names = list(all_eval_results.keys())
        for src_ds in ds_names:
            src_results = all_eval_results[src_ds]
            if not src_results:
                continue
            # Best config from src_ds
            best_src = max(src_results, key=lambda r: r.task_score)
            for tgt_ds in ds_names:
                tgt_results = all_eval_results[tgt_ds]
                # Find same config in tgt_ds
                matching = [
                    r for r in tgt_results
                    if r.config.short_name == best_src.config.short_name
                ]
                tgt_score = matching[0].task_score if matching else None
                transfer_rows.append({
                    "source_dataset": src_ds,
                    "target_dataset": tgt_ds,
                    "config": best_src.config.short_name,
                    "source_score": best_src.task_score,
                    "target_score": tgt_score,
                })

        if transfer_rows:
            transfer_path = output_dir / "transfer_matrix.csv"
            with transfer_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(transfer_rows[0].keys()))
                writer.writeheader()
                writer.writerows(transfer_rows)
    else:
        logger.info(
            "\n  [info] Transfer matrix requires multiple datasets. "
            "Only one dataset evaluated — skipping transfer_matrix.csv."
        )

    logger.info("\nPhase D complete.")
    for ds_name, results in all_eval_results.items():
        logger.info(f"\n  {ds_name} results:")
        for r in results:
            logger.info(f"    {r.config.short_name}: score={r.task_score:.3f}")


# ======================================================================
# Parallel Phase A orchestration
# ======================================================================

def _parallel_phase_a(
    args: argparse.Namespace,
    search_config: Dict,
    output_dir: Path,
    runner: ExperimentRunner,
    baseline_result: Optional[ExperimentResult],
) -> Tuple[Dict[str, List[str]], Dict[str, List[ExperimentResult]]]:
    """Run Phase A screening in parallel: one GPU per module."""
    gpu_ids_str = args.gpu_ids
    if gpu_ids_str:
        gpu_ids = [x.strip() for x in gpu_ids_str.split(",")]
    elif args.gpu:
        gpu_ids = [args.gpu]
    else:
        gpu_ids = ["0", "1", "2", "3"]

    modules = ["extraction", "storage", "retrieval", "management"]
    # Assign GPUs round-robin
    assignments = {mod: gpu_ids[i % len(gpu_ids)] for i, mod in enumerate(modules)}

    logger.info("=" * 60)
    logger.info("PHASE A: Parallel Module Screening")
    logger.info("=" * 60)
    for mod, gpu in assignments.items():
        logger.info(f"  {mod} -> GPU {gpu}")

    # Build base command (reuse current CLI args)
    config_path = args.config
    base_cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--config", config_path,
        "--phases", "A",
    ]
    if args.override_proxy_indices:
        base_cmd.extend(["--override-proxy-indices", args.override_proxy_indices])
    if args.seed is not None:
        base_cmd.extend(["--seed", str(args.seed)])
    if args.dry_run:
        base_cmd.append("--dry-run")
    if args.resume:
        base_cmd.append("--resume")

    # Launch workers
    workers = []
    for mod, gpu in assignments.items():
        cmd = base_cmd + ["--screen-module", mod, "--gpu", gpu]
        log_path = output_dir / f"phase_a_worker_{mod}.log"
        logger.info(f"  Launching: {mod} on GPU {gpu}")
        log_f = log_path.open("w")
        proc = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
        )
        workers.append((mod, proc, log_path, log_f))

    # Wait
    failed = []
    for mod, proc, log_path, log_f in workers:
        proc.wait()
        log_f.close()
        if proc.returncode != 0:
            failed.append(mod)
            logger.error(f"  Worker {mod} failed (exit {proc.returncode}). See {log_path}")
        else:
            logger.info(f"  Worker {mod} completed.")

    if failed:
        logger.warning(f"  Failed modules: {failed}")

    # Merge results from shared registry
    logger.info("  Merging Phase A results from registry...")
    registry = runner.registry

    anchor = search_config.get("anchor", DEFAULT_ANCHOR)
    top_k_cfg = search_config.get("phase_a", {}).get("top_k", {})
    scoring_weights = search_config.get("scoring", {})

    module_results: Dict[str, List[ExperimentResult]] = {}
    all_results: List[ExperimentResult] = []

    # Reconstruct module_results by scanning the registry
    module_configs = {
        "extraction": [
            ArchConfig(extraction=e, storage=anchor["storage"],
                       retrieval=anchor["retrieval"], management=anchor["management"])
            for e in EXTRACTION_PRESETS
        ],
        "storage": [
            ArchConfig(extraction=anchor["extraction"], storage=s,
                       retrieval=anchor["retrieval"], management=anchor["management"])
            for s in STORAGE_CANDIDATES
            if is_config_valid(ArchConfig(anchor["extraction"], s, anchor["retrieval"], anchor["management"]))[0]
        ],
        "retrieval": [
            ArchConfig(extraction=anchor["extraction"], storage=anchor["storage"],
                       retrieval=r, management=anchor["management"])
            for r in RETRIEVAL_CANDIDATES
            if is_config_valid(ArchConfig(anchor["extraction"], anchor["storage"], r, anchor["management"]))[0]
        ],
        "management": [
            ArchConfig(extraction=anchor["extraction"], storage=anchor["storage"],
                       retrieval=anchor["retrieval"], management=m)
            for m in MANAGEMENT_CANDIDATES
            if is_config_valid(ArchConfig(anchor["extraction"], anchor["storage"], anchor["retrieval"], m))[0]
        ],
    }

    for mod, configs in module_configs.items():
        results = []
        for cfg in configs:
            result = registry.get(cfg.config_id)
            if result is not None:
                results.append(result)
        module_results[mod] = results
        all_results.extend(results)

    # Select top-k per module (same logic as phase_a)
    retained: Dict[str, List[str]] = {}
    for module_name, results in module_results.items():
        k = top_k_cfg.get(module_name, 2)
        completed = [r for r in results if r.status in ("completed", "dry_run")]
        if not completed:
            retained[module_name] = [anchor[module_name]]
            continue

        scores = compute_composite_scores(completed, scoring_weights)
        ranked = sorted(zip(scores, completed), key=lambda x: -x[0])
        top = [getattr(r.config, module_name) for _, r in ranked[:k]]
        anchor_val = anchor[module_name]
        if anchor_val not in top:
            top.append(anchor_val)
        retained[module_name] = top
        logger.info(f"  Top-{k} {module_name}: {top}")

    # Write outputs
    _write_results_csv(output_dir / "screening_results.csv", all_results)
    with (output_dir / "retained_candidates.json").open("w") as f:
        json.dump(retained, f, indent=2)

    checkpoint_path = output_dir / "checkpoint_a.json"
    with checkpoint_path.open("w") as f:
        json.dump({
            "retained": retained,
            "module_results": {
                mod: [r.to_dict() for r in results]
                for mod, results in module_results.items()
            },
        }, f, indent=2)

    # Checkpoint analysis
    min_spread = search_config.get("phase_a", {}).get("min_spread_threshold", 0.08)
    screening_report = _analyze_screening(
        module_results, baseline_result, min_spread, output_dir,
    )
    for w in screening_report.get("warnings", []):
        logger.warning(w)

    logger.info(f"\nParallel Phase A complete. Retained candidates: {retained}")
    return retained, module_results


# ======================================================================
# Parallel Phase B
# ======================================================================

def _parallel_phase_b(
    args: argparse.Namespace,
    retained: Dict[str, List[str]],
    search_config: Dict,
    output_dir: Path,
    runner: ExperimentRunner,
    baseline_result: Optional[ExperimentResult],
) -> List[ExperimentResult]:
    """Run Phase B exhaustive search in parallel across multiple GPUs.

    Splits the valid config list into chunks, spawns one worker per GPU,
    then merges results from the shared registry.
    """
    gpu_ids_str = args.gpu_ids
    if gpu_ids_str:
        gpu_ids = [x.strip() for x in gpu_ids_str.split(",")]
    elif args.gpu:
        gpu_ids = [args.gpu]
    else:
        gpu_ids = ["0", "1", "2", "3"]

    scoring_weights = search_config.get("scoring", {})
    top_elites = search_config.get("phase_b", {}).get("top_elites", 6)

    # Check for existing checkpoint
    checkpoint_path = output_dir / "checkpoint_b.json"
    if checkpoint_path.exists():
        logger.info("Phase B checkpoint found, loading...")
        with checkpoint_path.open("r") as f:
            checkpoint = json.load(f)
        return [ExperimentResult.from_dict(d) for d in checkpoint["elites"]]

    # Generate all valid combinations (same logic as phase_b)
    combos = list(itertools.product(
        retained.get("extraction", ["mixed_all"]),
        retained.get("storage", ["hybrid"]),
        retained.get("retrieval", ["hybrid"]),
        retained.get("management", ["json_full"]),
    ))

    valid_configs: List[ArchConfig] = []
    skipped: List[Dict] = []
    for ext, st, ret, mgmt in combos:
        cfg = ArchConfig(extraction=ext, storage=st, retrieval=ret, management=mgmt)
        valid, reason = is_config_valid(cfg)
        if valid:
            valid_configs.append(cfg)
        else:
            skipped.append({"config": cfg.short_name, "reason": reason})

    # Cap combinations if max_combinations is set
    max_combos = search_config.get("phase_b", {}).get("max_combinations", 0)
    if max_combos > 0 and len(valid_configs) > max_combos:
        random.shuffle(valid_configs)
        valid_configs = valid_configs[:max_combos]

    # Filter out already-completed configs
    pending_configs = []
    for cfg in valid_configs:
        if not runner.registry.has(cfg.config_id):
            pending_configs.append(cfg)

    logger.info("=" * 60)
    logger.info("PHASE B: Parallel Reduced Space Search")
    logger.info("=" * 60)
    logger.info(
        f"Total valid: {len(valid_configs)}, already done: "
        f"{len(valid_configs) - len(pending_configs)}, pending: {len(pending_configs)}"
    )
    for s in skipped:
        logger.info(f"  [skip] {s['config']}: {s['reason']}")

    if pending_configs:
        # Split pending configs across GPUs (round-robin)
        n_gpus = len(gpu_ids)
        chunks: Dict[str, List[str]] = {gpu: [] for gpu in gpu_ids}
        for i, cfg in enumerate(pending_configs):
            gpu = gpu_ids[i % n_gpus]
            chunks[gpu].append(cfg.config_id)

        for gpu, ids in chunks.items():
            logger.info(f"  GPU {gpu}: {len(ids)} configs")

        # Build base command
        config_path = args.config
        base_cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--config", config_path,
            "--phases", "B",
            "--resume",
        ]
        if args.override_proxy_indices:
            base_cmd.extend(["--override-proxy-indices", args.override_proxy_indices])
        if args.seed is not None:
            base_cmd.extend(["--seed", str(args.seed)])
        if args.dry_run:
            base_cmd.append("--dry-run")

        # Launch workers
        workers = []
        for gpu, config_ids in chunks.items():
            if not config_ids:
                continue
            chunk_json = json.dumps(config_ids)
            cmd = base_cmd + [
                "--gpu", gpu,
                "--phase-b-chunk", chunk_json,
            ]
            log_path = output_dir / f"phase_b_worker_gpu{gpu}.log"
            logger.info(f"  Launching Phase B worker on GPU {gpu} ({len(config_ids)} configs)")
            log_f = log_path.open("w")
            proc = subprocess.Popen(
                cmd, stdout=log_f, stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).resolve().parent.parent.parent),
            )
            workers.append((gpu, proc, log_path, log_f))

        # Wait for all workers
        failed = []
        for gpu, proc, log_path, log_f in workers:
            proc.wait()
            log_f.close()
            if proc.returncode != 0:
                failed.append(gpu)
                logger.error(f"  Worker GPU {gpu} failed (exit {proc.returncode}). See {log_path}")
            else:
                logger.info(f"  Worker GPU {gpu} completed.")

        if failed:
            logger.warning(f"  Failed GPUs: {failed}")

    # Merge results from shared registry
    logger.info("  Merging Phase B results from registry...")
    all_results: List[ExperimentResult] = []
    for cfg in valid_configs:
        result = runner.registry.get(cfg.config_id)
        if result is not None:
            all_results.append(result)
        else:
            logger.warning(f"  Missing result for {cfg.short_name}")

    # Score and rank
    completed = [r for r in all_results if r.status in ("completed", "dry_run")]
    if not completed:
        logger.warning("No completed experiments in Phase B")
        return []

    scores = compute_composite_scores(completed, scoring_weights)
    ranked = sorted(zip(scores, completed), key=lambda x: -x[0])

    # Write outputs
    _write_results_csv(output_dir / "reduced_space_all_results.csv", all_results)
    _write_ranked_csv(output_dir / "reduced_space_ranked_results.csv", completed, scores)

    elites = [r for _, r in ranked[:top_elites]]
    with (output_dir / "top_elites.json").open("w") as f:
        json.dump([e.to_dict() for e in elites], f, indent=2)

    logger.info(f"\nParallel Phase B complete. Top {top_elites} elites:")
    for rank, (s, r) in enumerate(ranked[:top_elites], 1):
        logger.info(
            f"  #{rank} {r.config.short_name}: "
            f"score={r.task_score:.3f}, composite={s:.4f}"
        )

    # Save checkpoint
    with checkpoint_path.open("w") as f:
        json.dump({"elites": [e.to_dict() for e in elites]}, f, indent=2)

    # Checkpoint analysis
    b_cfg = search_config.get("phase_b", {})
    exhaustive_report = _analyze_exhaustive(
        completed, scoring_weights, baseline_result,
        top_n_spread_check=min(5, len(ranked)),
        min_top_spread=b_cfg.get("min_top_spread", 0.05),
        output_dir=output_dir,
    )
    for w in exhaustive_report.get("warnings", []):
        logger.warning(w)

    return elites


# ======================================================================
# Main
# ======================================================================

def load_search_config(path: Optional[str]) -> Dict:
    """Load search config from JSON file or return defaults."""
    if path and Path(path).exists():
        with Path(path).open("r") as f:
            return json.load(f)

    # Default config
    return {
        "search_name": "arch_search_default",
        "dataset": {
            "name": "gaia",
            "infile": "./data/gaia/validation/metadata.jsonl",
            "proxy_indices": "[level1]1-10",
            "holdout_indices": "[level1]11-53",
        },
        "runner": {"model": "gpt-5", "max_steps": 40, "token_budget": 8192},
        "anchor": DEFAULT_ANCHOR,
        "phase_a": {"top_k": {"extraction": 2, "storage": 3, "retrieval": 3, "management": 2}},
        "phase_b": {"top_elites": 6},
        "phase_c": {
            "population_size": 8, "elite_count": 4, "generations": 3,
            "mutation_rate": 0.5, "crossover_rate": 0.3, "refine_rate": 0.2,
            "enable_op_refinement": False,
        },
        "phase_d": {"top_n": 3, "datasets": ["gaia"]},
        "scoring": {
            "task_score_weight": 0.80, "token_cost_weight": 0.10,
            "latency_weight": 0.05, "memory_size_weight": 0.05,
        },
        "output_root": "./experiments/results/architecture_search",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical Evolutionary Memory Architecture Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to search config JSON (defaults to built-in config)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate configs and simulate scores without running experiments",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoints if available",
    )
    parser.add_argument(
        "--phases", type=str, default="ABCD",
        help="Which phases to run (e.g., 'AB', 'CD'). Default: 'ABCD'",
    )
    parser.add_argument(
        "--override-proxy-indices", type=str, default=None,
        help="Override proxy task indices (e.g., '[level1]1' for smoke test)",
    )
    parser.add_argument(
        "--override-holdout-indices", type=str, default=None,
        help="Override held-out task indices for Phase D",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for evolutionary search (default: from config or 42)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--gpu", type=str, default=None,
        help="GPU device ID(s) for CUDA_VISIBLE_DEVICES (e.g., '0' or '4,5')",
    )
    parser.add_argument(
        "--parallel-phase-a", action="store_true",
        help="Run Phase A screening in parallel across GPUs (one module per GPU)",
    )
    parser.add_argument(
        "--gpu-ids", type=str, default=None,
        help="Comma-separated GPU IDs for parallel Phase A (e.g., '4,5,6,7')",
    )
    parser.add_argument(
        "--screen-module", type=str, default=None,
        choices=["extraction", "storage", "retrieval", "management"],
        help="(Internal) Screen only this module in Phase A (used by parallel workers)",
    )
    parser.add_argument(
        "--parallel-phase-b", action="store_true",
        help="Run Phase B exhaustive search in parallel across GPUs",
    )
    parser.add_argument(
        "--phase-b-chunk", type=str, default=None,
        help="(Internal) JSON list of config_ids this worker should run in Phase B",
    )
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config
    search_config = load_search_config(args.config)

    # Apply overrides
    gpu_for_main = args.gpu
    if gpu_for_main is None and (args.parallel_phase_a or args.parallel_phase_b) and args.gpu_ids:
        # Use first GPU ID for the main process (Phase D)
        gpu_for_main = args.gpu_ids.split(",")[0].strip()
    if gpu_for_main is not None:
        if "runner" not in search_config:
            search_config["runner"] = {}
        search_config["runner"]["cuda_visible_devices"] = gpu_for_main
    if args.override_proxy_indices:
        search_config["dataset"]["proxy_indices"] = args.override_proxy_indices
    if args.override_holdout_indices:
        search_config["dataset"]["holdout_indices"] = args.override_holdout_indices

    seed = args.seed or search_config.get("runner", {}).get("seed", 42)
    random.seed(seed)

    # Resolve paths
    repo_root = Path(__file__).resolve().parent.parent.parent
    output_root = Path(
        search_config.get("output_root", "./experiments/results/architecture_search")
    )
    if not output_root.is_absolute():
        output_root = repo_root / output_root

    search_name = search_config.get("search_name", "arch_search")
    output_dir = output_root / search_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save search config
    with (output_dir / "search_config.json").open("w") as f:
        json.dump(search_config, f, indent=2, ensure_ascii=False)

    # Setup registry and runner
    registry = RunRegistry(output_dir / "run_registry.json")
    runner = ExperimentRunner(
        search_config=search_config,
        repo_root=repo_root,
        output_root=output_dir,
        registry=registry,
        dry_run=args.dry_run,
    )

    # Print search space summary
    phases = args.phases.upper()
    logger.info("=" * 60)
    logger.info("Memory Architecture Search")
    logger.info("=" * 60)
    logger.info(f"Search name: {search_name}")
    logger.info(f"Output dir:  {output_dir}")
    logger.info(f"Phases:      {phases}")
    logger.info(f"Dry run:     {args.dry_run}")
    logger.info(f"Resume:      {args.resume}")
    logger.info(f"Seed:        {seed}")
    logger.info(f"Proxy set:   {search_config['dataset'].get('proxy_indices')}")
    logger.info(f"Held-out:    {search_config['dataset'].get('holdout_indices')}")
    logger.info(f"Extraction candidates: {list(EXTRACTION_PRESETS.keys())}")
    logger.info(f"Storage candidates:    {STORAGE_CANDIDATES}")
    logger.info(f"Retrieval candidates:  {RETRIEVAL_CANDIDATES}")
    logger.info(f"Management candidates: {MANAGEMENT_CANDIDATES}")

    anchor = search_config.get("anchor", DEFAULT_ANCHOR)
    logger.info(f"Anchor: {anchor}")

    # Compute total search space
    total_combos = (
        len(EXTRACTION_PRESETS) * len(STORAGE_CANDIDATES)
        * len(RETRIEVAL_CANDIDATES) * len(MANAGEMENT_CANDIDATES)
    )
    logger.info(f"Full search space: {total_combos} combinations")
    logger.info("")

    # ---- No-memory baseline ----
    baseline_result = None
    if search_config.get("phase_a", {}).get("run_no_memory_baseline", False):
        logger.info("\n--- Running no-memory baseline ---")
        baseline_result = runner.run_baseline()

    # ---- Phase A ----
    retained = None
    module_results_a = {}
    if "A" in phases:
        if args.parallel_phase_a and not args.screen_module:
            # Parallel Phase A: spawn 4 workers, one per module
            retained, module_results_a = _parallel_phase_a(
                args, search_config, output_dir, runner, baseline_result,
            )
        else:
            retained, module_results_a = phase_a(
                runner, search_config, output_dir,
                baseline_result=baseline_result,
                screen_module=args.screen_module,
            )
        reduced = 1
        for v in retained.values():
            reduced *= len(v)
        logger.info(f"\nReduced search space: {reduced} combinations (from {total_combos})")

    # ---- Phase B ----
    elites = None
    if "B" in phases:
        if retained is None:
            # Try loading from checkpoint or retained_candidates.json
            rc_path = output_dir / "retained_candidates.json"
            if rc_path.exists():
                with rc_path.open("r") as f:
                    retained = json.load(f)
            else:
                logger.error("Phase B requires Phase A results. Run Phase A first.")
                sys.exit(1)

        if args.phase_b_chunk:
            # Worker mode: only run assigned configs
            chunk_ids = set(json.loads(args.phase_b_chunk))
            combos = list(itertools.product(
                retained.get("extraction", ["mixed_all"]),
                retained.get("storage", ["hybrid"]),
                retained.get("retrieval", ["hybrid"]),
                retained.get("management", ["json_full"]),
            ))
            assigned = []
            for ext, st, ret, mgmt in combos:
                cfg = ArchConfig(extraction=ext, storage=st, retrieval=ret, management=mgmt)
                valid, _ = is_config_valid(cfg)
                if valid and cfg.config_id in chunk_ids:
                    assigned.append(cfg)

            logger.info(f"Phase B worker: {len(assigned)} configs assigned")
            for i, cfg in enumerate(assigned, 1):
                logger.info(f"\n[{i}/{len(assigned)}] {cfg.short_name}")
                runner.run(cfg, "phase_b")
            logger.info("Phase B worker done.")
            # Worker exits after running its chunk — no checkpoint/ranking
        elif args.parallel_phase_b:
            # Orchestrator mode: spawn workers across GPUs
            elites = _parallel_phase_b(
                args, retained, search_config, output_dir, runner,
                baseline_result=baseline_result,
            )
        else:
            elites = phase_b(
                runner, retained, search_config, output_dir,
                baseline_result=baseline_result,
            )

    # ---- Phase C (optional — skipped if phase_c.skip=true in config) ----
    final_top = None
    skip_c = search_config.get("phase_c", {}).get("skip", False)
    if "C" in phases and not skip_c:
        if elites is None:
            # Try loading from checkpoint
            te_path = output_dir / "top_elites.json"
            if te_path.exists():
                with te_path.open("r") as f:
                    elites = [ExperimentResult.from_dict(d) for d in json.load(f)]
            else:
                logger.error("Phase C requires Phase B results. Run Phase B first.")
                sys.exit(1)
        if retained is None:
            rc_path = output_dir / "retained_candidates.json"
            if rc_path.exists():
                with rc_path.open("r") as f:
                    retained = json.load(f)
            else:
                retained = {
                    "extraction": list(EXTRACTION_PRESETS.keys()),
                    "storage": STORAGE_CANDIDATES,
                    "retrieval": RETRIEVAL_CANDIDATES,
                    "management": MANAGEMENT_CANDIDATES,
                }
        final_top = phase_c(runner, elites, retained, search_config, output_dir)
    elif "C" in phases and skip_c:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE C: Skipped (phase_c.skip=true)")
        logger.info("=" * 60)
        # Promote Phase B elites directly as final_top
        if elites is not None:
            final_top = elites[:search_config.get("phase_d", {}).get("top_n", 3)]
            with (output_dir / "final_top3_configs.json").open("w") as f:
                json.dump([r.to_dict() for r in final_top], f, indent=2)

    # ---- Phase D ----
    if "D" in phases:
        if final_top is None:
            # Try loading from top_elites.json or final_top3_configs.json
            for candidate_path in [
                output_dir / "final_top3_configs.json",
                output_dir / "top_elites.json",
            ]:
                if candidate_path.exists():
                    with candidate_path.open("r") as f:
                        final_top = [ExperimentResult.from_dict(d) for d in json.load(f)]
                    break
            if final_top is None:
                logger.error("Phase D requires Phase B or C results.")
                sys.exit(1)
        phase_d(runner, final_top, search_config, output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("Architecture search complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
