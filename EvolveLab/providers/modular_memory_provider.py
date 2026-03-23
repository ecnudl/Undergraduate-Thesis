"""
ModularMemoryProvider — Adapter that bridges decoupled Storage + Retrieval
layers into the existing BaseMemoryProvider interface.

Extraction reuses existing prompt-based logic from prompt_based_memory_provider.
Storage and Retrieval are pluggable via config.

Config keys:
    enabled_prompts: List[str]       — ["tip", "insight", ...]
    storage_type: str                — "json" | "graph"
    retriever_type: str              — "semantic" | "keyword" | "hybrid" | "contrastive" | "graph"
    retriever_config: Dict           — strategy-specific params (weights, top_k, etc.)
    storage_dir: str                 — base dir for persistence
    top_k: int                       — max memories to retrieve (default 5)
    embedding_model_name: str        — sentence-transformers model id
    embedding_cache_dir: str         — local model cache
    prompt_dir: str                  — directory containing prompt .txt files

Usage:
    provider = ModularMemoryProvider(config={
        "enabled_prompts": ["tip", "insight"],
        "storage_type": "json",
        "retriever_type": "hybrid",
        "retriever_config": {"weights": {"SemanticRetriever": 0.7, "KeywordRetriever": 0.3}},
        "storage_dir": "./storage/modular_experiment_1",
    })
    provider.config["model"] = llm_model
    provider.initialize()
"""

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from ..base_memory import BaseMemoryProvider
from ..memory_types import (
    MemoryItem,
    MemoryRequest,
    MemoryResponse,
    MemoryStatus,
    MemoryType,
    TrajectoryData,
)
from ..memory_schema import MemoryUnit, MemoryUnitType, split_extraction_output

from .prompt_based_memory_provider import (
    PROMPT_FILE_NAMES,
    PROMPT_TO_UNIT_TYPE,
    _build_template_context,
    _load_embedding_model,
    _parse_json_from_response,
    _render_prompt,
)

logger = logging.getLogger(__name__)


# Storage type → (class, module path, config builder)
_STORAGE_FACTORIES = {
    "json": lambda cfg: _make_json_storage(cfg),
    "vector": lambda cfg: _make_vector_storage(cfg),
    "hybrid": lambda cfg: _make_hybrid_storage(cfg),
    "graph": lambda cfg: _make_graph_storage(cfg),
    "llm_graph": lambda cfg: _make_llm_graph_storage(cfg),
}

_GRAPH_STORAGE_TYPES = {"graph", "llm_graph"}
_GRAPH_RETRIEVER_TYPES = {"graph", "hybrid_graph"}
_GRAPH_ONLY_PRESETS = {"graph_full"}


def _make_json_storage(cfg):
    from ..storage import JsonStorage
    storage_cfg = dict(cfg)
    storage_cfg.setdefault("db_path", os.path.join(cfg["storage_dir"], "memory_db.json"))
    return JsonStorage(storage_cfg)


def _make_vector_storage(cfg):
    from ..storage import VectorStorage
    return VectorStorage(dict(cfg))


def _make_hybrid_storage(cfg):
    from ..storage import HybridStorage
    return HybridStorage(dict(cfg))


def _make_graph_storage(cfg):
    from ..storage import GraphStore
    return GraphStore(dict(cfg))


def _make_llm_graph_storage(cfg):
    from ..storage import LLMGraphStore
    return LLMGraphStore(dict(cfg))


def _make_retriever(retriever_type, store, embedding_model, retriever_config):
    """Create a retriever instance by type string."""
    from ..retrieval import (
        SemanticRetriever,
        KeywordRetriever,
        HybridRetriever,
        GraphRetriever,
        ContrastiveRetriever,
        HybridGraphRetriever,
    )

    if retriever_type == "semantic":
        return SemanticRetriever(store, embedding_model, retriever_config)

    elif retriever_type == "keyword":
        return KeywordRetriever(store, retriever_config)

    elif retriever_type == "hybrid":
        semantic = SemanticRetriever(store, embedding_model, retriever_config)
        keyword = KeywordRetriever(store, retriever_config)
        return HybridRetriever(store, [semantic, keyword], retriever_config)

    elif retriever_type == "graph":
        return GraphRetriever(store, embedding_model, retriever_config)

    elif retriever_type == "contrastive":
        return ContrastiveRetriever(store, embedding_model, retriever_config)

    elif retriever_type == "hybrid_graph":
        return HybridGraphRetriever(store, embedding_model, retriever_config)

    else:
        logger.warning(f"Unknown retriever_type '{retriever_type}', falling back to semantic")
        return SemanticRetriever(store, embedding_model, retriever_config)


class ModularMemoryProvider(BaseMemoryProvider):
    """
    Adapter provider that composes:
      - Extraction: reuses prompt_based_memory_provider logic
      - Storage: pluggable backend (JsonStorage / GraphStore)
      - Retrieval: pluggable strategy (5 options)

    Implements BaseMemoryProvider interface so it works with existing
    GAIA runner and agent framework unchanged.
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(MemoryType.MODULAR, config)

        self.model = self.config.get("model")

        # Environment variable overrides (same pattern as prompt_based)
        # MODULAR_STORAGE_DIR, MODULAR_STORAGE_TYPE, MODULAR_RETRIEVER_TYPE,
        # MODULAR_RETRIEVER_CONFIG (JSON), MODULAR_ENABLED_PROMPTS (comma-sep)
        self.storage_dir = (
            os.environ.get("MODULAR_STORAGE_DIR")
            or self.config.get("storage_dir", "./storage/modular")
        )
        self.storage_type = (
            os.environ.get("MODULAR_STORAGE_TYPE")
            or self.config.get("storage_type", "json")
        )
        env_storage_config = os.environ.get("MODULAR_STORAGE_CONFIG", "").strip()
        if env_storage_config:
            import json as _json
            try:
                self.storage_config = _json.loads(env_storage_config)
            except Exception:
                logger.warning("Invalid MODULAR_STORAGE_CONFIG JSON, using config default")
                self.storage_config = self.config.get("storage_config", {})
        else:
            self.storage_config = self.config.get("storage_config", {})
        self.retriever_type = (
            os.environ.get("MODULAR_RETRIEVER_TYPE")
            or self.config.get("retriever_type", "semantic")
        )

        # Retriever config: env var as JSON, or from config dict
        env_retriever_config = os.environ.get("MODULAR_RETRIEVER_CONFIG", "").strip()
        if env_retriever_config:
            import json as _json
            try:
                self.retriever_config = _json.loads(env_retriever_config)
            except Exception:
                logger.warning(f"Invalid MODULAR_RETRIEVER_CONFIG JSON, using config default")
                self.retriever_config = self.config.get("retriever_config", {})
        else:
            self.retriever_config = self.config.get("retriever_config", {})

        env_top_k = os.environ.get("MODULAR_TOP_K", "").strip()
        if env_top_k:
            try:
                self.top_k = int(env_top_k)
            except ValueError:
                logger.warning(f"Invalid MODULAR_TOP_K='{env_top_k}', using config/default")
                self.top_k = self.config.get("top_k", 3)
        else:
            self.top_k = self.config.get("top_k", 3)

        # Injection threshold: memories below this relevance score are discarded
        env_min_rel = os.environ.get("MODULAR_MIN_RELEVANCE", "").strip()
        if env_min_rel:
            self.min_relevance = float(env_min_rel)
        else:
            self.min_relevance = float(self.config.get("min_relevance", 0.20))

        # Adaptive gating: skip injection entirely if best memory is below this
        env_gate = os.environ.get("MODULAR_GATE_THRESHOLD", "").strip()
        if env_gate:
            self.gate_threshold = float(env_gate)
        else:
            self.gate_threshold = float(self.config.get("gate_threshold", 0.30))

        # Extraction config
        env_prompts = os.environ.get("MODULAR_ENABLED_PROMPTS", "").strip()
        if env_prompts:
            self.enabled_prompts = [p.strip() for p in env_prompts.split(",") if p.strip()]
        else:
            self.enabled_prompts: List[str] = self.config.get(
                "enabled_prompts", ["tip", "insight"]
            )
        self.prompt_dir = (
            os.environ.get("MODULAR_PROMPT_DIR")
            or self.config.get("prompt_dir", ".")
        )

        # Embedding model config
        self.embedding_model_name = self.config.get(
            "embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_cache_dir = self.config.get(
            "embedding_cache_dir", "./storage/models"
        )

        # Management config
        self._management_enabled = (
            os.environ.get("MODULAR_MANAGEMENT_ENABLED", "true").lower()
            not in ("false", "0", "no")
        )
        self._management_preset = (
            os.environ.get("MODULAR_MANAGEMENT_PRESET", "").strip()
            or self.config.get("management_preset", "")
        )
        self._management_ops_override = (
            os.environ.get("MODULAR_MANAGEMENT_OPS", "").strip()
        )

        # Internal state (initialized during initialize())
        self.store = None
        self.retriever = None
        self.embedding_model = None
        self.manager = None
        self._prompt_templates: Dict[str, str] = {}
        self._last_provided_ids: List[str] = []
        self._experiment_metrics: Dict[str, Any] = {}
        self.reset_experiment_metrics()

    def _resolved_management_preset(self) -> str:
        if not self._management_enabled:
            return "none"
        return self._management_preset or self.storage_type

    def _graph_stats(self) -> Dict[str, Any]:
        if self.store is None or not hasattr(self.store, "stats"):
            return {"graph_nodes": None, "graph_edges": None}
        try:
            stats = self.store.stats()
            return {
                "graph_nodes": stats.get("total_nodes"),
                "graph_edges": stats.get("total_edges"),
            }
        except Exception:
            return {"graph_nodes": None, "graph_edges": None}

    def _update_memory_totals(self) -> None:
        num_units = self.store.count() if self.store is not None else 0
        self._experiment_metrics["num_memory_units"] = num_units
        self._experiment_metrics.update(self._graph_stats())

    def _record_management_results(self, results: List[Any]) -> None:
        triggered = 0
        serialized = self._experiment_metrics.setdefault("management_results", [])
        for phase_result in results:
            for op_result in phase_result.results:
                if op_result.triggered:
                    triggered += 1
                serialized.append(
                    {
                        "phase": phase_result.phase,
                        "op_name": op_result.op_name,
                        "triggered": op_result.triggered,
                        "units_affected": op_result.units_affected,
                        "units_created": op_result.units_created,
                        "units_deleted": op_result.units_deleted,
                        "units_modified": op_result.units_modified,
                        "duration_ms": op_result.duration_ms,
                        "details": op_result.details,
                    }
                )
        self._experiment_metrics["management_ops_triggered"] += triggered

    def _validate_runtime_compatibility(self) -> None:
        if (
            self.retriever_type in _GRAPH_RETRIEVER_TYPES
            and self.storage_type not in _GRAPH_STORAGE_TYPES
        ):
            raise ValueError(
                f"Retriever '{self.retriever_type}' requires graph storage, "
                f"got '{self.storage_type}'"
            )

        if (
            self._management_enabled
            and self._resolved_management_preset() in _GRAPH_ONLY_PRESETS
            and self.storage_type not in _GRAPH_STORAGE_TYPES
        ):
            raise ValueError(
                f"Management preset '{self._resolved_management_preset()}' requires "
                f"GraphStore or LLMGraphStore, got '{self.storage_type}'"
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            self._validate_runtime_compatibility()

            # 1. Load embedding model
            self.embedding_model = _load_embedding_model(
                self.embedding_model_name, self.embedding_cache_dir
            )

            # 2. Create and initialize storage backend
            factory = _STORAGE_FACTORIES.get(self.storage_type)
            if factory is None:
                logger.error(f"Unknown storage_type: {self.storage_type}")
                return False

            self.store = factory({"storage_dir": self.storage_dir, **self.storage_config})
            if not self.store.initialize():
                logger.error("Storage backend initialization failed")
                return False

            # 3. Create retriever
            self.retriever = _make_retriever(
                self.retriever_type,
                self.store,
                self.embedding_model,
                self.retriever_config,
            )

            # 4. Load prompt templates
            for prompt_name in self.enabled_prompts:
                fname = PROMPT_FILE_NAMES.get(prompt_name)
                if not fname:
                    logger.warning(f"Unknown prompt name: {prompt_name}, skipping")
                    continue
                fpath = os.path.join(self.prompt_dir, fname)
                if not os.path.exists(fpath):
                    logger.warning(f"Prompt file not found: {fpath}, skipping")
                    continue
                with open(fpath, "r", encoding="utf-8") as f:
                    self._prompt_templates[prompt_name] = f.read()

            # 5. Initialize management pipeline
            self.manager = None
            if self._management_enabled:
                try:
                    from ..management import ManagementPipeline, ManagementConfig
                    from ..management.presets import get_preset

                    mgmt_config = self.config.get("management_config")
                    if self._management_ops_override:
                        # Build config from env var override
                        ops_list = [
                            o.strip()
                            for o in self._management_ops_override.split(",")
                            if o.strip()
                        ]
                        mgmt_config = ManagementConfig(
                            post_task_ops=ops_list,
                            periodic_ops=ops_list,
                            on_insert_ops=[
                                o for o in ops_list
                                if o in ("signature_dedup", "conflict_detection")
                            ],
                        )
                    elif mgmt_config is not None:
                        mgmt_config = ManagementConfig(**mgmt_config)
                    else:
                        preset_name = self._resolved_management_preset()
                        mgmt_config = get_preset(preset_name)

                    self.manager = ManagementPipeline(
                        store=self.store,
                        config=mgmt_config,
                        embedding_model=self.embedding_model,
                        llm_client=self.model,
                    )
                except Exception as e:
                    logger.warning(f"Management pipeline init failed (non-fatal): {e}")
                    self.manager = None

            logger.info(
                f"ModularMemoryProvider initialized: "
                f"storage={self.storage_type}, retriever={self.retriever_type}, "
                f"prompts={list(self._prompt_templates.keys())}, "
                f"management={'enabled' if self.manager else 'disabled'}, "
                f"existing_units={self.store.count()}"
            )
            self.reset_experiment_metrics()
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ModularMemoryProvider: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    # Memory ingestion (reuses prompt_based extraction logic)
    # ------------------------------------------------------------------

    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple:
        if not self.model:
            return False, "No model provided for memory extraction"

        if not self._prompt_templates:
            return False, "No prompt templates loaded"

        metadata = trajectory_data.metadata or {}
        is_correct = metadata.get("is_correct", False)
        task_outcome = "success" if is_correct else "failure"
        task_id = metadata.get("task_id", str(uuid.uuid4())[:8])

        context = _build_template_context(trajectory_data, is_correct)

        new_units: List[MemoryUnit] = []
        prompts_used = []
        extracted_count = 0

        for prompt_name, template_str in self._prompt_templates.items():
            unit_type = PROMPT_TO_UNIT_TYPE.get(prompt_name)
            if unit_type is None:
                continue

            # Skip conditions
            if prompt_name == "insight" and is_correct:
                continue
            if prompt_name == "workflow" and not is_correct:
                continue

            try:
                filled_prompt = _render_prompt(template_str, context)
            except Exception as e:
                logger.error(f"Template rendering failed for {prompt_name}: {e}")
                continue

            try:
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": filled_prompt}]}
                ]
                response = self.model(messages)
                response_text = (
                    response.content if hasattr(response, "content") else str(response)
                )
            except Exception as e:
                logger.error(f"LLM call failed for {prompt_name}: {e}")
                continue

            parsed = _parse_json_from_response(response_text)
            if parsed is None:
                logger.warning(f"Failed to parse extraction result for {prompt_name}")
                continue

            if isinstance(parsed, dict) and parsed.get("skipped"):
                continue

            try:
                units = split_extraction_output(
                    extraction_result=parsed,
                    unit_type=unit_type,
                    source_task_id=task_id,
                    source_task_query=trajectory_data.query,
                    task_outcome=task_outcome,
                    extraction_model=str(getattr(self.model, "model_id", "unknown")),
                )
            except Exception as e:
                logger.error(f"split_extraction_output failed for {prompt_name}: {e}")
                continue

            extracted_count += len(units)

            # Dedup + embed + add to storage
            for unit in units:
                if self.store.exists_signature(unit.signature):
                    continue

                text = unit.content_text()
                if text and self.embedding_model is not None:
                    unit.embedding = self.embedding_model.encode(
                        text, convert_to_numpy=True
                    )

                new_units.append(unit)

            prompts_used.append(prompt_name)

        # Batch add to storage
        inserted_count = 0
        if new_units:
            before_count = self.store.count()
            if self.storage_type in ("graph", "llm_graph") and hasattr(self.store, 'upsert_memory_unit'):
                # For GraphStore/LLMGraphStore: use upsert_memory_unit
                # LLMGraphStore will auto-run LLM entity extraction pipeline
                for unit in new_units:
                    if self.storage_type == "llm_graph":
                        # LLMGraphStore extracts entities via LLM internally
                        self.store.upsert_memory_unit(unit)
                    else:
                        from ..storage.graph_storage import extract_entities_from_unit
                        entities = extract_entities_from_unit(unit)
                        self.store.upsert_memory_unit(unit, entities=entities)
            else:
                self.store.add(new_units)
            inserted_count = max(self.store.count() - before_count, 0)
            self.store.save()
            logger.info(
                f"ModularMemoryProvider: added {inserted_count} units from "
                f"{', '.join(prompts_used)} (total: {self.store.count()})"
            )

        msg = (
            f"Extracted {len(new_units)} units from "
            f"{len(prompts_used)} prompts ({', '.join(prompts_used)})"
        )
        self._experiment_metrics["num_extracted"] += extracted_count
        self._experiment_metrics["num_inserted"] += inserted_count
        self._experiment_metrics["num_deduped"] += max(extracted_count - inserted_count, 0)

        # Run management pipeline
        if self.manager is not None:
            try:
                if new_units:
                    self.manager.run_on_insert(
                        new_units,
                        {"new_unit_ids": [u.id for u in new_units]},
                    )
                self.manager.run_post_task({
                    "task_id": task_id,
                    "task_succeeded": is_correct,
                    "used_unit_ids": self._last_provided_ids,
                    "task_query": trajectory_data.query,
                })
                self._last_provided_ids = []
                self._record_management_results(self.manager.consume_recent_results())
            except Exception as e:
                logger.warning(f"Management pipeline error (non-fatal): {e}")

        self._update_memory_totals()

        return True, msg

    # ------------------------------------------------------------------
    # Memory retrieval (delegates to Retriever → MemoryPack)
    # ------------------------------------------------------------------

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        empty_response = MemoryResponse(
            memories=[],
            memory_type=self.memory_type,
            total_count=0,
            request_id=str(uuid.uuid4()),
        )

        # Only provide at BEGIN phase
        if request.status != MemoryStatus.BEGIN:
            return empty_response

        if self.retriever is None or self.store is None:
            return empty_response

        if self.store.count() == 0:
            return empty_response

        # Build query context
        from ..retrieval import QueryContext

        query_emb = None
        if self.embedding_model is not None:
            query_emb = self.embedding_model.encode(
                request.query, convert_to_numpy=True
            )

        ctx = QueryContext(
            query=request.query,
            embedding=query_emb,
        )

        # Retrieve
        self._experiment_metrics["retrieval_calls"] += 1
        pack = self.retriever.retrieve(ctx, top_k=self.top_k)

        if pack.is_empty():
            self._update_memory_totals()
            return empty_response

        # Adaptive gating: if the best retrieved memory is below gate_threshold,
        # the memory pool has nothing confidently relevant — skip injection entirely
        # to avoid polluting the agent with low-relevance noise.
        if self.gate_threshold > 0 and pack.scored_units:
            best_score = max(su.score for su in pack.scored_units)
            if best_score < self.gate_threshold:
                logger.info(
                    f"gate_threshold={self.gate_threshold}: best score {best_score:.3f} "
                    f"is below gate, skipping memory injection entirely"
                )
                self._update_memory_totals()
                return empty_response

        # Apply min_relevance threshold: discard individual low-relevance memories
        if self.min_relevance > 0:
            original_count = len(pack.scored_units)
            pack.scored_units = [
                su for su in pack.scored_units if su.score >= self.min_relevance
            ]
            filtered_count = original_count - len(pack.scored_units)
            if filtered_count > 0:
                logger.info(
                    f"min_relevance={self.min_relevance}: filtered {filtered_count}/{original_count} "
                    f"low-relevance memories, {len(pack.scored_units)} remaining"
                )
            if pack.is_empty():
                logger.info("All memories below min_relevance threshold, returning empty")
                self._update_memory_totals()
                return empty_response

        # Convert MemoryPack → guidance text → MemoryResponse
        guidance_text = pack.to_guidance_text()
        self._experiment_metrics["num_retrieved"] += len(pack.scored_units)
        self._experiment_metrics["retriever_name"] = pack.retriever_name

        memory_item = MemoryItem(
            id=f"modular_{uuid.uuid4()}",
            content=guidance_text,
            metadata={
                "retriever": pack.retriever_name,
                "num_units": len(pack.scored_units),
                "by_type": {k: len(v) for k, v in pack.by_type.items()},
                "total_memory_units": self.store.count(),
            },
            score=float(np.mean([su.score for su in pack.scored_units])) if pack.scored_units else 0.0,
        )

        logger.info(
            f"provide_memory: {pack.retriever_name} returned {len(pack.scored_units)} units "
            f"(query='{request.query[:60]}...', total={self.store.count()})"
        )

        # Track provided memory IDs for management feedback
        self._last_provided_ids = [su.unit.id for su in pack.scored_units]
        self._update_memory_totals()

        return MemoryResponse(
            memories=[memory_item],
            memory_type=self.memory_type,
            total_count=1,
            request_id=str(uuid.uuid4()),
        )

    def reset_experiment_metrics(self) -> None:
        self._last_provided_ids = []
        self._experiment_metrics = {
            "storage_backend": self.storage_type,
            "retrieval_strategy": self.retriever_type,
            "management_preset": self._resolved_management_preset(),
            "num_extracted": 0,
            "num_memory_units": self.store.count() if self.store is not None else 0,
            "num_inserted": 0,
            "num_deduped": 0,
            "num_retrieved": 0,
            "retrieval_calls": 0,
            "management_ops_triggered": 0,
            "management_results": [],
            "retriever_name": None,
            "graph_nodes": None,
            "graph_edges": None,
        }
        if self.manager is not None and hasattr(self.manager, "clear_recent_results"):
            self.manager.clear_recent_results()
        self._update_memory_totals()

    def get_experiment_metrics(self) -> Dict[str, Any]:
        self._update_memory_totals()
        return dict(self._experiment_metrics)
