"""
Memory Management Module.

Provides pluggable management operations that maintain memory quality through:
  - Episodic consolidation (clustering, workflow extraction, cross-task generalization)
  - Deduplication (signature, semantic, cross-type, conflict detection)
  - Failure-driven adjustment (penalize, boost, reflection, dynamic discard)
  - Maintenance (access stats, time decay, pruning, quality curation)

Operations are composed into pipelines via ManagementConfig and orchestrated
by ManagementPipeline.

Usage:
    from EvolveLab.management import ManagementPipeline, ManagementConfig
    from EvolveLab.management.presets import get_preset

    config = get_preset("json_basic")
    pipeline = ManagementPipeline(store, config, embedding_model, llm_client)
    pipeline.run_post_task({"task_succeeded": True, "used_unit_ids": [...]})
"""

from .base_op import (
    BaseManageOp,
    OpResult,
    ManagementConfig,
    ManagementResult,
    StorageCompatibility,
    TriggerType,
)
from .pipeline import ManagementPipeline, get_op_registry
from .presets import get_preset, list_presets

__all__ = [
    # Base types
    "BaseManageOp",
    "OpResult",
    "ManagementConfig",
    "ManagementResult",
    "StorageCompatibility",
    "TriggerType",
    # Pipeline
    "ManagementPipeline",
    "get_op_registry",
    # Presets
    "get_preset",
    "list_presets",
]
