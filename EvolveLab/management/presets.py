"""
Preset pipeline configurations for common storage + management combinations.

Each preset returns a ManagementConfig tailored to a specific storage type
and management intensity level.
"""

from .base_op import ManagementConfig


# ======================================================================
# Preset definitions
# ======================================================================

def json_basic() -> ManagementConfig:
    """Basic management for JsonStorage — no LLM, minimal overhead."""
    return ManagementConfig(
        post_task_ops=[
            "access_stats_update",
            "boost_on_success",
            "penalize_on_failure",
            "signature_dedup",
        ],
        periodic_ops=[
            "time_decay",
            "quality_curation",
            "semantic_dedup",
            "score_based_prune",
        ],
        on_insert_ops=[
            "signature_dedup",
            "conflict_detection",
        ],
        periodic_interval=10,
    )


def json_full() -> ManagementConfig:
    """Full management for JsonStorage — includes LLM-based operations."""
    return ManagementConfig(
        post_task_ops=[
            "access_stats_update",
            "boost_on_success",
            "penalize_on_failure",
            "reflection_correction",
            "signature_dedup",
        ],
        periodic_ops=[
            "time_decay",
            "quality_curation",
            "semantic_dedup",
            "cross_type_dedup",
            "dynamic_discard",
            "cluster_merge",
            "trajectory_to_workflow",
            "score_based_prune",
            "reindex_relations",
        ],
        on_insert_ops=[
            "signature_dedup",
            "conflict_detection",
        ],
        periodic_interval=10,
    )


def graph_full() -> ManagementConfig:
    """Full management for GraphStore/LLMGraphStore — leverages graph structure."""
    return ManagementConfig(
        post_task_ops=[
            "access_stats_update",
            "boost_on_success",
            "penalize_on_failure",
            "reflection_correction",
            "signature_dedup",
        ],
        periodic_ops=[
            "time_decay",
            "quality_curation",
            "semantic_dedup",
            "cross_type_dedup",
            "dynamic_discard",
            "cluster_merge",
            "trajectory_to_workflow",
            "cross_task_generalize",
            "score_based_prune",
            "reindex_relations",
        ],
        on_insert_ops=[
            "signature_dedup",
            "conflict_detection",
        ],
        periodic_interval=10,
    )


def lightweight() -> ManagementConfig:
    """Minimal overhead — no LLM, no embedding computation."""
    return ManagementConfig(
        post_task_ops=[
            "access_stats_update",
            "boost_on_success",
            "penalize_on_failure",
            "signature_dedup",
        ],
        periodic_ops=[
            "time_decay",
            "quality_curation",
            "dynamic_discard",
            "score_based_prune",
        ],
        on_insert_ops=[
            "signature_dedup",
        ],
        periodic_interval=10,
    )


def none() -> ManagementConfig:
    """Disable all management operations."""
    return ManagementConfig(
        post_task_ops=[],
        periodic_ops=[],
        on_insert_ops=[],
        periodic_interval=10,
    )


# ======================================================================
# Preset lookup
# ======================================================================

_PRESETS = {
    "none": none,
    "json_basic": json_basic,
    "json_full": json_full,
    "graph_full": graph_full,
    "lightweight": lightweight,
}

# Storage type -> default preset mapping
_STORAGE_DEFAULT_PRESET = {
    "json": "json_basic",
    "vector": "json_basic",
    "hybrid": "json_basic",
    "graph": "graph_full",
    "llm_graph": "graph_full",
}


def get_preset(name_or_storage_type: str) -> ManagementConfig:
    """
    Get a preset ManagementConfig by name or storage type.

    Args:
        name_or_storage_type: Either a preset name ("json_basic", "lightweight", etc.)
            or a storage type ("json", "graph", etc.) to auto-select the default preset.

    Returns:
        ManagementConfig instance.
    """
    # Try direct preset name first
    if name_or_storage_type in _PRESETS:
        return _PRESETS[name_or_storage_type]()

    # Try storage type -> default preset
    preset_name = _STORAGE_DEFAULT_PRESET.get(name_or_storage_type, "json_basic")
    return _PRESETS[preset_name]()


def list_presets() -> list:
    """Return available preset names."""
    return list(_PRESETS.keys())
