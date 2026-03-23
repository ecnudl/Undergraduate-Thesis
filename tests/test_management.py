"""
Tests for the management module — Phase 4 of the memory evolution system.

Covers all 16 operations, the pipeline orchestrator, and presets.
Uses JsonStorage with mock LLM/embedding for deterministic testing.
"""

import json
import os
import tempfile
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from EvolveLab.memory_schema import (
    MemoryUnit,
    MemoryUnitType,
    MemoryRelation,
    RelationType,
)
from EvolveLab.storage.json_storage import JsonStorage
from EvolveLab.storage.graph_storage import GraphStore
from EvolveLab.management.base_op import (
    BaseManageOp,
    ManagementConfig,
    ManagementResult,
    OpResult,
    StorageCompatibility,
    TriggerType,
)
from EvolveLab.management.pipeline import ManagementPipeline, get_op_registry
from EvolveLab.management.presets import get_preset, list_presets


# ======================================================================
# Fixtures
# ======================================================================

def _make_unit(
    unit_type=MemoryUnitType.TIP,
    content=None,
    task_id="task-1",
    task_outcome="success",
    confidence=0.8,
    usage_count=0,
    success_count=0,
    embedding=None,
    is_active=True,
    decay_weight=1.0,
    signature=None,
) -> MemoryUnit:
    """Create a test MemoryUnit with defaults."""
    if content is None:
        content = {"topic": "test", "principle": "test principle"}
    unit = MemoryUnit(
        id=str(uuid.uuid4()),
        type=unit_type,
        content=content,
        source_task_id=task_id,
        source_task_query="test query",
        task_outcome=task_outcome,
        confidence=confidence,
        usage_count=usage_count,
        success_count=success_count,
        is_active=is_active,
        decay_weight=decay_weight,
        created_at=(datetime.now() - timedelta(hours=48)).isoformat(),
    )
    if embedding is None:
        embedding = np.random.randn(384).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
    unit.embedding = embedding
    if signature:
        unit.signature = signature
    else:
        unit.compute_signature()
    return unit


def _make_json_store(units: List[MemoryUnit]) -> JsonStorage:
    """Create a temp JsonStorage populated with units."""
    tmpdir = tempfile.mkdtemp()
    store = JsonStorage({"db_path": os.path.join(tmpdir, "test_db.json")})
    store.initialize()
    if units:
        store.add(units)
    return store


def _make_graph_store(units: List[MemoryUnit]) -> GraphStore:
    """Create a temp GraphStore populated with units."""
    tmpdir = tempfile.mkdtemp()
    store = GraphStore({"storage_dir": os.path.join(tmpdir, "graph")})
    store.initialize()
    if units:
        store.add(units)
    return store


def _mock_embedding_model():
    """Mock embedding model that returns random vectors."""
    model = MagicMock()
    model.encode = MagicMock(
        side_effect=lambda text, **kwargs: np.random.randn(384).astype(np.float32)
    )
    return model


def _mock_llm_client(response_text="{}"):
    """Mock LLM client that returns a fixed response."""
    response = MagicMock()
    response.content = response_text

    def call_fn(messages):
        return response

    return call_fn


# ======================================================================
# Test: BaseManageOp
# ======================================================================

class TestBaseManageOp:
    def test_storage_compatibility_all(self):
        """ALL compatibility works with any storage type."""
        op = MagicMock(spec=BaseManageOp)
        op.storage_compatibility = StorageCompatibility.ALL
        op.is_compatible = BaseManageOp.is_compatible.__get__(op)
        assert op.is_compatible("json")
        assert op.is_compatible("graph")
        assert op.is_compatible("vector")

    def test_storage_compatibility_graph_only(self):
        op = MagicMock(spec=BaseManageOp)
        op.storage_compatibility = StorageCompatibility.GRAPH_ONLY
        op.is_compatible = BaseManageOp.is_compatible.__get__(op)
        assert op.is_compatible("graph")
        assert op.is_compatible("llm_graph")
        assert not op.is_compatible("json")

    def test_op_result_repr(self):
        r = OpResult(op_name="test", triggered=True, units_affected=3)
        assert "test" in repr(r)
        assert "affected=3" in repr(r)

    def test_management_config_defaults(self):
        cfg = ManagementConfig()
        assert cfg.post_task_ops == []
        assert cfg.periodic_interval == 10


# ======================================================================
# Test: AccessStatsUpdateOp
# ======================================================================

class TestAccessStatsUpdateOp:
    def test_updates_access_stats(self):
        from EvolveLab.management.ops.access_stats_update import AccessStatsUpdateOp

        # Use unique content to avoid signature dedup in store.add()
        units = [_make_unit(content={"topic": f"unique-{i}", "principle": f"p-{i}"}) for i in range(3)]
        store = _make_json_store(units)

        # Verify units are in store
        for u in units:
            assert store.get(u.id) is not None, f"Unit {u.id} not in store"

        op = AccessStatsUpdateOp(store, {})
        result = op.execute({"used_unit_ids": [units[0].id, units[1].id]})

        assert result.triggered
        assert result.units_modified == 2

        u0 = store.get(units[0].id)
        assert u0.access_count == 1
        assert u0.last_accessed is not None

    def test_empty_used_ids(self):
        from EvolveLab.management.ops.access_stats_update import AccessStatsUpdateOp

        store = _make_json_store([])
        op = AccessStatsUpdateOp(store, {})
        result = op.execute({"used_unit_ids": []})
        assert result.units_modified == 0


# ======================================================================
# Test: TimeDecayOp
# ======================================================================

class TestTimeDecayOp:
    def test_applies_decay(self):
        from EvolveLab.management.ops.time_decay import TimeDecayOp

        units = [
            _make_unit(decay_weight=1.0, content={"topic": f"decay-{i}", "principle": f"p-{i}"})
            for i in range(3)
        ]
        store = _make_json_store(units)

        op = TimeDecayOp(store, {"decay_rate": 0.9, "access_refresh": 0.1})
        result = op.execute({})

        assert result.triggered
        for u in units:
            updated = store.get(u.id)
            assert updated is not None, f"Unit {u.id} not found"
            assert updated.decay_weight < 1.0


# ======================================================================
# Test: QualityCurationOp
# ======================================================================

class TestQualityCurationOp:
    def test_updates_confidence(self):
        from EvolveLab.management.ops.quality_curation import QualityCurationOp

        unit = _make_unit(usage_count=10, success_count=8, confidence=0.5)
        store = _make_json_store([unit])

        op = QualityCurationOp(store, {"bayesian_weight": 0.7})
        result = op.execute({})

        assert result.triggered
        updated = store.get(unit.id)
        # new_confidence = 0.7 * 0.8 + 0.3 * 0.5 = 0.56 + 0.15 = 0.71
        assert abs(updated.confidence - 0.71) < 0.02


# ======================================================================
# Test: PenalizeOnFailureOp
# ======================================================================

class TestPenalizeOnFailureOp:
    def test_penalizes_on_failure(self):
        from EvolveLab.management.ops.penalize_on_failure import PenalizeOnFailureOp

        unit = _make_unit(confidence=0.8)
        store = _make_json_store([unit])

        op = PenalizeOnFailureOp(store, {"confidence_penalty": 0.15})
        result = op.execute({
            "task_succeeded": False,
            "used_unit_ids": [unit.id],
        })

        assert result.triggered
        updated = store.get(unit.id)
        assert updated.confidence < 0.8
        assert updated.usage_count == 1
        assert updated.success_count == 0

    def test_skips_on_success(self):
        from EvolveLab.management.ops.penalize_on_failure import PenalizeOnFailureOp

        store = _make_json_store([_make_unit()])
        op = PenalizeOnFailureOp(store, {})
        result = op.execute({"task_succeeded": True, "used_unit_ids": []})
        assert not result.triggered


# ======================================================================
# Test: BoostOnSuccessOp
# ======================================================================

class TestBoostOnSuccessOp:
    def test_boosts_on_success(self):
        from EvolveLab.management.ops.boost_on_success import BoostOnSuccessOp

        unit = _make_unit(confidence=0.7)
        store = _make_json_store([unit])

        op = BoostOnSuccessOp(store, {"success_boost": 0.1})
        result = op.execute({
            "task_succeeded": True,
            "used_unit_ids": [unit.id],
        })

        assert result.triggered
        updated = store.get(unit.id)
        assert abs(updated.confidence - 0.8) < 0.01

    def test_skips_on_failure(self):
        from EvolveLab.management.ops.boost_on_success import BoostOnSuccessOp

        store = _make_json_store([_make_unit()])
        op = BoostOnSuccessOp(store, {})
        result = op.execute({"task_succeeded": False, "used_unit_ids": []})
        assert not result.triggered


# ======================================================================
# Test: SignatureDedupOp
# ======================================================================

class TestSignatureDedupOp:
    def test_deduplicates_same_signature(self):
        from EvolveLab.management.ops.signature_dedup import SignatureDedupOp

        content = {"topic": "same", "principle": "same principle"}
        u1 = _make_unit(content=content, confidence=0.9)
        u2 = _make_unit(content=content, confidence=0.5)
        # Force same signature
        u2.signature = u1.signature

        # Use graph store which has dict-based _units
        store = _make_graph_store([])
        store._units[u1.id] = u1
        store._units[u2.id] = u2
        store._sig_to_id[u1.signature] = u1.id

        # Add graph nodes
        nid1 = store._content_nid(u1.id)
        nid2 = store._content_nid(u2.id)
        store._graph.add_node(nid1, layer="content")
        store._graph.add_node(nid2, layer="content")

        op = SignatureDedupOp(store, {})
        result = op.execute({})

        assert result.triggered
        assert result.units_deleted >= 1

    def test_no_duplicates(self):
        from EvolveLab.management.ops.signature_dedup import SignatureDedupOp

        units = [
            _make_unit(content={"topic": f"unique-{i}"})
            for i in range(3)
        ]
        store = _make_json_store(units)

        op = SignatureDedupOp(store, {})
        result = op.execute({})
        assert not result.triggered


# ======================================================================
# Test: SemanticDedupOp
# ======================================================================

class TestSemanticDedupOp:
    def test_deduplicates_similar_embeddings(self):
        from EvolveLab.management.ops.semantic_dedup import SemanticDedupOp

        # Create two units with nearly identical embeddings
        base_emb = np.random.randn(384).astype(np.float32)
        base_emb /= np.linalg.norm(base_emb)

        u1 = _make_unit(
            content={"topic": "a"},
            embedding=base_emb.copy(),
            confidence=0.9,
        )
        u2 = _make_unit(
            content={"topic": "b"},
            embedding=base_emb + np.random.randn(384).astype(np.float32) * 0.01,
            confidence=0.5,
        )
        # Normalize u2 embedding
        u2.embedding /= np.linalg.norm(u2.embedding)

        store = _make_json_store([u1, u2])

        op = SemanticDedupOp(store, {"semantic_threshold": 0.95}, _mock_embedding_model())
        result = op.execute({})

        assert result.triggered
        assert result.units_deleted >= 1

    def test_keeps_dissimilar(self):
        from EvolveLab.management.ops.semantic_dedup import SemanticDedupOp

        units = [_make_unit(content={"topic": f"topic-{i}"}) for i in range(3)]
        store = _make_json_store(units)

        op = SemanticDedupOp(store, {"semantic_threshold": 0.99}, _mock_embedding_model())
        result = op.execute({})
        # Random embeddings unlikely to exceed 0.99 threshold
        assert result.units_deleted == 0


# ======================================================================
# Test: CrossTypeDedupOp
# ======================================================================

class TestCrossTypeDedupOp:
    def test_deactivates_covered_insight(self):
        from EvolveLab.management.ops.cross_type_dedup import CrossTypeDedupOp

        base_emb = np.random.randn(384).astype(np.float32)
        base_emb /= np.linalg.norm(base_emb)

        tip = _make_unit(
            unit_type=MemoryUnitType.TIP,
            content={"topic": "same topic"},
            embedding=base_emb.copy(),
        )
        insight = _make_unit(
            unit_type=MemoryUnitType.INSIGHT,
            content={"root_cause_conclusion": "same topic"},
            embedding=base_emb + np.random.randn(384).astype(np.float32) * 0.01,
        )
        insight.embedding /= np.linalg.norm(insight.embedding)

        store = _make_json_store([tip, insight])

        op = CrossTypeDedupOp(store, {"cross_type_threshold": 0.95}, _mock_embedding_model())
        result = op.execute({})

        assert result.triggered


# ======================================================================
# Test: ConflictDetectionOp
# ======================================================================

class TestConflictDetectionOp:
    def test_detects_conflict(self):
        from EvolveLab.management.ops.conflict_detection import ConflictDetectionOp

        base_emb = np.random.randn(384).astype(np.float32)
        base_emb /= np.linalg.norm(base_emb)

        u_success = _make_unit(
            content={"topic": "conflict-a", "principle": "p-a"},
            task_outcome="success",
            embedding=base_emb.copy(),
            confidence=0.9,
        )
        u_failure = _make_unit(
            content={"topic": "conflict-b", "principle": "p-b"},
            task_outcome="failure",
            embedding=base_emb.copy(),  # exact same embedding for guaranteed match
            confidence=0.3,
        )

        store = _make_json_store([u_success, u_failure])

        op = ConflictDetectionOp(store, {"conflict_threshold": 0.95}, _mock_embedding_model())
        result = op.execute({"new_unit_ids": [u_success.id, u_failure.id]})

        assert result.triggered


# ======================================================================
# Test: DynamicDiscardOp
# ======================================================================

class TestDynamicDiscardOp:
    def test_discards_high_failure_rate(self):
        from EvolveLab.management.ops.dynamic_discard import DynamicDiscardOp

        unit = _make_unit(usage_count=10, success_count=1)  # 10% success
        store = _make_json_store([unit])

        op = DynamicDiscardOp(store, {"min_usage": 5, "max_failure_rate": 0.8})
        result = op.execute({})

        assert result.triggered
        updated = store.get(unit.id)
        assert not updated.is_active

    def test_keeps_successful_units(self):
        from EvolveLab.management.ops.dynamic_discard import DynamicDiscardOp

        unit = _make_unit(usage_count=10, success_count=8)  # 80% success
        store = _make_json_store([unit])

        op = DynamicDiscardOp(store, {"min_usage": 5, "max_failure_rate": 0.8})
        result = op.execute({})

        assert not result.triggered
        assert store.get(unit.id).is_active


# ======================================================================
# Test: ScoreBasedPruneOp
# ======================================================================

class TestScoreBasedPruneOp:
    def test_prunes_low_score(self):
        from EvolveLab.management.ops.score_based_prune import ScoreBasedPruneOp

        low_unit = _make_unit(
            confidence=0.01, decay_weight=0.01,
            content={"topic": "low-score", "principle": "p-low"},
        )
        high_unit = _make_unit(
            confidence=0.9, decay_weight=1.0,
            content={"topic": "high-score", "principle": "p-high"},
        )
        store = _make_json_store([low_unit, high_unit])

        # Verify both are in the store
        assert store.get(low_unit.id) is not None
        assert store.get(high_unit.id) is not None

        op = ScoreBasedPruneOp(store, {"min_effective_score": 0.05, "max_memory_count": 500})
        result = op.execute({})

        assert result.triggered
        assert not store.get(low_unit.id).is_active
        assert store.get(high_unit.id).is_active

    def test_budget_pruning(self):
        from EvolveLab.management.ops.score_based_prune import ScoreBasedPruneOp

        units = [_make_unit(confidence=0.1 * (i + 1)) for i in range(5)]
        store = _make_json_store(units)

        op = ScoreBasedPruneOp(store, {"min_effective_score": 0.0, "max_memory_count": 3})
        result = op.execute({})

        active = [u for u in store.get_all() if u.is_active]
        assert len(active) <= 3


# ======================================================================
# Test: ReflectionCorrectionOp (with mock LLM)
# ======================================================================

class TestReflectionCorrectionOp:
    def test_skips_on_success(self):
        from EvolveLab.management.ops.reflection_correction import ReflectionCorrectionOp

        store = _make_json_store([_make_unit()])
        op = ReflectionCorrectionOp(store, {}, _mock_embedding_model(), _mock_llm_client())
        result = op.execute({"task_succeeded": True, "used_unit_ids": []})
        assert not result.triggered

    def test_handles_empty_used_ids(self):
        from EvolveLab.management.ops.reflection_correction import ReflectionCorrectionOp

        store = _make_json_store([_make_unit()])
        op = ReflectionCorrectionOp(store, {}, _mock_embedding_model(), _mock_llm_client())
        result = op.execute({"task_succeeded": False, "used_unit_ids": []})
        # Should not crash, either triggered=False or triggered with 0 affected
        assert result.units_modified == 0


# ======================================================================
# Test: ClusterMergeOp (with mock LLM)
# ======================================================================

class TestClusterMergeOp:
    def test_merges_similar_cluster(self):
        from EvolveLab.management.ops.cluster_merge import ClusterMergeOp

        base_emb = np.random.randn(384).astype(np.float32)
        base_emb /= np.linalg.norm(base_emb)

        units = []
        for i in range(4):
            noise = np.random.randn(384).astype(np.float32) * 0.02
            emb = base_emb + noise
            emb /= np.linalg.norm(emb)
            units.append(_make_unit(
                content={"topic": f"topic-{i}", "principle": f"principle-{i}"},
                embedding=emb,
            ))

        store = _make_json_store(units)

        llm_response = json.dumps({
            "topic": "merged topic",
            "principle": "merged principle",
        })
        op = ClusterMergeOp(
            store, {"similarity_threshold": 0.95, "min_cluster_size": 3},
            _mock_embedding_model(), _mock_llm_client(llm_response),
        )
        result = op.execute({})

        # Whether it triggers depends on the random noise, but should not crash
        assert isinstance(result, OpResult)


# ======================================================================
# Test: TrajectoryToWorkflowOp (with mock LLM)
# ======================================================================

class TestTrajectoryToWorkflowOp:
    def test_converts_eligible_trajectory(self):
        from EvolveLab.management.ops.trajectory_to_workflow import TrajectoryToWorkflowOp

        traj = _make_unit(
            unit_type=MemoryUnitType.TRAJECTORY,
            content={"steps": [{"action": "search", "observation": "found"}]},
            usage_count=5,
            success_count=4,
        )
        store = _make_json_store([traj])

        llm_response = json.dumps({
            "agent_workflow": [
                {"step": 1, "action": "search", "rationale": "find info"}
            ],
        })
        op = TrajectoryToWorkflowOp(
            store, {"min_usage": 3, "min_success_rate": 0.7},
            llm_client=_mock_llm_client(llm_response),
        )
        result = op.execute({})

        assert result.triggered
        assert result.units_created >= 1


# ======================================================================
# Test: CrossTaskGeneralizeOp
# ======================================================================

class TestCrossTaskGeneralizeOp:
    def test_no_crash_with_few_units(self):
        from EvolveLab.management.ops.cross_task_generalize import CrossTaskGeneralizeOp

        unit = _make_unit(task_id="task-1")
        store = _make_json_store([unit])

        llm_response = json.dumps({
            "topic": "general", "principle": "general principle",
        })
        op = CrossTaskGeneralizeOp(
            store, {}, _mock_embedding_model(), _mock_llm_client(llm_response),
        )
        result = op.execute({})
        assert isinstance(result, OpResult)


# ======================================================================
# Test: ReindexRelationsOp
# ======================================================================

class TestReindexRelationsOp:
    def test_adds_similar_relations(self):
        from EvolveLab.management.ops.reindex_relations import ReindexRelationsOp

        base_emb = np.random.randn(384).astype(np.float32)
        base_emb /= np.linalg.norm(base_emb)

        u1 = _make_unit(embedding=base_emb.copy())
        u2 = _make_unit(
            embedding=base_emb + np.random.randn(384).astype(np.float32) * 0.05,
        )
        u2.embedding /= np.linalg.norm(u2.embedding)

        store = _make_json_store([u1, u2])

        op = ReindexRelationsOp(store, {"similarity_threshold": 0.8}, _mock_embedding_model())
        result = op.execute({})

        assert isinstance(result, OpResult)


# ======================================================================
# Test: Pipeline
# ======================================================================

class TestManagementPipeline:
    def test_pipeline_initialization(self):
        store = _make_json_store([])
        config = ManagementConfig(
            post_task_ops=["access_stats_update", "boost_on_success", "penalize_on_failure"],
            periodic_ops=["time_decay"],
            on_insert_ops=["signature_dedup"],
        )
        pipeline = ManagementPipeline(store, config)
        assert len(pipeline._post_task_ops) == 3
        assert len(pipeline._periodic_ops) == 1
        assert len(pipeline._on_insert_ops) == 1

    def test_skips_llm_ops_without_client(self):
        store = _make_json_store([])
        config = ManagementConfig(
            post_task_ops=["reflection_correction"],  # requires LLM
        )
        pipeline = ManagementPipeline(store, config)
        # reflection_correction should be skipped
        assert len(pipeline._post_task_ops) == 0

    def test_skips_embedding_ops_without_model(self):
        store = _make_json_store([])
        config = ManagementConfig(
            periodic_ops=["semantic_dedup"],  # requires embedding
        )
        pipeline = ManagementPipeline(store, config)
        assert len(pipeline._periodic_ops) == 0

    def test_run_post_task(self):
        unit = _make_unit(confidence=0.7)
        store = _make_json_store([unit])
        config = ManagementConfig(
            post_task_ops=["access_stats_update", "boost_on_success"],
        )
        pipeline = ManagementPipeline(store, config)

        result = pipeline.run_post_task({
            "task_succeeded": True,
            "used_unit_ids": [unit.id],
        })

        assert isinstance(result, ManagementResult)
        assert result.phase == "post_task"
        assert len(result.results) == 2

        # Check access was updated
        updated = store.get(unit.id)
        assert updated.access_count == 1

    def test_boost_penalize_mutual_exclusion(self):
        unit = _make_unit()
        store = _make_json_store([unit])
        config = ManagementConfig(
            post_task_ops=["boost_on_success", "penalize_on_failure"],
        )
        pipeline = ManagementPipeline(store, config)

        # On success: boost triggered, penalize not
        result = pipeline.run_post_task({
            "task_succeeded": True,
            "used_unit_ids": [unit.id],
        })
        boost_result = result.results[0]
        penalize_result = result.results[1]
        assert boost_result.triggered
        assert not penalize_result.triggered

    def test_periodic_trigger(self):
        store = _make_json_store([_make_unit(decay_weight=1.0)])
        config = ManagementConfig(
            post_task_ops=["access_stats_update"],
            periodic_ops=["time_decay"],
            periodic_interval=2,
        )
        pipeline = ManagementPipeline(store, config)

        # Task 1: no periodic
        pipeline.run_post_task({"task_succeeded": True, "used_unit_ids": []})
        # Task 2: periodic triggers
        pipeline.run_post_task({"task_succeeded": True, "used_unit_ids": []})

        assert pipeline._task_counter == 2

    def test_run_on_insert(self):
        unit = _make_unit()
        store = _make_json_store([unit])
        config = ManagementConfig(
            on_insert_ops=["signature_dedup"],
        )
        pipeline = ManagementPipeline(store, config)

        result = pipeline.run_on_insert([unit])
        assert isinstance(result, ManagementResult)
        assert result.phase == "on_insert"

    def test_unknown_op_skipped(self):
        store = _make_json_store([])
        config = ManagementConfig(
            post_task_ops=["nonexistent_op", "access_stats_update"],
        )
        pipeline = ManagementPipeline(store, config)
        assert len(pipeline._post_task_ops) == 1  # only access_stats_update


# ======================================================================
# Test: Presets
# ======================================================================

class TestPresets:
    def test_list_presets(self):
        presets = list_presets()
        assert "json_basic" in presets
        assert "json_full" in presets
        assert "graph_full" in presets
        assert "lightweight" in presets

    def test_get_preset_by_name(self):
        config = get_preset("json_basic")
        assert isinstance(config, ManagementConfig)
        assert "access_stats_update" in config.post_task_ops

    def test_get_preset_by_storage_type(self):
        config = get_preset("json")
        assert isinstance(config, ManagementConfig)
        assert "signature_dedup" in config.on_insert_ops

    def test_graph_preset_has_graph_ops(self):
        config = get_preset("graph_full")
        assert "cross_task_generalize" in config.periodic_ops
        assert "reindex_relations" in config.periodic_ops


# ======================================================================
# Test: OpRegistry
# ======================================================================

class TestOpRegistry:
    def test_registry_has_all_ops(self):
        registry = get_op_registry()
        expected = [
            "cluster_merge", "trajectory_to_workflow", "cross_task_generalize",
            "reindex_relations", "signature_dedup", "semantic_dedup",
            "cross_type_dedup", "conflict_detection", "penalize_on_failure",
            "boost_on_success", "reflection_correction", "dynamic_discard",
            "access_stats_update", "time_decay", "score_based_prune",
            "quality_curation",
        ]
        for name in expected:
            assert name in registry, f"Missing op: {name}"

    def test_all_ops_have_unique_rl_action_ids(self):
        registry = get_op_registry()
        ids = set()
        for name, cls in registry.items():
            if cls.rl_action_id >= 0:
                assert cls.rl_action_id not in ids, f"Duplicate RL action ID: {cls.rl_action_id}"
                ids.add(cls.rl_action_id)


# ======================================================================
# Test: Graph store enhanced operations
# ======================================================================

class TestGraphStoreOps:
    def test_signature_dedup_with_graph(self):
        from EvolveLab.management.ops.signature_dedup import SignatureDedupOp

        content = {"topic": "same", "principle": "same"}
        u1 = _make_unit(content=content, confidence=0.9)
        u2 = _make_unit(content=content, confidence=0.5)
        u2.signature = u1.signature

        store = _make_graph_store([])
        store._units[u1.id] = u1
        store._units[u2.id] = u2
        store._sig_to_id[u1.signature] = u1.id

        # Add graph nodes
        nid1 = store._content_nid(u1.id)
        nid2 = store._content_nid(u2.id)
        store._graph.add_node(nid1, layer="content")
        store._graph.add_node(nid2, layer="content")

        op = SignatureDedupOp(store, {})
        result = op.execute({})

        assert isinstance(result, OpResult)

    def test_pipeline_with_graph_store(self):
        store = _make_graph_store([_make_unit()])
        config = get_preset("graph_full")
        # Only test with non-LLM, non-embedding ops
        config.post_task_ops = ["access_stats_update"]
        config.periodic_ops = []
        config.on_insert_ops = []

        pipeline = ManagementPipeline(store, config)
        result = pipeline.run_post_task({
            "task_succeeded": True,
            "used_unit_ids": [],
        })
        assert isinstance(result, ManagementResult)


# ======================================================================
# Test: End-to-end pipeline flow
# ======================================================================

class TestEndToEnd:
    def test_full_lifecycle(self):
        """Test a complete lifecycle: add units, run management, verify state."""
        # Create units
        units = []
        for i in range(5):
            u = _make_unit(
                content={"topic": f"topic-{i}", "principle": f"principle-{i}"},
                usage_count=i,
                success_count=max(0, i - 1),
            )
            units.append(u)

        store = _make_json_store(units)

        # Create pipeline with lightweight preset
        config = get_preset("lightweight")
        pipeline = ManagementPipeline(store, config)

        # Simulate a successful task
        result = pipeline.run_post_task({
            "task_succeeded": True,
            "used_unit_ids": [units[0].id, units[1].id],
        })
        assert isinstance(result, ManagementResult)

        # Run periodic
        result = pipeline.run_periodic()
        assert isinstance(result, ManagementResult)

        # Run on_insert
        new_unit = _make_unit(content={"topic": "new"})
        store.add([new_unit])
        result = pipeline.run_on_insert([new_unit])
        assert isinstance(result, ManagementResult)

        # Verify store state is consistent
        all_units = store.get_all()
        assert len(all_units) >= 5  # original + new

    def test_pipeline_with_all_ops(self):
        """Smoke test: run json_full preset with mock LLM and embedding."""
        units = [_make_unit(usage_count=3, success_count=2) for _ in range(3)]
        store = _make_json_store(units)

        config = get_preset("json_full")
        pipeline = ManagementPipeline(
            store, config,
            embedding_model=_mock_embedding_model(),
            llm_client=_mock_llm_client('{"action": "KEEP"}'),
        )

        # Post-task
        result = pipeline.run_post_task({
            "task_succeeded": False,
            "used_unit_ids": [units[0].id],
            "task_query": "test query",
        })
        assert isinstance(result, ManagementResult)

        # Periodic
        result = pipeline.run_periodic()
        assert isinstance(result, ManagementResult)
