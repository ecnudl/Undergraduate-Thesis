"""
Unit tests for GraphStore — Three-layer graph storage backend.

Covers:
  - Basic upsert and dedup
  - MemoryUnit.relations -> Content->Content edges
  - Entity linking (Content->Entity edges)
  - Query layer (Query->Content edges)
  - save / load persistence round-trip
  - Stats, neighbors, get_units_by_query, get_units_by_entity
  - Update, delete
  - Soft-delete reactivation
  - Heuristic entity extractor
"""

import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime

import numpy as np

from EvolveLab.memory_schema import (
    MemoryUnit,
    MemoryRelation,
    MemoryUnitType,
    RelationType,
)
from EvolveLab.storage.graph_storage import (
    GraphStore,
    QueryNode,
    EntityNode,
    normalize_entity_name,
    extract_entities_from_unit,
    LAYER_QUERY,
    LAYER_CONTENT,
    LAYER_ENTITY,
    EDGE_HAS_MEMORY,
    EDGE_HAS_ENTITY,
)


def _make_unit(
    uid: str = "u1",
    utype: MemoryUnitType = MemoryUnitType.TIP,
    content: dict = None,
    task_id: str = "task_001",
    query: str = "What is 2+2?",
    outcome: str = "success",
    relations: list = None,
    embedding: np.ndarray = None,
    signature: str = "",
    is_active: bool = True,
) -> MemoryUnit:
    if content is None:
        content = {"topic": "arithmetic", "principle": "basic addition", "category": "planning_and_decision"}
    unit = MemoryUnit(
        id=uid,
        type=utype,
        content=content,
        source_task_id=task_id,
        source_task_query=query,
        task_outcome=outcome,
        extraction_model="test-model",
        relations=relations or [],
        embedding=embedding if embedding is not None else np.random.randn(384).astype(np.float32),
        is_active=is_active,
    )
    if signature:
        unit.signature = signature
    else:
        unit.compute_signature()
    unit.token_estimate()
    return unit


class TestGraphStoreBasic(unittest.TestCase):
    """Test basic CRUD operations."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = GraphStore({"storage_dir": self.tmpdir})
        self.assertTrue(self.store.initialize())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_upsert_single_unit(self):
        unit = _make_unit("u1")
        nid = self.store.upsert_memory_unit(unit)
        self.assertEqual(nid, "m:u1")
        self.assertEqual(self.store.count(), 1)
        retrieved = self.store.get("u1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.type, MemoryUnitType.TIP)

    def test_add_multiple(self):
        u1 = _make_unit("u1", content={"topic": "a", "principle": "b"})
        u2 = _make_unit("u2", content={"topic": "c", "principle": "d"})
        added = self.store.add([u1, u2])
        self.assertEqual(added, 2)
        self.assertEqual(self.store.count(), 2)

    def test_get_all_filtered(self):
        u1 = _make_unit("u1", utype=MemoryUnitType.TIP, content={"topic": "a"})
        u2 = _make_unit("u2", utype=MemoryUnitType.INSIGHT, content={"root_cause_conclusion": "b"})
        self.store.add([u1, u2])
        tips = self.store.get_all(unit_type=MemoryUnitType.TIP)
        self.assertEqual(len(tips), 1)
        self.assertEqual(tips[0].id, "u1")

    def test_update(self):
        unit = _make_unit("u1")
        self.store.upsert_memory_unit(unit)
        unit.confidence = 0.42
        self.assertTrue(self.store.update(unit))
        check = self.store.get("u1")
        self.assertAlmostEqual(check.confidence, 0.42)

    def test_delete(self):
        unit = _make_unit("u1")
        self.store.upsert_memory_unit(unit)
        self.assertTrue(self.store.delete("u1"))
        self.assertEqual(self.store.count(), 0)
        self.assertIsNone(self.store.get("u1"))
        self.assertFalse(self.store.exists_signature(unit.signature))


class TestDedup(unittest.TestCase):
    """Test deduplication and merge behavior."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = GraphStore({"storage_dir": self.tmpdir})
        self.store.initialize()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_signature_dedup_merges_stats(self):
        u1 = _make_unit("u1", signature="sig_aaa", content={"topic": "x"})
        u1.usage_count = 3
        u1.success_count = 2
        self.store.upsert_memory_unit(u1)

        u2 = _make_unit("u2", signature="sig_aaa", content={"topic": "x"})
        u2.usage_count = 1
        u2.success_count = 1
        self.store.upsert_memory_unit(u2)

        # Should still be 1 unit (merged)
        self.assertEqual(self.store.count(), 1)
        merged = self.store.get("u1")
        self.assertEqual(merged.usage_count, 4)
        self.assertEqual(merged.success_count, 3)

    def test_add_dedup(self):
        u1 = _make_unit("u1", content={"topic": "same"})
        u2 = _make_unit("u2", content={"topic": "same"})  # same content -> same signature
        added = self.store.add([u1, u2])
        # u2 has same signature as u1, so it should be merged
        self.assertEqual(added, 1)
        self.assertEqual(self.store.count(), 1)

    def test_exists_signature(self):
        unit = _make_unit("u1", signature="sig_bbb")
        self.store.upsert_memory_unit(unit)
        self.assertTrue(self.store.exists_signature("sig_bbb"))
        self.assertFalse(self.store.exists_signature("sig_ccc"))

    def test_soft_delete_reactivation(self):
        u1 = _make_unit("u1", signature="sig_ddd", is_active=False)
        self.store.upsert_memory_unit(u1)
        self.assertFalse(self.store.get("u1").is_active)

        # Upsert same signature -> should reactivate
        u2 = _make_unit("u2", signature="sig_ddd", content=u1.content)
        u2.confidence = 0.99
        self.store.upsert_memory_unit(u2)

        self.assertEqual(self.store.count(), 1)
        reactivated = self.store.get("u1")
        self.assertTrue(reactivated.is_active)
        self.assertAlmostEqual(reactivated.confidence, 0.99)


class TestRelations(unittest.TestCase):
    """Test Content->Content relation edges."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = GraphStore({"storage_dir": self.tmpdir})
        self.store.initialize()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_relation_edges_created(self):
        u2 = _make_unit("u2", content={"topic": "target"})
        self.store.upsert_memory_unit(u2)

        rel = MemoryRelation(target_id="u2", relation_type=RelationType.SIMILAR, weight=0.9)
        u1 = _make_unit("u1", content={"topic": "source"}, relations=[rel])
        self.store.upsert_memory_unit(u1)

        # m:u1 should have an outgoing SIMILAR edge to m:u2
        nbrs = self.store.neighbors("m:u1", edge_type="SIMILAR", direction="out")
        self.assertIn("m:u2", nbrs)

    def test_relation_to_missing_target_skipped(self):
        rel = MemoryRelation(target_id="nonexistent", relation_type=RelationType.DEPENDS)
        u1 = _make_unit("u1", content={"topic": "solo"}, relations=[rel])
        self.store.upsert_memory_unit(u1)
        # Should not crash, edge silently skipped
        nbrs = self.store.neighbors("m:u1", direction="out")
        content_nbrs = [n for n in nbrs if n.startswith("m:")]
        self.assertEqual(len(content_nbrs), 0)

    def test_relation_merge_on_dedup(self):
        u_target = _make_unit("u_target", content={"topic": "t"})
        self.store.upsert_memory_unit(u_target)

        rel1 = MemoryRelation(target_id="u_target", relation_type=RelationType.SIMILAR)
        u1 = _make_unit("u1", signature="sig_rel", content={"topic": "s"}, relations=[rel1])
        self.store.upsert_memory_unit(u1)

        rel2 = MemoryRelation(target_id="u_target", relation_type=RelationType.DEPENDS)
        u2 = _make_unit("u2", signature="sig_rel", content={"topic": "s"}, relations=[rel2])
        self.store.upsert_memory_unit(u2)

        # u1 should now have both SIMILAR and DEPENDS relations
        merged = self.store.get("u1")
        rel_types = {r.relation_type for r in merged.relations}
        self.assertIn(RelationType.SIMILAR, rel_types)
        self.assertIn(RelationType.DEPENDS, rel_types)


class TestEntityLinking(unittest.TestCase):
    """Test Entity layer and Content->Entity edges."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = GraphStore({"storage_dir": self.tmpdir})
        self.store.initialize()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_upsert_with_entities(self):
        unit = _make_unit("u1")
        entities = [
            {"name": "Web Search", "type": "tool"},
            {"name": "Python", "type": "language"},
        ]
        self.store.upsert_memory_unit(unit, entities=entities)

        stats = self.store.stats()
        self.assertEqual(stats["nodes_by_layer"][LAYER_ENTITY], 2)

        # Content -> Entity edges
        nbrs = self.store.neighbors("m:u1", edge_type=EDGE_HAS_ENTITY, direction="out")
        self.assertEqual(len(nbrs), 2)

    def test_get_units_by_entity(self):
        u1 = _make_unit("u1", content={"topic": "a"})
        u2 = _make_unit("u2", content={"topic": "b"})
        ents = [{"name": "web_search", "type": "tool"}]
        self.store.upsert_memory_unit(u1, entities=ents)
        self.store.upsert_memory_unit(u2, entities=ents)

        results = self.store.get_units_by_entity("tool", "web_search")
        self.assertEqual(len(results), 2)
        ids = {u.id for u in results}
        self.assertEqual(ids, {"u1", "u2"})

    def test_entity_name_normalization(self):
        self.assertEqual(normalize_entity_name("Web Search Tool"), "web_search_tool")
        self.assertEqual(normalize_entity_name("  Hello, World! "), "hello_world")

    def test_upsert_entities_separately(self):
        unit = _make_unit("u1")
        self.store.upsert_memory_unit(unit)
        self.store.upsert_entities("u1", [{"name": "FAISS", "type": "tool"}])

        nbrs = self.store.neighbors("m:u1", edge_type=EDGE_HAS_ENTITY, direction="out")
        self.assertEqual(len(nbrs), 1)
        self.assertIn("e:tool:faiss", nbrs)


class TestQueryLayer(unittest.TestCase):
    """Test Query nodes and Query->Content edges."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = GraphStore({"storage_dir": self.tmpdir})
        self.store.initialize()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_query_node_created(self):
        unit = _make_unit("u1", task_id="task_42", query="What is X?")
        self.store.upsert_memory_unit(unit)

        self.assertTrue(self.store._graph.has_node("q:task_42"))
        attrs = self.store._graph.nodes["q:task_42"]
        self.assertEqual(attrs["layer"], LAYER_QUERY)
        self.assertEqual(attrs["query_text"], "What is X?")

    def test_query_to_content_edge(self):
        unit = _make_unit("u1", task_id="task_42")
        self.store.upsert_memory_unit(unit)

        nbrs = self.store.neighbors("q:task_42", edge_type=EDGE_HAS_MEMORY, direction="out")
        self.assertIn("m:u1", nbrs)

    def test_get_units_by_query(self):
        u1 = _make_unit("u1", task_id="task_1", content={"topic": "a"})
        u2 = _make_unit("u2", task_id="task_1", content={"topic": "b"})
        u3 = _make_unit("u3", task_id="task_2", content={"topic": "c"})
        self.store.add([u1, u2, u3])

        results = self.store.get_units_by_query("task_1")
        self.assertEqual(len(results), 2)
        ids = {u.id for u in results}
        self.assertEqual(ids, {"u1", "u2"})

    def test_empty_task_id_no_query_node(self):
        unit = _make_unit("u1", task_id="")
        self.store.upsert_memory_unit(unit)
        stats = self.store.stats()
        self.assertEqual(stats["nodes_by_layer"][LAYER_QUERY], 0)


class TestPersistence(unittest.TestCase):
    """Test save/load round-trip."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_save_load_roundtrip(self):
        store1 = GraphStore({"storage_dir": self.tmpdir})
        store1.initialize()

        u1 = _make_unit("u1", task_id="task_1", content={"topic": "a", "principle": "b"})
        u2 = _make_unit("u2", task_id="task_1", content={"topic": "c", "principle": "d"},
                        relations=[MemoryRelation("u1", RelationType.SIMILAR, 0.8)])
        store1.upsert_memory_unit(u1, entities=[{"name": "math", "type": "concept"}])
        store1.upsert_memory_unit(u2, entities=[{"name": "math", "type": "concept"}])
        store1.save()

        stats1 = store1.stats()

        # Load into fresh instance
        store2 = GraphStore({"storage_dir": self.tmpdir})
        store2.initialize()

        stats2 = store2.stats()
        self.assertEqual(stats1["total_nodes"], stats2["total_nodes"])
        self.assertEqual(stats1["total_edges"], stats2["total_edges"])
        self.assertEqual(stats1["memory_units"], stats2["memory_units"])

        # Verify unit data
        loaded = store2.get("u1")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.content["topic"], "a")
        self.assertIsNotNone(loaded.embedding)
        self.assertEqual(loaded.embedding.shape[0], 384)

        # Verify graph structure
        nbrs = store2.neighbors("q:task_1", edge_type=EDGE_HAS_MEMORY)
        self.assertEqual(len(nbrs), 2)

        entity_results = store2.get_units_by_entity("concept", "math")
        self.assertEqual(len(entity_results), 2)

    def test_save_load_with_real_data(self):
        """Load real MemoryUnits from existing storage if available."""
        real_db = os.path.join(
            os.path.dirname(__file__), "..", "..",
            "storage", "prompt_based_tip_trajectory", "memory_db.json"
        )
        if not os.path.exists(real_db):
            self.skipTest("Real memory_db.json not found")

        with open(real_db, "r") as f:
            data = json.load(f)
        units = [MemoryUnit.from_dict(d) for d in data[:5]]

        store = GraphStore({"storage_dir": self.tmpdir})
        store.initialize()
        added = store.add(units)
        self.assertGreater(added, 0)
        store.save()

        store2 = GraphStore({"storage_dir": self.tmpdir})
        store2.initialize()
        self.assertEqual(store2.count(), store.count())

    def test_embedding_persistence(self):
        store1 = GraphStore({"storage_dir": self.tmpdir})
        store1.initialize()

        emb = np.array([0.1, 0.2, 0.3] * 128, dtype=np.float32)  # 384-d
        unit = _make_unit("u1", embedding=emb)
        store1.upsert_memory_unit(unit)
        store1.save()

        store2 = GraphStore({"storage_dir": self.tmpdir})
        store2.initialize()
        loaded = store2.get("u1")
        np.testing.assert_allclose(loaded.embedding, emb, atol=1e-6)

    def test_embedding_index(self):
        store = GraphStore({"storage_dir": self.tmpdir})
        store.initialize()

        u1 = _make_unit("u1", content={"topic": "a"})
        u2 = _make_unit("u2", content={"topic": "b"}, is_active=False)
        u3 = _make_unit("u3", content={"topic": "c"})
        store.add([u1, u2, u3])

        mat, units = store.get_embedding_index(active_only=True)
        self.assertEqual(mat.shape[0], 2)  # u2 is inactive
        ids = {u.id for u in units}
        self.assertNotIn("u2", ids)


class TestNeighborsAndSubgraph(unittest.TestCase):
    """Test neighbors() and get_subgraph()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = GraphStore({"storage_dir": self.tmpdir})
        self.store.initialize()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_neighbors_direction(self):
        u1 = _make_unit("u1", content={"topic": "a"})
        u2 = _make_unit("u2", content={"topic": "b"},
                        relations=[MemoryRelation("u1", RelationType.DEPENDS)])
        self.store.add([u1, u2])

        # u2 -> u1 (DEPENDS)
        out_nbrs = self.store.neighbors("m:u2", direction="out")
        self.assertIn("m:u1", out_nbrs)

        in_nbrs = self.store.neighbors("m:u1", edge_type="DEPENDS", direction="in")
        self.assertIn("m:u2", in_nbrs)

        both_nbrs = self.store.neighbors("m:u1", direction="both")
        # u1 has incoming DEPENDS from u2, plus query edges
        self.assertIn("m:u2", both_nbrs)

    def test_get_subgraph(self):
        u1 = _make_unit("u1", task_id="t1", content={"topic": "a"})
        u2 = _make_unit("u2", task_id="t1", content={"topic": "b"})
        self.store.add([u1, u2])

        sub = self.store.get_subgraph(["m:u1", "m:u2", "q:t1"])
        self.assertEqual(sub.number_of_nodes(), 3)

    def test_neighbors_nonexistent_node(self):
        self.assertEqual(self.store.neighbors("m:nonexistent"), [])


class TestStats(unittest.TestCase):
    """Test stats() output."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = GraphStore({"storage_dir": self.tmpdir})
        self.store.initialize()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_stats_counts(self):
        u1 = _make_unit("u1", task_id="t1", content={"topic": "a"})
        u2 = _make_unit("u2", task_id="t2", content={"topic": "b"})
        ents = [{"name": "tool_x", "type": "tool"}]
        self.store.upsert_memory_unit(u1, entities=ents)
        self.store.upsert_memory_unit(u2, entities=ents)

        s = self.store.stats()
        self.assertEqual(s["memory_units"], 2)
        self.assertEqual(s["nodes_by_layer"][LAYER_QUERY], 2)
        self.assertEqual(s["nodes_by_layer"][LAYER_CONTENT], 2)
        self.assertEqual(s["nodes_by_layer"][LAYER_ENTITY], 1)  # shared entity
        self.assertEqual(s["edges_by_type"][EDGE_HAS_MEMORY], 2)
        self.assertEqual(s["edges_by_type"][EDGE_HAS_ENTITY], 2)


class TestHeuristicEntityExtractor(unittest.TestCase):
    """Test the simple entity extraction heuristic."""

    def test_tip_entities(self):
        unit = _make_unit("u1", content={"topic": "Web Search", "category": "tool_and_search"})
        ents = extract_entities_from_unit(unit)
        names = {e["normalized_name"] for e in ents}
        self.assertIn("web_search", names)
        self.assertIn("tool_and_search", names)

    def test_shortcut_entities(self):
        unit = _make_unit(
            "u1",
            utype=MemoryUnitType.SHORTCUT,
            content={"name": "Quick Calculator", "description": "fast math"},
        )
        ents = extract_entities_from_unit(unit)
        self.assertTrue(any(e["type"] == "action" for e in ents))

    def test_workflow_tool_extraction(self):
        unit = _make_unit(
            "u1",
            utype=MemoryUnitType.WORKFLOW,
            content={
                "agent_workflow": [
                    {"step": 1, "action": "Use WebSearchTool to find info"},
                    {"step": 2, "action": "Use CrawlPageTool to read page"},
                ]
            },
        )
        ents = extract_entities_from_unit(unit)
        names = {e["normalized_name"] for e in ents}
        self.assertIn("websearchtool", names)
        self.assertIn("crawlpagetool", names)


if __name__ == "__main__":
    unittest.main()
