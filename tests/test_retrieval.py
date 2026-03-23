"""
Unit tests for the Retrieval layer — all 5 retrievers + MemoryPack formatting.

Covers:
  - SemanticRetriever: cosine similarity ranking
  - KeywordRetriever: TF-IDF text matching
  - HybridRetriever: weighted fusion of sub-retrievers
  - GraphRetriever: seed + graph expansion
  - ContrastiveRetriever: success/failure differentiated search
  - MemoryPack: to_prompt_string(), by_type grouping, evidence
"""

import shutil
import tempfile
import unittest

import numpy as np

from EvolveLab.memory_schema import MemoryUnit, MemoryUnitType, MemoryRelation, RelationType
from EvolveLab.storage import JsonStorage, GraphStore
from EvolveLab.retrieval import (
    QueryContext,
    ScoredUnit,
    MemoryPack,
    TraceEntry,
    EvidenceRef,
    SemanticRetriever,
    KeywordRetriever,
    HybridRetriever,
    GraphRetriever,
    ContrastiveRetriever,
)


def _make_unit(
    uid: str,
    utype: MemoryUnitType = MemoryUnitType.TIP,
    content: dict = None,
    task_id: str = "task_1",
    query: str = "How to search the web?",
    outcome: str = "success",
    embedding: np.ndarray = None,
) -> MemoryUnit:
    if content is None:
        content = {"topic": f"topic_{uid}", "principle": f"principle_{uid}", "micro_example": "ex"}
    unit = MemoryUnit(
        id=uid,
        type=utype,
        content=content,
        source_task_id=task_id,
        source_task_query=query,
        task_outcome=outcome,
        extraction_model="test",
        embedding=embedding if embedding is not None else np.random.randn(384).astype(np.float32),
    )
    unit.compute_signature()
    unit.token_estimate()
    return unit


def _make_store_with_units(units, tmpdir):
    """Create a JsonStorage with pre-loaded units."""
    import os
    store = JsonStorage({"db_path": os.path.join(tmpdir, "db.json")})
    store.initialize()
    store.add(units)
    return store


class TestMemoryPack(unittest.TestCase):
    """Test MemoryPack formatting and accessors."""

    def test_empty_pack(self):
        ctx = QueryContext(query="test")
        pack = MemoryPack(query_context=ctx)
        self.assertTrue(pack.is_empty())
        self.assertEqual(pack.to_prompt_string(), "")
        self.assertEqual(pack.total_tokens, 0)

    def test_to_prompt_string_with_markers(self):
        ctx = QueryContext(query="test")
        u = _make_unit("u1")
        su = ScoredUnit(unit=u, score=0.85, method="semantic")
        pack = MemoryPack(query_context=ctx, scored_units=[su])

        result = pack.to_prompt_string()
        self.assertIn("----Memory System Guidance----", result)
        self.assertIn("----End Memory----", result)
        self.assertIn("[TIP]", result)
        self.assertIn("topic_u1", result)
        self.assertIn("0.85", result)

    def test_custom_markers(self):
        ctx = QueryContext(query="test")
        u = _make_unit("u1")
        su = ScoredUnit(unit=u, score=0.5, method="test")
        pack = MemoryPack(query_context=ctx, scored_units=[su])

        result = pack.to_prompt_string(
            begin_marker="<BEGIN>", end_marker="<END>"
        )
        self.assertTrue(result.startswith("<BEGIN>"))
        self.assertTrue(result.endswith("<END>"))

    def test_by_type_grouping(self):
        ctx = QueryContext(query="test")
        u1 = _make_unit("u1", utype=MemoryUnitType.TIP)
        u2 = _make_unit("u2", utype=MemoryUnitType.INSIGHT,
                        content={"root_cause_conclusion": "bad", "state_mismatch_analysis": "x"})
        pack = MemoryPack(
            query_context=ctx,
            scored_units=[
                ScoredUnit(unit=u1, score=0.9, method="test"),
                ScoredUnit(unit=u2, score=0.8, method="test"),
            ],
        )
        by_type = pack.by_type
        self.assertIn("tip", by_type)
        self.assertIn("insight", by_type)
        self.assertEqual(len(by_type["tip"]), 1)

    def test_selected_units(self):
        ctx = QueryContext(query="test")
        u1 = _make_unit("u1")
        u2 = _make_unit("u2")
        pack = MemoryPack(
            query_context=ctx,
            scored_units=[
                ScoredUnit(unit=u1, score=0.9, method="test"),
                ScoredUnit(unit=u2, score=0.8, method="test"),
            ],
        )
        self.assertEqual(len(pack.selected_units), 2)
        self.assertEqual(pack.selected_units[0].id, "u1")

    def test_to_guidance_text_no_markers(self):
        ctx = QueryContext(query="test")
        u = _make_unit("u1")
        pack = MemoryPack(
            query_context=ctx,
            scored_units=[ScoredUnit(unit=u, score=0.5, method="test")],
        )
        text = pack.to_guidance_text()
        self.assertNotIn("Memory System Guidance", text)
        self.assertIn("[TIP]", text)

    def test_to_dict(self):
        ctx = QueryContext(query="test query")
        u = _make_unit("u1")
        pack = MemoryPack(
            query_context=ctx,
            scored_units=[ScoredUnit(unit=u, score=0.9, method="semantic")],
            trace=[TraceEntry(step=1, method="semantic", candidates=10, selected=1)],
            evidence=[EvidenceRef(unit_id="u1", unit_type="tip", snippet="test", score=0.9)],
            retriever_name="SemanticRetriever",
        )
        d = pack.to_dict()
        self.assertEqual(d["query"], "test query")
        self.assertEqual(d["num_units"], 1)
        self.assertEqual(d["retriever"], "SemanticRetriever")

    def test_max_units(self):
        ctx = QueryContext(query="test")
        units = [_make_unit(f"u{i}") for i in range(10)]
        scored = [ScoredUnit(unit=u, score=0.5, method="test") for u in units]
        pack = MemoryPack(query_context=ctx, scored_units=scored)
        result = pack.to_prompt_string(max_units=3)
        # Should only contain 3 units
        self.assertEqual(result.count("[TIP]"), 3)

    def test_workflow_format(self):
        ctx = QueryContext(query="test")
        u = _make_unit("u1", utype=MemoryUnitType.WORKFLOW, content={
            "agent_workflow": [
                {"step": 1, "action": "Search web"},
                {"step": 2, "action": "Parse results"},
            ]
        })
        pack = MemoryPack(
            query_context=ctx,
            scored_units=[ScoredUnit(unit=u, score=0.7, method="test")],
        )
        text = pack.to_prompt_string()
        self.assertIn("[WORKFLOW]", text)
        self.assertIn("Step 1: Search web", text)


class TestSemanticRetriever(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create units with known embeddings
        self.u1 = _make_unit("u1", embedding=np.array([1, 0, 0] * 128, dtype=np.float32))
        self.u2 = _make_unit("u2", embedding=np.array([0, 1, 0] * 128, dtype=np.float32))
        self.u3 = _make_unit("u3", embedding=np.array([0.9, 0.1, 0] * 128, dtype=np.float32))
        self.store = _make_store_with_units([self.u1, self.u2, self.u3], self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_basic_retrieval(self):
        retriever = SemanticRetriever(self.store)
        # Query similar to u1
        ctx = QueryContext(
            query="test",
            embedding=np.array([1, 0, 0] * 128, dtype=np.float32),
        )
        pack = retriever.retrieve(ctx, top_k=2)
        self.assertFalse(pack.is_empty())
        self.assertEqual(len(pack.scored_units), 2)
        # u1 should be most similar
        self.assertEqual(pack.scored_units[0].unit.id, "u1")
        self.assertAlmostEqual(pack.scored_units[0].score, 1.0, places=3)

    def test_min_score_filter(self):
        retriever = SemanticRetriever(self.store, config={"min_score": 0.9})
        ctx = QueryContext(
            query="test",
            embedding=np.array([1, 0, 0] * 128, dtype=np.float32),
        )
        pack = retriever.retrieve(ctx, top_k=5)
        # Only u1 and u3 should pass (u2 is orthogonal)
        ids = {su.unit.id for su in pack.scored_units}
        self.assertIn("u1", ids)
        self.assertNotIn("u2", ids)

    def test_no_embedding(self):
        retriever = SemanticRetriever(self.store)
        ctx = QueryContext(query="test")  # no embedding, no model
        pack = retriever.retrieve(ctx)
        self.assertTrue(pack.is_empty())
        self.assertEqual(len(pack.trace), 1)
        self.assertEqual(pack.trace[0].params.get("error"), "no_embedding")

    def test_trace_populated(self):
        retriever = SemanticRetriever(self.store)
        ctx = QueryContext(
            query="test",
            embedding=np.array([1, 0, 0] * 128, dtype=np.float32),
        )
        pack = retriever.retrieve(ctx)
        self.assertEqual(len(pack.trace), 1)
        self.assertEqual(pack.trace[0].method, "semantic")
        self.assertEqual(pack.trace[0].candidates, 3)

    def test_evidence_populated(self):
        retriever = SemanticRetriever(self.store)
        ctx = QueryContext(
            query="test",
            embedding=np.array([1, 0, 0] * 128, dtype=np.float32),
        )
        pack = retriever.retrieve(ctx, top_k=2)
        self.assertEqual(len(pack.evidence), 2)
        self.assertEqual(pack.evidence[0].unit_type, "tip")

    def test_retriever_name(self):
        retriever = SemanticRetriever(self.store)
        self.assertEqual(retriever.name, "SemanticRetriever")
        ctx = QueryContext(query="test", embedding=np.random.randn(384).astype(np.float32))
        pack = retriever.retrieve(ctx)
        self.assertEqual(pack.retriever_name, "SemanticRetriever")


class TestKeywordRetriever(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.u1 = _make_unit("u1", content={"topic": "web search optimization", "principle": "use keywords"})
        self.u2 = _make_unit("u2", content={"topic": "file parsing PDF", "principle": "extract text"})
        self.u3 = _make_unit("u3", content={"topic": "web crawling", "principle": "follow links"})
        self.store = _make_store_with_units([self.u1, self.u2, self.u3], self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_keyword_match(self):
        retriever = KeywordRetriever(self.store)
        ctx = QueryContext(query="web search")
        pack = retriever.retrieve(ctx, top_k=2)
        self.assertFalse(pack.is_empty())
        # u1 ("web search optimization") should rank highest
        self.assertEqual(pack.scored_units[0].unit.id, "u1")

    def test_no_match(self):
        retriever = KeywordRetriever(self.store, config={"min_score": 0.5})
        ctx = QueryContext(query="quantum computing entanglement")
        pack = retriever.retrieve(ctx, top_k=5)
        # No units should match
        self.assertTrue(pack.is_empty())

    def test_empty_query(self):
        retriever = KeywordRetriever(self.store)
        ctx = QueryContext(query="")
        pack = retriever.retrieve(ctx)
        self.assertTrue(pack.is_empty())

    def test_pdf_query(self):
        retriever = KeywordRetriever(self.store)
        ctx = QueryContext(query="parse PDF file")
        pack = retriever.retrieve(ctx, top_k=1)
        self.assertEqual(pack.scored_units[0].unit.id, "u2")


class TestHybridRetriever(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # u1: high semantic match to query, low keyword
        self.u1 = _make_unit("u1",
            content={"topic": "vector similarity", "principle": "cosine distance"},
            embedding=np.array([1, 0, 0] * 128, dtype=np.float32))
        # u2: low semantic, high keyword
        self.u2 = _make_unit("u2",
            content={"topic": "web search optimization tips", "principle": "use web search"},
            embedding=np.array([0, 1, 0] * 128, dtype=np.float32))
        # u3: medium both
        self.u3 = _make_unit("u3",
            content={"topic": "web search", "principle": "use vectors"},
            embedding=np.array([0.7, 0.7, 0] * 128, dtype=np.float32))
        self.store = _make_store_with_units([self.u1, self.u2, self.u3], self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_hybrid_fusion(self):
        semantic = SemanticRetriever(self.store)
        keyword = KeywordRetriever(self.store)
        hybrid = HybridRetriever(
            self.store,
            sub_retrievers=[semantic, keyword],
            config={"weights": {"SemanticRetriever": 0.7, "KeywordRetriever": 0.3}},
        )
        ctx = QueryContext(
            query="web search",
            embedding=np.array([1, 0, 0] * 128, dtype=np.float32),
        )
        pack = hybrid.retrieve(ctx, top_k=3)
        self.assertFalse(pack.is_empty())
        self.assertEqual(pack.retriever_name, "HybridRetriever")
        # Should have trace entries from both sub-retrievers + fusion
        self.assertGreaterEqual(len(pack.trace), 3)

    def test_equal_weights(self):
        semantic = SemanticRetriever(self.store)
        keyword = KeywordRetriever(self.store)
        hybrid = HybridRetriever(self.store, sub_retrievers=[semantic, keyword])
        ctx = QueryContext(
            query="web search",
            embedding=np.array([0.7, 0.7, 0] * 128, dtype=np.float32),
        )
        pack = hybrid.retrieve(ctx, top_k=3)
        self.assertFalse(pack.is_empty())
        # All methods should be "hybrid"
        for su in pack.scored_units:
            self.assertEqual(su.method, "hybrid")


class TestGraphRetriever(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.graph_store = GraphStore({"storage_dir": self.tmpdir})
        self.graph_store.initialize()

        # Create units with known embeddings
        self.u1 = _make_unit("u1", task_id="t1",
            content={"topic": "web search", "principle": "use keywords"},
            embedding=np.array([1, 0, 0] * 128, dtype=np.float32))
        self.u2 = _make_unit("u2", task_id="t1",
            content={"topic": "crawl pages", "principle": "follow links"},
            embedding=np.array([0, 1, 0] * 128, dtype=np.float32))
        self.u2.relations = [MemoryRelation("u1", RelationType.COOCCURS)]
        self.u3 = _make_unit("u3", task_id="t2",
            content={"topic": "parse PDF", "principle": "extract text"},
            embedding=np.array([0, 0, 1] * 128, dtype=np.float32))

        self.graph_store.add([self.u1, self.u2, self.u3])

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_seed_and_expand(self):
        retriever = GraphRetriever(self.graph_store, config={
            "seed_k": 1, "max_hops": 1, "decay_factor": 0.7,
        })
        ctx = QueryContext(
            query="web search",
            embedding=np.array([1, 0, 0] * 128, dtype=np.float32),
        )
        pack = retriever.retrieve(ctx, top_k=5)
        self.assertFalse(pack.is_empty())

        # u1 should be seed (highest similarity)
        ids = [su.unit.id for su in pack.scored_units]
        self.assertIn("u1", ids)

        # u2 should be found via graph expansion (COOCCURS with u1, or same query node)
        self.assertIn("u2", ids)

        # Trace should have seed + expand steps
        methods = [t.method for t in pack.trace]
        self.assertIn("graph_seed", methods)
        self.assertIn("graph_expand", methods)

    def test_no_graph_fallback(self):
        """Non-graph store should still work (seed only)."""
        import os
        json_store = JsonStorage({"db_path": os.path.join(self.tmpdir, "json", "db.json")})
        json_store.initialize()
        json_store.add([self.u1, self.u2])

        retriever = GraphRetriever(json_store)
        ctx = QueryContext(
            query="test",
            embedding=np.array([1, 0, 0] * 128, dtype=np.float32),
        )
        pack = retriever.retrieve(ctx, top_k=2)
        self.assertFalse(pack.is_empty())
        # Should only have seed results (no graph expansion)
        methods = [t.method for t in pack.trace]
        self.assertIn("graph_seed", methods)
        self.assertNotIn("graph_expand", methods)


class TestContrastiveRetriever(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Success units
        self.u_success1 = _make_unit("s1", outcome="success",
            content={"topic": "web search", "principle": "use keywords"},
            embedding=np.array([1, 0, 0] * 128, dtype=np.float32))
        self.u_success2 = _make_unit("s2", outcome="success",
            content={"topic": "crawl pages", "principle": "follow links"},
            embedding=np.array([0.9, 0.1, 0] * 128, dtype=np.float32))
        # Failure units
        self.u_fail1 = _make_unit("f1", outcome="failure",
            utype=MemoryUnitType.INSIGHT,
            content={"root_cause_conclusion": "wrong tool", "state_mismatch_analysis": "expected A got B"},
            embedding=np.array([0.8, 0.2, 0] * 128, dtype=np.float32))
        self.u_fail2 = _make_unit("f2", outcome="failure",
            content={"topic": "avoid timeout", "principle": "set limits"},
            embedding=np.array([0, 1, 0] * 128, dtype=np.float32))

        self.store = _make_store_with_units(
            [self.u_success1, self.u_success2, self.u_fail1, self.u_fail2],
            self.tmpdir,
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_contrastive_retrieval(self):
        retriever = ContrastiveRetriever(self.store, config={
            "success_weight": 0.6,
            "failure_weight": 0.4,
            "success_k": 2,
            "failure_k": 2,
        })
        ctx = QueryContext(
            query="web search",
            embedding=np.array([1, 0, 0] * 128, dtype=np.float32),
        )
        pack = retriever.retrieve(ctx, top_k=4)
        self.assertFalse(pack.is_empty())

        # Should have both success and failure methods
        methods = {su.method for su in pack.scored_units}
        self.assertTrue(
            methods & {"contrastive_success", "contrastive_failure"},
            f"Expected both pools, got {methods}"
        )

        # Trace should have 3 entries (success, failure, merge)
        self.assertEqual(len(pack.trace), 3)

    def test_empty_failure_pool(self):
        """All success units — failure pool is empty."""
        import os
        store = JsonStorage({"db_path": os.path.join(self.tmpdir, "all_success", "db.json")})
        store.initialize()
        store.add([self.u_success1, self.u_success2])

        retriever = ContrastiveRetriever(store)
        ctx = QueryContext(
            query="web search",
            embedding=np.array([1, 0, 0] * 128, dtype=np.float32),
        )
        pack = retriever.retrieve(ctx, top_k=5)
        self.assertFalse(pack.is_empty())
        # All results should be from success pool
        for su in pack.scored_units:
            self.assertEqual(su.unit.task_outcome, "success")


class TestEndToEnd(unittest.TestCase):
    """Integration test: store → retriever → MemoryPack → prompt string."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_full_pipeline(self):
        import os
        store = JsonStorage({"db_path": os.path.join(self.tmpdir, "e2e.json")})
        store.initialize()

        units = [
            _make_unit("u1", content={"topic": "web search", "principle": "use Google", "micro_example": "search 'site:...'"},
                       embedding=np.array([1, 0, 0] * 128, dtype=np.float32)),
            _make_unit("u2", content={"topic": "file download", "principle": "check MIME type", "micro_example": "verify ext"},
                       embedding=np.array([0, 1, 0] * 128, dtype=np.float32)),
            _make_unit("u3", utype=MemoryUnitType.INSIGHT, outcome="failure",
                       content={"root_cause_conclusion": "wrong format", "state_mismatch_analysis": "expected JSON got XML"},
                       embedding=np.array([0.5, 0.5, 0] * 128, dtype=np.float32)),
        ]
        store.add(units)

        # Use semantic retriever
        retriever = SemanticRetriever(store)
        ctx = QueryContext(
            query="how to search web effectively",
            embedding=np.array([0.9, 0.1, 0] * 128, dtype=np.float32),
        )
        pack = retriever.retrieve(ctx, top_k=3)

        # Verify pack
        self.assertFalse(pack.is_empty())
        self.assertEqual(pack.retriever_name, "SemanticRetriever")

        # Generate prompt string
        prompt = pack.to_prompt_string()
        self.assertIn("----Memory System Guidance----", prompt)
        self.assertIn("----End Memory----", prompt)
        self.assertIn("[TIP]", prompt)

        # Verify serialization
        d = pack.to_dict()
        self.assertIn("trace", d)
        self.assertIn("evidence", d)

    def test_hybrid_e2e(self):
        import os
        store = JsonStorage({"db_path": os.path.join(self.tmpdir, "hybrid_e2e.json")})
        store.initialize()

        units = [
            _make_unit("u1", content={"topic": "web search tips", "principle": "use operators"},
                       embedding=np.array([1, 0, 0] * 128, dtype=np.float32)),
            _make_unit("u2", content={"topic": "PDF parsing guide", "principle": "use pdfplumber"},
                       embedding=np.array([0, 1, 0] * 128, dtype=np.float32)),
        ]
        store.add(units)

        semantic = SemanticRetriever(store)
        keyword = KeywordRetriever(store)
        hybrid = HybridRetriever(
            store,
            sub_retrievers=[semantic, keyword],
            config={"weights": {"SemanticRetriever": 0.7, "KeywordRetriever": 0.3}},
        )

        ctx = QueryContext(
            query="web search",
            embedding=np.array([0.9, 0.1, 0] * 128, dtype=np.float32),
        )
        pack = hybrid.retrieve(ctx, top_k=2)

        prompt = pack.to_prompt_string()
        self.assertIn("----Memory System Guidance----", prompt)
        self.assertTrue(len(pack.scored_units) > 0)


if __name__ == "__main__":
    unittest.main()
