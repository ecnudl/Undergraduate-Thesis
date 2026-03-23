"""
Graph Retriever — Vector seed + graph neighbor expansion.

Two-phase retrieval:
  1. Seed: Use semantic similarity to find initial relevant units
  2. Expand: Walk the graph (1-hop or multi-hop) from seed nodes,
     propagating scores with decay along edges

Requires a GraphStore backend to access the graph structure.
Falls back to pure semantic retrieval if the store lacks graph methods.

Inspired by cerebra_fusion's _graph_expand() pattern:
  propagated_score = base_score * edge_weight * decay_factor
"""

from typing import Any, Dict, List, Optional, Set

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from EvolveLab.memory_schema import MemoryUnit
from EvolveLab.retrieval.base_retriever import (
    BaseRetriever,
    MemoryPack,
    QueryContext,
    ScoredUnit,
    TraceEntry,
)


class GraphRetriever(BaseRetriever):
    """
    Semantic seed + graph neighbor expansion retriever.

    Config options:
        seed_k (int): Number of seed nodes from semantic search. Default 3.
        max_hops (int): Maximum graph expansion hops. Default 1.
        decay_factor (float): Score decay per hop. Default 0.7.
        min_score (float): Minimum score threshold. Default 0.1.
        active_only (bool): Only consider active units. Default True.
        expand_edge_types (List[str]): Edge types to follow.
            Default: ["SIMILAR", "COOCCURS", "REINFORCES", "HAS_MEMORY"]
    """

    def __init__(self, store, embedding_model=None, config: Optional[Dict[str, Any]] = None):
        super().__init__(store, config)
        self.embedding_model = embedding_model
        self.seed_k = self.config.get("seed_k", 3)
        self.max_hops = self.config.get("max_hops", 1)
        self.decay_factor = self.config.get("decay_factor", 0.7)
        self.min_score = self.config.get("min_score", 0.1)
        self.active_only = self.config.get("active_only", True)
        self.expand_edge_types = self.config.get("expand_edge_types", [
            "SIMILAR", "COOCCURS", "REINFORCES", "HAS_MEMORY",
        ])

    def retrieve(self, ctx: QueryContext, top_k: int = 5) -> MemoryPack:
        trace: List[TraceEntry] = []
        has_graph = hasattr(self.store, 'neighbors') and hasattr(self.store, 'get')

        # Phase 1: Semantic seed
        query_emb = ctx.embedding
        if query_emb is None and self.embedding_model is not None:
            query_emb = self.embedding_model.encode(
                ctx.query, convert_to_numpy=True
            )

        emb_matrix, emb_units = self.store.get_embedding_index(
            active_only=self.active_only
        )

        seed_scored: List[ScoredUnit] = []
        if query_emb is not None and emb_matrix is not None and len(emb_units) > 0:
            query_emb = query_emb.reshape(1, -1)
            sims = cosine_similarity(query_emb, emb_matrix)[0]
            seed_count = min(self.seed_k, len(emb_units))
            top_indices = sims.argsort()[-seed_count:][::-1]

            for idx in top_indices:
                score = float(sims[idx])
                if score >= self.min_score:
                    seed_scored.append(ScoredUnit(
                        unit=emb_units[idx],
                        score=score,
                        method="graph_seed",
                    ))

        trace.append(TraceEntry(
            step=1,
            method="graph_seed",
            candidates=len(emb_units) if emb_matrix is not None else 0,
            selected=len(seed_scored),
            params={"seed_k": self.seed_k},
        ))

        if not has_graph or not seed_scored:
            # No graph available or no seeds — return seed results only
            for su in seed_scored:
                su.unit.record_access()
            return self._make_pack(ctx, seed_scored[:top_k], trace)

        # Phase 2: Graph expansion
        score_board: Dict[str, float] = {}
        unit_map: Dict[str, MemoryUnit] = {}

        # Initialize with seed scores
        for su in seed_scored:
            score_board[su.unit.id] = su.score
            unit_map[su.unit.id] = su.unit

        visited: Set[str] = set()
        frontier = [(su.unit.id, su.score) for su in seed_scored]
        total_expanded = 0

        for hop in range(self.max_hops):
            next_frontier = []
            for unit_id, base_score in frontier:
                if unit_id in visited:
                    continue
                visited.add(unit_id)

                node_id = f"m:{unit_id}"
                for edge_type in self.expand_edge_types:
                    neighbors = self.store.neighbors(
                        node_id, edge_type=edge_type, direction="both"
                    )
                    for nbr_id in neighbors:
                        if not nbr_id.startswith("m:"):
                            continue
                        nbr_unit_id = nbr_id[2:]
                        if nbr_unit_id in visited:
                            continue

                        propagated = base_score * self.decay_factor
                        if propagated < self.min_score:
                            continue

                        # Get the neighbor unit
                        nbr_unit = self.store.get(nbr_unit_id)
                        if nbr_unit is None:
                            continue
                        if self.active_only and not nbr_unit.is_active:
                            continue

                        # Accumulate score
                        old_score = score_board.get(nbr_unit_id, 0.0)
                        new_score = max(old_score, propagated)
                        score_board[nbr_unit_id] = new_score
                        unit_map[nbr_unit_id] = nbr_unit
                        next_frontier.append((nbr_unit_id, new_score))
                        total_expanded += 1

            frontier = next_frontier

        trace.append(TraceEntry(
            step=2,
            method="graph_expand",
            candidates=total_expanded,
            selected=len(score_board),
            params={
                "max_hops": self.max_hops,
                "decay_factor": self.decay_factor,
                "edge_types": self.expand_edge_types,
            },
        ))

        # Rank all candidates and select top-k
        ranked = sorted(score_board.items(), key=lambda x: x[1], reverse=True)
        ranked = ranked[:top_k]

        scored = [
            ScoredUnit(
                unit=unit_map[uid],
                score=score,
                method="graph" if uid not in {su.unit.id for su in seed_scored} else "graph_seed",
            )
            for uid, score in ranked
        ]

        for su in scored:
            su.unit.record_access()

        return self._make_pack(ctx, scored, trace)
