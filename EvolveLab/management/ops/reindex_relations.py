"""
ReindexRelationsOp — Recompute inter-unit similarity relations and
co-occurrence edges based on embeddings and task provenance.

Part of the 'episodic_consolidation' operation group.
"""

import time
import logging
from typing import Any, Dict, List

import numpy as np

from ..base_op import BaseManageOp, OpResult, StorageCompatibility, TriggerType
from ...memory_schema import MemoryUnit, MemoryUnitType, MemoryRelation, RelationType

logger = logging.getLogger(__name__)


class ReindexRelationsOp(BaseManageOp):
    """
    Reindex similarity and co-occurrence relations between memory units.

    Graph stores (native): iterate content node pairs, compute cosine
    similarity, and upsert SIMILAR / COOCCURS edges directly in the graph.

    Non-graph stores (weak): compute pairwise embedding similarity,
    update each unit's relations list, and persist via store.update().
    """

    op_name = "reindex_relations"
    op_group = "episodic_consolidation"
    trigger_type = TriggerType.PERIODIC
    storage_compatibility = StorageCompatibility.GRAPH_ENHANCED
    requires_llm = False
    requires_embedding = True
    rl_action_id = 3

    _DEFAULT_CONFIG = {
        "similarity_threshold": 0.7,
    }

    def execute(self, context: Dict[str, Any]) -> OpResult:
        t0 = time.time()
        result = OpResult(op_name=self.op_name, triggered=True)

        try:
            sim_threshold = self.config.get(
                "similarity_threshold",
                self._DEFAULT_CONFIG["similarity_threshold"],
            )

            all_units: List[MemoryUnit] = self.store.get_all()
            active_units = [u for u in all_units if u.is_active]

            if len(active_units) < 2:
                logger.info("reindex_relations: fewer than 2 active units, skipping")
                result.triggered = False
                result.duration_ms = (time.time() - t0) * 1000
                return result

            if self._is_graph_store():
                edges_added, edges_updated = self._reindex_graph(
                    active_units, sim_threshold
                )
                result.units_affected = edges_added + edges_updated
                result.details = {
                    "mode": "graph",
                    "edges_added": edges_added,
                    "edges_updated": edges_updated,
                    "active_units": len(active_units),
                }
            else:
                units_modified = self._reindex_non_graph(
                    active_units, sim_threshold
                )
                result.units_modified = units_modified
                result.units_affected = units_modified
                result.details = {
                    "mode": "non_graph",
                    "units_modified": units_modified,
                    "active_units": len(active_units),
                }

        except Exception as e:
            logger.error(
                "reindex_relations: execution failed: %s", e, exc_info=True
            )
            result.details["error"] = str(e)

        result.duration_ms = (time.time() - t0) * 1000
        logger.info("reindex_relations: completed in %.1fms", result.duration_ms)
        return result

    # ------------------------------------------------------------------
    # Graph-native reindexing
    # ------------------------------------------------------------------

    def _reindex_graph(
        self, units: List[MemoryUnit], sim_threshold: float
    ) -> tuple:
        """Reindex SIMILAR and COOCCURS edges in the graph store."""
        graph = self.store._graph
        edges_added = 0
        edges_updated = 0

        # Filter to units with embeddings
        with_emb = [u for u in units if u.embedding is not None]

        if len(with_emb) >= 2:
            embeddings = np.array([u.embedding for u in with_emb])
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
            normed = embeddings / norms
            sim_matrix = normed @ normed.T

            n = len(with_emb)
            for i in range(n):
                for j in range(i + 1, n):
                    u_i = with_emb[i]
                    u_j = with_emb[j]
                    sim = float(sim_matrix[i, j])

                    nid_i = self.store._content_nid(u_i.id)
                    nid_j = self.store._content_nid(u_j.id)

                    # SIMILAR edges
                    if sim >= sim_threshold:
                        if self.store._has_edge(nid_i, nid_j, "SIMILAR"):
                            # Update weight
                            for _, _, _, data in graph.edges(
                                nid_i, data=True, keys=True
                            ):
                                pass
                            # Simpler: remove and re-add
                            try:
                                graph.remove_edge(nid_i, nid_j, key="SIMILAR")
                            except Exception:
                                pass
                            graph.add_edge(
                                nid_i, nid_j,
                                key="SIMILAR",
                                edge_type="SIMILAR",
                                weight=sim,
                            )
                            edges_updated += 1
                        else:
                            graph.add_edge(
                                nid_i, nid_j,
                                key="SIMILAR",
                                edge_type="SIMILAR",
                                weight=sim,
                            )
                            edges_added += 1

                    # COOCCURS edges for units from the same task
                    if (
                        u_i.source_task_id
                        and u_i.source_task_id == u_j.source_task_id
                        and not self.store._has_edge(nid_i, nid_j, "COOCCURS")
                    ):
                        graph.add_edge(
                            nid_i, nid_j,
                            key="COOCCURS",
                            edge_type="COOCCURS",
                            weight=1.0,
                        )
                        edges_added += 1

        # Also handle COOCCURS for units without embeddings
        task_groups: Dict[str, List[MemoryUnit]] = {}
        for u in units:
            if u.source_task_id:
                task_groups.setdefault(u.source_task_id, []).append(u)

        for task_id, group in task_groups.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    nid_i = self.store._content_nid(group[i].id)
                    nid_j = self.store._content_nid(group[j].id)
                    if (
                        graph.has_node(nid_i)
                        and graph.has_node(nid_j)
                        and not self.store._has_edge(nid_i, nid_j, "COOCCURS")
                    ):
                        graph.add_edge(
                            nid_i, nid_j,
                            key="COOCCURS",
                            edge_type="COOCCURS",
                            weight=1.0,
                        )
                        edges_added += 1

        logger.info(
            "reindex_relations (graph): added=%d, updated=%d",
            edges_added, edges_updated,
        )
        return edges_added, edges_updated

    # ------------------------------------------------------------------
    # Non-graph (weak) reindexing
    # ------------------------------------------------------------------

    def _reindex_non_graph(
        self, units: List[MemoryUnit], sim_threshold: float
    ) -> int:
        """Reindex relations via unit.relations lists and store.update()."""
        # Filter to units with embeddings
        with_emb = [u for u in units if u.embedding is not None]
        units_modified = 0

        if len(with_emb) < 2:
            return 0

        embeddings = np.array([u.embedding for u in with_emb])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        normed = embeddings / norms
        sim_matrix = normed @ normed.T

        # Build a map of existing relation targets per unit for fast lookup
        unit_id_to_idx = {u.id: idx for idx, u in enumerate(with_emb)}

        n = len(with_emb)
        modified_ids = set()

        for i in range(n):
            u_i = with_emb[i]
            existing_similar = {
                r.target_id
                for r in u_i.relations
                if r.relation_type == RelationType.SIMILAR
            }
            existing_cooccurs = {
                r.target_id
                for r in u_i.relations
                if r.relation_type == RelationType.COOCCURS
            }
            changed = False

            for j in range(n):
                if i == j:
                    continue
                u_j = with_emb[j]
                sim = float(sim_matrix[i, j])

                # Add SIMILAR relation if above threshold
                if sim >= sim_threshold and u_j.id not in existing_similar:
                    u_i.relations.append(
                        MemoryRelation(
                            target_id=u_j.id,
                            relation_type=RelationType.SIMILAR,
                            weight=sim,
                        )
                    )
                    changed = True

                # Add COOCCURS relation for same-task units
                if (
                    u_i.source_task_id
                    and u_i.source_task_id == u_j.source_task_id
                    and u_j.id not in existing_cooccurs
                ):
                    u_i.relations.append(
                        MemoryRelation(
                            target_id=u_j.id,
                            relation_type=RelationType.COOCCURS,
                            weight=1.0,
                        )
                    )
                    changed = True

            if changed:
                self.store.update(u_i)
                modified_ids.add(u_i.id)

        units_modified = len(modified_ids)
        logger.info(
            "reindex_relations (non-graph): %d units modified", units_modified
        )
        return units_modified
