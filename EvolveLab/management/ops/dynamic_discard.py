"""
DynamicDiscardOp — Periodically deactivate memories with persistently
high failure rates.

Part of the 'failure_adjustment' operation group.
"""

import time
import logging
from typing import Any, Dict, List

from ..base_op import BaseManageOp, OpResult, StorageCompatibility, TriggerType
from ...memory_schema import MemoryUnit, MemoryUnitType, MemoryRelation, RelationType

logger = logging.getLogger(__name__)


class DynamicDiscardOp(BaseManageOp):
    """
    Periodically scan all active memory units and soft-delete those whose
    failure rate exceeds a threshold (after a minimum number of usages).

    For graph stores the node is kept in the graph but marked inactive,
    preserving edges for audit trails.  For Vector/Hybrid stores the FAISS
    index is rebuilt after the batch to remove discarded embeddings.
    """

    op_name = "dynamic_discard"
    op_group = "failure_adjustment"
    trigger_type = TriggerType.PERIODIC
    storage_compatibility = StorageCompatibility.GRAPH_ENHANCED
    requires_llm = False
    requires_embedding = False
    rl_action_id = 11

    _DEFAULT_CONFIG = {
        "min_usage": 5,
        "max_failure_rate": 0.8,
    }

    def execute(self, context: Dict[str, Any]) -> OpResult:
        t0 = time.time()
        result = OpResult(op_name=self.op_name, triggered=True)

        try:
            min_usage = self.config.get(
                "min_usage",
                self._DEFAULT_CONFIG["min_usage"],
            )
            max_failure_rate = self.config.get(
                "max_failure_rate",
                self._DEFAULT_CONFIG["max_failure_rate"],
            )
            # success_rate threshold: units with success_rate < (1 - max_failure_rate)
            success_threshold = 1.0 - max_failure_rate

            # Get all active units
            all_units: List[MemoryUnit] = self.store.get_all(active_only=True)

            # Filter candidates: enough usage and high failure rate
            candidates = [
                u for u in all_units
                if u.usage_count >= min_usage
                and u.success_rate < success_threshold
            ]

            if not candidates:
                logger.info(
                    "dynamic_discard: no units meet discard criteria "
                    "(min_usage=%d, max_failure_rate=%.2f)",
                    min_usage, max_failure_rate,
                )
                result.triggered = False
                result.duration_ms = (time.time() - t0) * 1000
                return result

            units_deleted = 0

            for unit in candidates:
                # Soft-delete: mark as inactive
                unit.is_active = False
                self.store.update(unit)
                units_deleted += 1

                # Graph enhanced: keep node but mark inactive, preserve edges
                if self._is_graph_store():
                    try:
                        nid = self.store._content_nid(unit.id)
                        graph = self.store._graph
                        if graph.has_node(nid):
                            graph.nodes[nid]["is_active"] = False
                    except Exception as e:
                        logger.warning(
                            "dynamic_discard: graph node update failed for %s: %s",
                            unit.id[:8], e,
                        )

                logger.debug(
                    "dynamic_discard: deactivated unit %s "
                    "(usage=%d, success_rate=%.2f)",
                    unit.id[:8], unit.usage_count, unit.success_rate,
                )

            # Rebuild FAISS index after batch for Vector/Hybrid stores
            if units_deleted > 0:
                self._rebuild_faiss_if_needed()

            result.units_deleted = units_deleted
            result.units_affected = units_deleted
            result.details = {
                "total_active_scanned": len(all_units),
                "candidates_found": len(candidates),
                "min_usage": min_usage,
                "max_failure_rate": max_failure_rate,
                "success_threshold": success_threshold,
            }

        except Exception as e:
            logger.error(
                "dynamic_discard: execution failed: %s", e, exc_info=True
            )
            result.details["error"] = str(e)

        result.duration_ms = (time.time() - t0) * 1000
        logger.info(
            "dynamic_discard: completed in %.1fms, deactivated %d units",
            result.duration_ms, result.units_deleted,
        )
        return result
