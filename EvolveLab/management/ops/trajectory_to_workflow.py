"""
TrajectoryToWorkflowOp — Promote high-performing trajectories into
generalized workflow memories via LLM abstraction.

Part of the 'episodic_consolidation' operation group.
"""

import time
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from ..base_op import BaseManageOp, OpResult, StorageCompatibility, TriggerType
from ...memory_schema import MemoryUnit, MemoryUnitType, MemoryRelation, RelationType

logger = logging.getLogger(__name__)


class TrajectoryToWorkflowOp(BaseManageOp):
    """
    Identify high-usage, high-success-rate TRAJECTORY units and ask an LLM
    to generalize each into a reusable WORKFLOW memory.  The original
    trajectory is decayed but preserved with a DEPENDS relation.
    """

    op_name = "trajectory_to_workflow"
    op_group = "episodic_consolidation"
    trigger_type = TriggerType.PERIODIC
    storage_compatibility = StorageCompatibility.GRAPH_ENHANCED
    requires_llm = True
    requires_embedding = False
    rl_action_id = 1

    _DEFAULT_CONFIG = {
        "min_usage": 3,
        "min_success_rate": 0.7,
    }

    def execute(self, context: Dict[str, Any]) -> OpResult:
        t0 = time.time()
        result = OpResult(op_name=self.op_name, triggered=True)

        try:
            min_usage = self.config.get(
                "min_usage", self._DEFAULT_CONFIG["min_usage"]
            )
            min_success_rate = self.config.get(
                "min_success_rate", self._DEFAULT_CONFIG["min_success_rate"]
            )

            # Step 1: Find qualifying TRAJECTORY units
            all_units: List[MemoryUnit] = self.store.get_all()
            candidates = [
                u for u in all_units
                if u.is_active
                and u.type == MemoryUnitType.TRAJECTORY
                and u.usage_count >= min_usage
                and u.success_rate >= min_success_rate
            ]

            if not candidates:
                logger.info("trajectory_to_workflow: no qualifying trajectories found")
                result.triggered = False
                result.duration_ms = (time.time() - t0) * 1000
                return result

            units_created = 0
            units_modified = 0

            for traj in candidates:
                # Check if a workflow already depends on this trajectory
                already_promoted = any(
                    r.relation_type == RelationType.DEPENDS
                    for r in traj.relations
                    if any(
                        wu.type == MemoryUnitType.WORKFLOW
                        for wu in all_units
                        if wu.id == r.target_id
                    )
                )
                # Also check if any existing workflow has DEPENDS on this traj
                already_promoted = already_promoted or any(
                    any(
                        r.target_id == traj.id
                        and r.relation_type == RelationType.DEPENDS
                        for r in wu.relations
                    )
                    for wu in all_units
                    if wu.type == MemoryUnitType.WORKFLOW and wu.is_active
                )
                if already_promoted:
                    continue

                # Step 2: LLM call to generalize trajectory into workflow
                traj_text = traj.content_text()[:2000]
                prompt = (
                    "You are a workflow extraction assistant. Given the following "
                    "task trajectory (a sequence of actions and observations), "
                    "generalize it into a reusable workflow template.\n\n"
                    f"Task query: {traj.source_task_query}\n"
                    f"Trajectory:\n{traj_text}\n\n"
                    "Output a generalized workflow as a numbered list of steps. "
                    "Each step should include: action, rationale, and generalized "
                    "execution guidance. Output ONLY the workflow, no preamble."
                )

                messages = [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ]
                response = self.llm_client(messages)
                response_text = (
                    response.content
                    if hasattr(response, "content")
                    else str(response)
                )

                # Step 3: Create new WORKFLOW MemoryUnit
                workflow_unit = MemoryUnit(
                    id=str(uuid.uuid4()),
                    type=MemoryUnitType.WORKFLOW,
                    content={
                        "agent_workflow": response_text.strip(),
                        "source_trajectory_id": traj.id,
                        "source_task_query": traj.source_task_query,
                    },
                    source_task_id=traj.source_task_id,
                    source_task_query=traj.source_task_query,
                    task_outcome=traj.task_outcome,
                    confidence=traj.confidence,
                    usage_count=0,
                    success_count=0,
                    decay_weight=1.0,
                    is_active=True,
                )
                workflow_unit.compute_signature()
                workflow_unit.token_estimate()

                # Compute embedding if embedding_model is available
                if self.embedding_model is not None:
                    try:
                        workflow_unit.embedding = self.embedding_model.encode(
                            workflow_unit.content_text()
                        )
                    except Exception as e:
                        logger.warning(
                            "trajectory_to_workflow: embedding failed: %s", e
                        )

                # Step 4: Add DEPENDS relation (workflow -> trajectory)
                workflow_unit.relations.append(
                    MemoryRelation(
                        target_id=traj.id,
                        relation_type=RelationType.DEPENDS,
                        weight=1.0,
                    )
                )

                self.store.add([workflow_unit])
                units_created += 1

                # Step 6: Decay original trajectory
                traj.decay_weight *= 0.5
                self.store.update(traj)
                units_modified += 1

                # Step 5: Graph-enhanced logic
                if self._is_graph_store():
                    try:
                        graph = self.store._graph
                        wf_nid = self.store._content_nid(workflow_unit.id)
                        traj_nid = self.store._content_nid(traj.id)

                        # Add DEPENDS edge
                        if not self.store._has_edge(wf_nid, traj_nid, "DEPENDS"):
                            graph.add_edge(
                                wf_nid,
                                traj_nid,
                                key="DEPENDS",
                                edge_type="DEPENDS",
                                weight=1.0,
                            )

                        # Transfer entity associations from trajectory to workflow
                        if graph.has_node(traj_nid):
                            for _, target, data in list(
                                graph.edges(traj_nid, data=True)
                            ):
                                if data.get("edge_type") == "HAS_ENTITY":
                                    if not self.store._has_edge(
                                        wf_nid, target, "HAS_ENTITY"
                                    ):
                                        graph.add_edge(
                                            wf_nid,
                                            target,
                                            key="HAS_ENTITY",
                                            edge_type="HAS_ENTITY",
                                            weight=data.get("weight", 1.0),
                                        )
                    except Exception as e:
                        logger.warning(
                            "trajectory_to_workflow: graph enhancement failed: %s", e
                        )

                logger.info(
                    "trajectory_to_workflow: promoted trajectory %s -> workflow %s",
                    traj.id[:8], workflow_unit.id[:8],
                )

            result.units_created = units_created
            result.units_modified = units_modified
            result.units_affected = units_created + units_modified
            result.details = {
                "total_trajectories": len(all_units),
                "qualifying_trajectories": len(candidates),
                "workflows_created": units_created,
            }

        except Exception as e:
            logger.error(
                "trajectory_to_workflow: execution failed: %s", e, exc_info=True
            )
            result.details["error"] = str(e)

        result.duration_ms = (time.time() - t0) * 1000
        logger.info(
            "trajectory_to_workflow: completed in %.1fms", result.duration_ms
        )
        return result
