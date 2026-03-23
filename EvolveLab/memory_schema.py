"""
Unified Memory Unit Schema for RL-based Memory Management.

Design principles:
  1. Strict envelope (identity, lifecycle, quality, cost, scope) + flexible content (Dict)
  2. All RL-observable/controllable fields are explicit, numeric, top-level
  3. Downcastable to existing MemoryItem for retrieval compatibility
  4. Atomic granularity: one MemoryUnit = one independently manageable piece of knowledge
     - Insight  → one per failed task   (already atomic from extraction)
     - Tip      → one per individual tip (split from extraction batch)
     - Trajectory → one per task         (steps are a coherent sequence)
     - Workflow → one per task           (steps are a coherent sequence)
     - Shortcut → one per individual macro (split from extraction batch)

Usage:
  from EvolveLab.memory_schema import MemoryUnit, MemoryUnitType, RelationType

  unit = MemoryUnit(
      type=MemoryUnitType.TIP,
      content={"topic": "...", "principle": "...", ...},
      source_task_id="04a04a9b-...",
      source_task_query="If we assume all articles...",
      task_outcome="success",
      extraction_model="qwen3-max",
  )
  unit.compute_signature()

  # Persist
  d = unit.to_dict()
  unit2 = MemoryUnit.from_dict(d)

  # Feed to RL policy network
  state_vec = unit.to_rl_state()  # np.ndarray, shape (16,)

  # Pass to retrieval layer
  item = unit.to_memory_item(score=0.87)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import hashlib
import json
import uuid

import numpy as np

from EvolveLab.memory_types import MemoryItem, MemoryItemType


# ============================================================
# Enums
# ============================================================

class MemoryUnitType(Enum):
    """Atomic memory unit types, one per extraction prompt module."""
    INSIGHT    = "insight"      # Failure diagnostics    (failure-only)
    TIP        = "tip"          # Cognitive heuristics   (success + failure)
    TRAJECTORY = "trajectory"   # Compressed action-observation chain
    WORKFLOW   = "workflow"     # Orchestration logic    (success-only)
    SHORTCUT   = "shortcut"     # Executable macros      (success + failure)


class RelationType(Enum):
    """Edge types between memory units."""
    SIMILAR    = "similar"      # Semantically related content
    DEPENDS    = "depends"      # A requires knowledge from B
    CONFLICTS  = "conflicts"    # A contradicts B
    SUPERSEDES = "supersedes"   # A replaces B (newer / higher confidence)
    COOCCURS   = "cooccurs"     # Extracted from same task
    REINFORCES = "reinforces"   # A supports / strengthens B


# ============================================================
# Relation (lightweight edge, stored on the unit itself)
# ============================================================

@dataclass
class MemoryRelation:
    """Directed edge from the owning MemoryUnit to target_id."""
    target_id: str
    relation_type: RelationType
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "weight": self.weight,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MemoryRelation":
        return MemoryRelation(
            target_id=d["target_id"],
            relation_type=RelationType(d["relation_type"]),
            weight=d.get("weight", 1.0),
        )


# ============================================================
# Core Memory Unit
# ============================================================

@dataclass
class MemoryUnit:
    """
    Universal memory unit for RL-based memory management.

    Field groups
    ────────────
    Identity      id, type, signature
    Content       content  (Dict — type-specific, kept flexible)
    Source        source_task_id, source_task_query, task_outcome, extraction_model
    Quality       confidence, usage_count, success_count   ← RL-observable & adjustable
    Lifecycle     created_at, last_accessed, access_count,
                  decay_weight, is_active                  ← RL-observable & adjustable
    Cost          storage_tokens
    Scope         applicable_domains, applicable_task_types
    Relations     relations  (List[MemoryRelation])
    Embedding     embedding  (np.ndarray, 384-d by default)

    Content schemas per type (extraction prompt output → atomic unit)
    ─────────────────────────────────────────────────────────────────
    INSIGHT:
        root_cause_conclusion: str
        state_mismatch_analysis: str          # "Expected: X; Actual: Y"
        divergence_point: str                 # "[Step N - Action]: ..."
        knowledge_graph: List[List[str]]      # [[S, P, O], ...]

    TIP:
        category: str                         # "planning_and_decision" | "tool_and_search"
        topic: str
        principle: str
        micro_example: str
        counterfactual: str

    TRAJECTORY:
        steps: List[Dict]                     # [{step_id, action, observation}, ...]
        task_outcome: str
        failure_reason: Optional[str]

    WORKFLOW:
        agent_workflow: List[Dict]            # [{step, action, rationale, generalized_execution}]
        search_workflow: List[Dict]           # [{step, query_formulation, validation_criteria}]

    SHORTCUT:
        name: str
        description: str
        precondition: str
        extraction_type: str
        assumptions: List[str]
    """

    # === Identity ==========================================================
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MemoryUnitType = MemoryUnitType.TIP
    signature: str = ""

    # === Content (type-specific, flexible Dict) ============================
    content: Dict[str, Any] = field(default_factory=dict)

    # === Source / Provenance ================================================
    source_task_id: str = ""
    source_task_query: str = ""
    task_outcome: str = ""                    # "success" | "failure"
    extraction_model: str = ""

    # === Quality (RL-observable & adjustable) ===============================
    confidence: float = 1.0                   # [0, 1]  — RL action: adjust
    usage_count: int = 0                      # Retrieved & presented to agent
    success_count: int = 0                    # Task succeeded after using this

    # === Lifecycle (RL-observable & adjustable) =============================
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: Optional[str] = None
    access_count: int = 0
    decay_weight: float = 1.0                 # [0, 1]  — RL action: decay speed
    is_active: bool = True                    # False   — RL action: forget (soft)

    # === Cost ==============================================================
    storage_tokens: int = 0

    # === Scope =============================================================
    applicable_domains: List[str] = field(default_factory=list)
    applicable_task_types: List[str] = field(default_factory=list)

    # === Relations =========================================================
    relations: List[MemoryRelation] = field(default_factory=list)

    # === Embedding =========================================================
    embedding: Optional[np.ndarray] = None

    # -----------------------------------------------------------------------
    # Derived properties
    # -----------------------------------------------------------------------

    @property
    def success_rate(self) -> float:
        """Success rate when this memory was used. Returns 0 if never used."""
        return self.success_count / self.usage_count if self.usage_count > 0 else 0.0

    @property
    def age_hours(self) -> float:
        """Hours since creation."""
        try:
            created = datetime.fromisoformat(self.created_at)
            return (datetime.now() - created).total_seconds() / 3600.0
        except (ValueError, TypeError):
            return 0.0

    @property
    def hours_since_access(self) -> float:
        """Hours since last retrieval. Returns age_hours if never accessed."""
        if self.last_accessed is None:
            return self.age_hours
        try:
            accessed = datetime.fromisoformat(self.last_accessed)
            return (datetime.now() - accessed).total_seconds() / 3600.0
        except (ValueError, TypeError):
            return self.age_hours

    @property
    def effective_score(self) -> float:
        """Composite score combining confidence, success rate, and recency decay."""
        recency = 1.0 / (1.0 + self.hours_since_access / 168.0)  # 168h = 1 week half-life
        return self.confidence * (0.5 + 0.5 * self.success_rate) * (self.decay_weight * recency)

    # -----------------------------------------------------------------------
    # Mutations (called by retrieval / management layers)
    # -----------------------------------------------------------------------

    def record_access(self) -> None:
        """Update access metadata. Called each time this memory is retrieved."""
        self.access_count += 1
        self.last_accessed = datetime.now().isoformat()

    def record_outcome(self, task_succeeded: bool) -> None:
        """Update usage/success counts after a task completes."""
        self.usage_count += 1
        if task_succeeded:
            self.success_count += 1

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def compute_signature(self) -> str:
        """Compute content hash for deduplication. Sets and returns self.signature."""
        raw = json.dumps(self.content, sort_keys=True, ensure_ascii=False)
        self.signature = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return self.signature

    def content_text(self) -> str:
        """Flatten content dict to a single text string for embedding."""
        parts = []
        for v in self.content.values():
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        parts.extend(str(val) for val in item.values())
        return " ".join(parts)

    def token_estimate(self) -> int:
        """Rough token count estimate (1 token ≈ 4 chars)."""
        text = self.content_text()
        self.storage_tokens = max(1, len(text) // 4)
        return self.storage_tokens

    # -----------------------------------------------------------------------
    # RL interface
    # -----------------------------------------------------------------------

    # Type one-hot dimension order (for to_rl_state)
    _TYPE_INDEX = {
        MemoryUnitType.INSIGHT:    0,
        MemoryUnitType.TIP:        1,
        MemoryUnitType.TRAJECTORY: 2,
        MemoryUnitType.WORKFLOW:   3,
        MemoryUnitType.SHORTCUT:   4,
    }

    def to_rl_state(self) -> np.ndarray:
        """
        Convert to a fixed-size numeric vector for the RL policy network.

        Layout (16 dims):
          [0:5]   type one-hot            (5)
          [5]     confidence              (1)
          [6]     usage_count (log-scaled)(1)
          [7]     success_rate            (1)
          [8]     age_hours (log-scaled)  (1)
          [9]     hours_since_access (log)(1)
          [10]    access_count (log)      (1)
          [11]    decay_weight            (1)
          [12]    is_active (0/1)         (1)
          [13]    storage_tokens (log)    (1)
          [14]    num_relations           (1)
          [15]    task_outcome (1=success) (1)
        """
        vec = np.zeros(16, dtype=np.float32)

        # type one-hot
        idx = self._TYPE_INDEX.get(self.type, 0)
        vec[idx] = 1.0

        # quality
        vec[5] = self.confidence
        vec[6] = np.log1p(self.usage_count)
        vec[7] = self.success_rate

        # lifecycle
        vec[8]  = np.log1p(self.age_hours)
        vec[9]  = np.log1p(self.hours_since_access)
        vec[10] = np.log1p(self.access_count)
        vec[11] = self.decay_weight
        vec[12] = 1.0 if self.is_active else 0.0

        # cost
        vec[13] = np.log1p(self.storage_tokens)

        # relations
        vec[14] = float(len(self.relations))

        # source
        vec[15] = 1.0 if self.task_outcome == "success" else 0.0

        return vec

    # -----------------------------------------------------------------------
    # Compatibility with existing MemoryItem
    # -----------------------------------------------------------------------

    def to_memory_item(self, score: Optional[float] = None) -> MemoryItem:
        """Downcast to MemoryItem for the retrieval interface."""
        return MemoryItem(
            id=self.id,
            content=self.content_text(),
            metadata={
                "type": self.type.value,
                "source_task_id": self.source_task_id,
                "confidence": self.confidence,
                "success_rate": self.success_rate,
                "decay_weight": self.decay_weight,
            },
            score=score if score is not None else self.effective_score,
            type=MemoryItemType.TEXT,
        )

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict (embedding stored as list)."""
        return {
            "id": self.id,
            "type": self.type.value,
            "signature": self.signature,
            "content": self.content,
            "source_task_id": self.source_task_id,
            "source_task_query": self.source_task_query,
            "task_outcome": self.task_outcome,
            "extraction_model": self.extraction_model,
            "confidence": self.confidence,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "decay_weight": self.decay_weight,
            "is_active": self.is_active,
            "storage_tokens": self.storage_tokens,
            "applicable_domains": self.applicable_domains,
            "applicable_task_types": self.applicable_task_types,
            "relations": [r.to_dict() for r in self.relations],
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MemoryUnit":
        """Deserialize from dict."""
        emb = d.get("embedding")
        return MemoryUnit(
            id=d.get("id", str(uuid.uuid4())),
            type=MemoryUnitType(d["type"]),
            signature=d.get("signature", ""),
            content=d.get("content", {}),
            source_task_id=d.get("source_task_id", ""),
            source_task_query=d.get("source_task_query", ""),
            task_outcome=d.get("task_outcome", ""),
            extraction_model=d.get("extraction_model", ""),
            confidence=d.get("confidence", 1.0),
            usage_count=d.get("usage_count", 0),
            success_count=d.get("success_count", 0),
            created_at=d.get("created_at", datetime.now().isoformat()),
            last_accessed=d.get("last_accessed"),
            access_count=d.get("access_count", 0),
            decay_weight=d.get("decay_weight", 1.0),
            is_active=d.get("is_active", True),
            storage_tokens=d.get("storage_tokens", 0),
            applicable_domains=d.get("applicable_domains", []),
            applicable_task_types=d.get("applicable_task_types", []),
            relations=[MemoryRelation.from_dict(r) for r in d.get("relations", [])],
            embedding=np.array(emb, dtype=np.float32) if emb is not None else None,
        )

    def __repr__(self) -> str:
        return (
            f"MemoryUnit(id={self.id[:8]}..., type={self.type.value}, "
            f"confidence={self.confidence:.2f}, usage={self.usage_count}, "
            f"success_rate={self.success_rate:.2f}, active={self.is_active})"
        )


# ============================================================
# Batch splitter: extraction output → atomic MemoryUnits
# ============================================================

def split_extraction_output(
    extraction_result: Dict[str, Any],
    unit_type: MemoryUnitType,
    source_task_id: str,
    source_task_query: str,
    task_outcome: str,
    extraction_model: str = "",
) -> List[MemoryUnit]:
    """
    Split a raw extraction output (from LLM) into atomic MemoryUnits.

    Handles the fact that some extraction types produce batches:
      - Tips:      {planning_and_decision_tips: [...], tool_and_search_tips: [...]}
                   → one MemoryUnit per individual tip
      - Shortcuts: [macro1, macro2, ...]
                   → one MemoryUnit per macro
      - Others:    already atomic, wrapped as-is
    """
    common = dict(
        source_task_id=source_task_id,
        source_task_query=source_task_query,
        task_outcome=task_outcome,
        extraction_model=extraction_model,
    )

    units: List[MemoryUnit] = []

    if unit_type == MemoryUnitType.TIP:
        for category in ("planning_and_decision_tips", "tool_and_search_tips", "answer_format_tips"):
            cat_label = category.replace("_tips", "")
            for item in extraction_result.get(category, []):
                content = dict(item)
                content["category"] = cat_label
                u = MemoryUnit(type=unit_type, content=content, **common)
                u.compute_signature()
                u.token_estimate()
                units.append(u)

    elif unit_type == MemoryUnitType.SHORTCUT:
        items = extraction_result if isinstance(extraction_result, list) else [extraction_result]
        for item in items:
            u = MemoryUnit(type=unit_type, content=dict(item), **common)
            u.compute_signature()
            u.token_estimate()
            units.append(u)

    elif unit_type == MemoryUnitType.TRAJECTORY:
        steps = extraction_result if isinstance(extraction_result, list) else extraction_result.get("steps", [])
        content = {
            "steps": steps,
            "task_outcome": task_outcome,
        }
        u = MemoryUnit(type=unit_type, content=content, **common)
        u.compute_signature()
        u.token_estimate()
        units.append(u)

    else:
        # INSIGHT, WORKFLOW — already atomic
        u = MemoryUnit(type=unit_type, content=dict(extraction_result), **common)
        u.compute_signature()
        u.token_estimate()
        units.append(u)

    return units
