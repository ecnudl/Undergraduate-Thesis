"""
Base Retriever — Abstract interface and unified output types for the Retrieval layer.

Core types:
  - QueryContext:  Encapsulates query + optional embedding + metadata
  - ScoredUnit:    MemoryUnit + retrieval score + retrieval method tag
  - TraceEntry:    One step in the retrieval path (for explainability)
  - EvidenceRef:   Lightweight source reference (unit_id, type, snippet)
  - MemoryPack:    Unified retrieval output — the single return type for all retrievers

Usage:
  from EvolveLab.retrieval import SemanticRetriever, MemoryPack

  retriever = SemanticRetriever(store, embedding_model)
  pack = retriever.retrieve(QueryContext(query="How to parse PDF?"))
  prompt_str = pack.to_prompt_string()   # ready for LLM injection
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from EvolveLab.memory_schema import MemoryUnit, MemoryUnitType


# ============================================================
# Query Context
# ============================================================

@dataclass
class QueryContext:
    """Input to a retriever: the current task query + optional enrichments."""
    query: str
    embedding: Optional[np.ndarray] = None
    task_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Scored Unit (retrieval result atom)
# ============================================================

@dataclass
class ScoredUnit:
    """A MemoryUnit annotated with a retrieval score and method tag."""
    unit: MemoryUnit
    score: float
    method: str = ""  # e.g. "semantic", "keyword", "graph"

    def __repr__(self) -> str:
        return (
            f"ScoredUnit({self.unit.type.value}, "
            f"score={self.score:.3f}, method={self.method})"
        )


# ============================================================
# Trace & Evidence (explainability)
# ============================================================

@dataclass
class TraceEntry:
    """One step in the retrieval path, for debugging and explainability."""
    step: int
    method: str           # "semantic", "keyword", "graph_expand", etc.
    candidates: int       # number of candidates at this step
    selected: int         # number selected / passed to next step
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "method": self.method,
            "candidates": self.candidates,
            "selected": self.selected,
            "params": self.params,
        }


@dataclass
class EvidenceRef:
    """Lightweight source reference for a retrieved memory."""
    unit_id: str
    unit_type: str
    snippet: str          # short text excerpt
    score: float
    source_task_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "unit_type": self.unit_type,
            "snippet": self.snippet,
            "score": self.score,
            "source_task_id": self.source_task_id,
        }


# ============================================================
# MemoryPack — Unified retrieval output
# ============================================================

# Type-specific formatting templates
def _format_tip(c: dict, s: float) -> str:
    lines = [
        f"[TIP] {c.get('topic', '')} (relevance: {s:.2f})",
        f"  Principle: {c.get('principle', '')}",
        f"  Example: {c.get('micro_example', '')}",
    ]
    if c.get('applicability'):
        lines.append(f"  Applies when: {c['applicability']}")
    tags = c.get('task_type_tags', [])
    if tags:
        lines.append(f"  Domain: {', '.join(tags)}")
    return "\n".join(lines)


def _format_insight(c: dict, s: float) -> str:
    lines = [
        f"[INSIGHT] (relevance: {s:.2f})",
        f"  Root cause: {c.get('root_cause_conclusion', '')}",
        f"  Mismatch: {c.get('state_mismatch_analysis', '')}",
    ]
    if c.get('failure_pattern'):
        lines.append(f"  Pattern: {c['failure_pattern']}")
    if c.get('applicability'):
        lines.append(f"  Applies when: {c['applicability']}")
    tags = c.get('task_type_tags', [])
    if tags:
        lines.append(f"  Domain: {', '.join(tags)}")
    return "\n".join(lines)


_TYPE_FORMATTERS = {
    MemoryUnitType.TIP: lambda c, s: _format_tip(c, s),
    MemoryUnitType.INSIGHT: lambda c, s: _format_insight(c, s),
    MemoryUnitType.WORKFLOW: lambda c, s: _format_workflow(c, s),
    MemoryUnitType.TRAJECTORY: lambda c, s: _format_trajectory(c, s),
    MemoryUnitType.SHORTCUT: lambda c, s: (
        f"[SHORTCUT] {c.get('name', '')} (relevance: {s:.2f})\n"
        f"  {c.get('description', '')}\n"
        f"  Precondition: {c.get('precondition', '')}"
    ),
}


def _format_workflow(c: dict, s: float) -> str:
    parts = [f"[WORKFLOW] (relevance: {s:.2f})"]
    for wf_key in ("agent_workflow", "search_workflow"):
        steps = c.get(wf_key, [])
        if not steps:
            continue
        parts.append(f"  {wf_key}:")
        for st in steps:
            step_num = st.get("step", "?")
            action = st.get("action", st.get("query_formulation", ""))
            parts.append(f"    Step {step_num}: {action}")
    return "\n".join(parts)


def _format_trajectory(c: dict, s: float) -> str:
    steps = c.get("steps", [])
    parts = [f"[TRAJECTORY] ({len(steps)} steps, relevance: {s:.2f})"]
    for st in steps[:5]:  # truncate to first 5 steps
        sid = st.get("step_id", st.get("step", "?"))
        action = st.get("action", "")
        parts.append(f"  Step {sid}: {action}")
    if len(steps) > 5:
        parts.append(f"  ... ({len(steps) - 5} more steps)")
    return "\n".join(parts)


def _format_scored_unit(su: ScoredUnit) -> str:
    """Format a single ScoredUnit into human-readable text."""
    formatter = _TYPE_FORMATTERS.get(su.unit.type)
    if formatter:
        return formatter(su.unit.content, su.score)
    # Fallback: generic format
    text = su.unit.content_text()[:200]
    return f"[{su.unit.type.value.upper()}] (relevance: {su.score:.2f})\n  {text}"


@dataclass
class MemoryPack:
    """
    Unified retrieval output.

    Contains the selected memories, grouped by type, with full trace
    and evidence references for explainability.

    The main consumer-facing method is `to_prompt_string()`, which
    produces a formatted string wrapped in begin/end markers, ready
    for injection into the LLM context.
    """
    query_context: QueryContext
    scored_units: List[ScoredUnit] = field(default_factory=list)
    trace: List[TraceEntry] = field(default_factory=list)
    evidence: List[EvidenceRef] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    retriever_name: str = ""

    # ---- Derived accessors ----

    @property
    def by_type(self) -> Dict[str, List[ScoredUnit]]:
        """Group scored units by MemoryUnitType value."""
        groups: Dict[str, List[ScoredUnit]] = {}
        for su in self.scored_units:
            key = su.unit.type.value
            groups.setdefault(key, []).append(su)
        return groups

    @property
    def selected_units(self) -> List[MemoryUnit]:
        """Flat list of MemoryUnit objects (no scores)."""
        return [su.unit for su in self.scored_units]

    @property
    def total_tokens(self) -> int:
        """Rough total token estimate for all selected units."""
        return sum(su.unit.storage_tokens for su in self.scored_units)

    def is_empty(self) -> bool:
        return len(self.scored_units) == 0

    # ---- Formatting ----

    def to_prompt_string(
        self,
        begin_marker: str = "----Memory System Guidance----",
        end_marker: str = "----End Memory----",
        max_units: Optional[int] = None,
        group_by_type: bool = True,
    ) -> str:
        """
        Format the retrieval result into a string for LLM context injection.

        Compatible with the existing agent framework format:
          ----Memory System Guidance----
          <formatted memories>
          ----End Memory----
        """
        if self.is_empty():
            return ""

        units_to_format = self.scored_units
        if max_units is not None:
            units_to_format = units_to_format[:max_units]

        if group_by_type:
            body = self._format_grouped(units_to_format)
        else:
            body = "\n\n".join(_format_scored_unit(su) for su in units_to_format)

        return f"{begin_marker}\n{body}\n{end_marker}"

    def _format_grouped(self, units: List[ScoredUnit]) -> str:
        """Format units grouped by type, with section headers."""
        groups: Dict[str, List[ScoredUnit]] = {}
        for su in units:
            key = su.unit.type.value
            groups.setdefault(key, []).append(su)

        sections = []
        # Order: tip > insight > workflow > trajectory > shortcut
        type_order = ["tip", "insight", "workflow", "trajectory", "shortcut"]
        for t in type_order:
            if t not in groups:
                continue
            section_units = groups[t]
            formatted = [_format_scored_unit(su) for su in section_units]
            sections.append("\n\n".join(formatted))

        return "\n\n".join(sections)

    def to_guidance_text(self) -> str:
        """Return only the body text (no begin/end markers), for embedding into MemoryItem."""
        if self.is_empty():
            return ""
        return "\n\n".join(_format_scored_unit(su) for su in self.scored_units)

    # ---- Serialization ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query_context.query,
            "retriever": self.retriever_name,
            "num_units": len(self.scored_units),
            "by_type": {k: len(v) for k, v in self.by_type.items()},
            "total_tokens": self.total_tokens,
            "trace": [t.to_dict() for t in self.trace],
            "evidence": [e.to_dict() for e in self.evidence],
            "created_at": self.created_at,
        }


# ============================================================
# Base Retriever — Abstract interface
# ============================================================

class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval strategies.

    Subclasses implement `retrieve()` which takes a QueryContext
    and returns a MemoryPack.

    All retrievers operate on a BaseMemoryStorage backend.
    """

    def __init__(self, store, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            store: A BaseMemoryStorage instance (JsonStorage, VectorStorage, etc.)
            config: Strategy-specific configuration dict.
        """
        self.store = store
        self.config = config or {}

    @property
    def name(self) -> str:
        """Retriever name, used in MemoryPack.retriever_name and trace."""
        return self.__class__.__name__

    @abstractmethod
    def retrieve(self, ctx: QueryContext, top_k: int = 5) -> MemoryPack:
        """
        Retrieve relevant memories for the given query context.

        Args:
            ctx: QueryContext with query text, optional embedding, metadata.
            top_k: Maximum number of units to return.

        Returns:
            MemoryPack with scored units, trace, and evidence.
        """
        ...

    def _build_evidence(self, scored_units: List[ScoredUnit]) -> List[EvidenceRef]:
        """Build evidence references from scored units."""
        evidence = []
        for su in scored_units:
            text = su.unit.content_text()
            snippet = text[:120] + "..." if len(text) > 120 else text
            evidence.append(EvidenceRef(
                unit_id=su.unit.id,
                unit_type=su.unit.type.value,
                snippet=snippet,
                score=su.score,
                source_task_id=su.unit.source_task_id,
            ))
        return evidence

    def _make_pack(
        self,
        ctx: QueryContext,
        scored_units: List[ScoredUnit],
        trace: List[TraceEntry],
    ) -> MemoryPack:
        """Convenience: build a complete MemoryPack from retrieval results."""
        return MemoryPack(
            query_context=ctx,
            scored_units=scored_units,
            trace=trace,
            evidence=self._build_evidence(scored_units),
            retriever_name=self.name,
        )
