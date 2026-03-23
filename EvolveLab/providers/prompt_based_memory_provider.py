"""
PromptBasedMemoryProvider - Configurable multi-prompt memory extraction and retrieval.

Uses a subset of 5 extraction prompts (insight/tip/trajectory/workflow/shortcut)
controlled by `enabled_prompts` config to extract MemoryUnits from trajectories,
then provides semantic retrieval at query time.
"""

import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional

import jinja2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..base_memory import BaseMemoryProvider
from ..memory_types import (
    MemoryItem,
    MemoryRequest,
    MemoryResponse,
    MemoryStatus,
    MemoryType,
    TrajectoryData,
)
from ..memory_schema import MemoryUnit, MemoryUnitType, split_extraction_output

logger = logging.getLogger(__name__)

# MemoryUnitType corresponding to each prompt name
PROMPT_TO_UNIT_TYPE = {
    "insight": MemoryUnitType.INSIGHT,
    "tip": MemoryUnitType.TIP,
    "trajectory": MemoryUnitType.TRAJECTORY,
    "workflow": MemoryUnitType.WORKFLOW,
    "shortcut": MemoryUnitType.SHORTCUT,
}

# Prompt file name mapping
PROMPT_FILE_NAMES = {
    "insight": "insights_prompt.txt",
    "tip": "tips_prompt.txt",
    "trajectory": "trajectory_prompt.txt",
    "workflow": "workflow_prompt.txt",
    "shortcut": "shortcut_prompt.txt",
}


def _load_embedding_model(model_name: str, cache_dir: str):
    """Load SentenceTransformer with local cache fallback."""
    from sentence_transformers import SentenceTransformer

    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, model_name.replace("/", "_"))
    hf_cache_root = os.path.expanduser("~/.cache/huggingface/hub")
    hf_repo_dir = os.path.join(
        hf_cache_root, f"models--{model_name.replace('/', '--')}"
    )
    device = (
        os.environ.get("MEMORY_EMBEDDING_DEVICE")
        or os.environ.get("SENTENCE_TRANSFORMERS_DEVICE")
        or None
    )

    def _make_model(model_path: str):
        if device:
            return SentenceTransformer(model_path, device=device)
        return SentenceTransformer(model_path)

    try:
        if os.path.exists(local_path) and os.listdir(local_path):
            return _make_model(local_path)
    except Exception as e:
        logger.warning(f"Failed to load local embedding model: {e}")

    # Prefer an existing Hugging Face snapshot if it is already cached locally.
    try:
        snapshots_dir = os.path.join(hf_repo_dir, "snapshots")
        if os.path.isdir(snapshots_dir):
            snapshot_names = sorted(os.listdir(snapshots_dir), reverse=True)
            for snapshot_name in snapshot_names:
                snapshot_path = os.path.join(snapshots_dir, snapshot_name)
                if os.path.isfile(os.path.join(snapshot_path, "modules.json")):
                    logger.info(
                        "Loading embedding model from local Hugging Face cache: %s",
                        snapshot_path,
                    )
                    return _make_model(snapshot_path)
    except Exception as e:
        logger.warning(f"Failed to load Hugging Face cached embedding model: {e}")

    model = _make_model(model_name)
    model.save(local_path)
    return model


def _parse_json_from_response(text: str):
    """Multi-strategy JSON extraction from LLM response."""
    # Try 1: direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try 2: ```json ... ``` code block
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try 3: outermost [ ... ] or { ... }
    for pattern in [r"\[[\s\S]*\]", r"\{[\s\S]*\}"]:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue

    logger.warning(f"Failed to parse JSON from LLM response: {text[:300]}...")
    return None


def _format_trajectory_text(trajectory_data: TrajectoryData) -> str:
    """
    Format TrajectoryData.trajectory list into readable text.

    Replicates the logic from test_memory_extraction.py:format_trajectory_text
    but operates on TrajectoryData instead of raw dict.
    """
    parts = []
    trajectory = trajectory_data.trajectory or []

    for i, step in enumerate(trajectory):
        step_name = step.get("name", "unknown")

        if step_name == "plan":
            plan_text = step.get("value", "")
            if len(plan_text) > 2000:
                plan_text = plan_text[:2000] + "\n... [truncated]"
            parts.append(f"[Step {i} - Plan]\n{plan_text}")

        elif step_name == "action":
            tool_calls = step.get("tool_calls", [])
            obs = step.get("obs", "")
            think = step.get("think", "")

            tc_texts = []
            for tc in tool_calls:
                tc_name = tc.get("name", "unknown")
                tc_args = tc.get("arguments", {})
                if isinstance(tc_args, dict):
                    tc_args_str = json.dumps(tc_args, ensure_ascii=False)
                else:
                    tc_args_str = str(tc_args)
                tc_texts.append(f"  - {tc_name}({tc_args_str})")

            action_text = f"[Step {i} - Action]"
            if think:
                think_display = think[:500] + "..." if len(think) > 500 else think
                action_text += f"\nThinking: {think_display}"
            if tc_texts:
                action_text += f"\nTool Calls:\n" + "\n".join(tc_texts)
            if obs:
                obs_display = (
                    obs[:1500] + "\n... [truncated]" if len(obs) > 1500 else obs
                )
                action_text += f"\nObservations:\n{obs_display}"

            parts.append(action_text)

        elif step_name == "summary":
            summary_text = step.get("value", "")
            if len(summary_text) > 1000:
                summary_text = summary_text[:1000] + "\n... [truncated]"
            parts.append(f"[Step {i} - Summary]\n{summary_text}")

    return "\n\n".join(parts)


def _build_template_context(
    trajectory_data: TrajectoryData, is_correct: bool
) -> Dict[str, Any]:
    """Build Jinja2 template context from TrajectoryData."""
    metadata = trajectory_data.metadata or {}

    raw_trajectory = _format_trajectory_text(trajectory_data)

    # Derive failure reason
    failure_reason = None
    if not is_correct:
        golden = metadata.get("golden_answer", "")
        if golden:
            failure_reason = (
                f"Answer mismatch: agent answered '{trajectory_data.result}', "
                f"expected '{golden}'"
            )
        else:
            failure_reason = metadata.get("error", "Unknown failure")

    return {
        "task_query": trajectory_data.query,
        "is_success": is_correct,
        "raw_trajectory": raw_trajectory,
        "final_result": str(trajectory_data.result) if trajectory_data.result else "",
        "golden_answer": str(metadata.get("golden_answer", "")),
        "task_id": metadata.get("task_id", ""),
        "task_order": metadata.get("task_order", 0),
        "memory_guidance": None,
        "failure_reason": failure_reason,
        "reference_trajectory": None,
        "memory_count_before": metadata.get("memory_count_before", 0),
    }


def _render_prompt(template_str: str, context: dict) -> str:
    """Render Jinja2 template with context."""
    env = jinja2.Environment(undefined=jinja2.Undefined)
    template = env.from_string(template_str)
    return template.render(**context)


class PromptBasedMemoryProvider(BaseMemoryProvider):
    """
    Multi-prompt memory provider that can be configured to use any combination
    of the 5 extraction prompts (insight/tip/trajectory/workflow/shortcut).

    Key config fields:
      enabled_prompts: List[str]   – which prompts to activate
      prompt_dir: str              – directory containing the prompt .txt files
      storage_dir: str             – base directory for persistence
      db_path: str                 – JSON file storing MemoryUnits
      top_k: int                   – number of memories to retrieve
      embedding_model_name: str    – sentence-transformers model id
      embedding_cache_dir: str     – local model cache directory
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(MemoryType.PROMPT_BASED, config)

        self.model = self.config.get("model")
        # Support PROMPT_STORAGE_DIR env var for parallel experiments
        self.storage_dir = os.environ.get("PROMPT_STORAGE_DIR") or self.config.get(
            "storage_dir", "./storage/prompt_based"
        )
        self.db_path = os.path.join(self.storage_dir, "memory_db.json")
        # Support ENABLED_PROMPTS env var override (env takes precedence)
        env_enabled_prompts = os.environ.get("ENABLED_PROMPTS", "").strip()
        if env_enabled_prompts:
            self.enabled_prompts = [
                p.strip() for p in env_enabled_prompts.split(",") if p.strip()
            ]
        else:
            self.enabled_prompts: List[str] = self.config.get(
                "enabled_prompts", ["tip", "workflow"]
            )
        self.prompt_dir = self.config.get("prompt_dir", ".")
        env_top_k = os.environ.get("PROMPT_TOP_K", "").strip()
        if env_top_k:
            try:
                self.top_k = int(env_top_k)
            except ValueError:
                logger.warning(
                    f"Invalid PROMPT_TOP_K='{env_top_k}', fallback to config/default"
                )
                self.top_k = self.config.get("top_k", 5)
        else:
            self.top_k = self.config.get("top_k", 5)
        self.embedding_model_name = self.config.get(
            "embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_cache_dir = self.config.get(
            "embedding_cache_dir", "./storage/models"
        )

        # Prompt templates (loaded during initialize)
        self._prompt_templates: Dict[str, str] = {}

        # Internal state
        self.memory_units: List[MemoryUnit] = []
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_model = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        try:
            os.makedirs(self.storage_dir, exist_ok=True)

            # Load embedding model
            self.embedding_model = _load_embedding_model(
                self.embedding_model_name, self.embedding_cache_dir
            )

            # Load existing memories from db
            if os.path.exists(self.db_path):
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.memory_units = [MemoryUnit.from_dict(d) for d in data]
                self._rebuild_embeddings()
                logger.info(
                    f"Loaded {len(self.memory_units)} existing memory units from {self.db_path}"
                )
            else:
                self.memory_units = []
                self.embeddings = None

            # Load enabled prompt templates
            for prompt_name in self.enabled_prompts:
                fname = PROMPT_FILE_NAMES.get(prompt_name)
                if not fname:
                    logger.warning(f"Unknown prompt name: {prompt_name}, skipping")
                    continue
                fpath = os.path.join(self.prompt_dir, fname)
                if not os.path.exists(fpath):
                    logger.warning(f"Prompt file not found: {fpath}, skipping")
                    continue
                with open(fpath, "r", encoding="utf-8") as f:
                    self._prompt_templates[prompt_name] = f.read()

            logger.info(
                f"PromptBasedMemoryProvider initialized with prompts: "
                f"{list(self._prompt_templates.keys())}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize PromptBasedMemoryProvider: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    # Memory ingestion
    # ------------------------------------------------------------------

    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple:
        """
        Extract memory units from a trajectory using all enabled prompts.

        Returns:
            (bool, str): success flag and description message
        """
        if not self.model:
            return False, "No model provided for memory extraction"

        if not self._prompt_templates:
            return False, "No prompt templates loaded"

        metadata = trajectory_data.metadata or {}
        is_correct = metadata.get("is_correct", False)
        task_outcome = "success" if is_correct else "failure"
        task_id = metadata.get("task_id", str(uuid.uuid4())[:8])

        # Build template context
        context = _build_template_context(trajectory_data, is_correct)

        new_units: List[MemoryUnit] = []
        prompts_used = []

        for prompt_name, template_str in self._prompt_templates.items():
            unit_type = PROMPT_TO_UNIT_TYPE.get(prompt_name)
            if unit_type is None:
                continue

            # Skip conditions based on prompt type and task outcome
            if prompt_name == "insight" and is_correct:
                # Insight is failure-only
                continue
            if prompt_name == "workflow" and not is_correct:
                # Workflow is success-only
                continue

            # Render prompt
            try:
                filled_prompt = _render_prompt(template_str, context)
            except Exception as e:
                logger.error(
                    f"Template rendering failed for {prompt_name}: {e}"
                )
                continue

            # Call LLM
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": filled_prompt}],
                    }
                ]
                response = self.model(messages)
                response_text = (
                    response.content
                    if hasattr(response, "content")
                    else str(response)
                )
            except Exception as e:
                logger.error(f"LLM call failed for {prompt_name}: {e}")
                continue

            # Parse JSON
            parsed = _parse_json_from_response(response_text)
            if parsed is None:
                logger.warning(f"Failed to parse extraction result for {prompt_name}")
                continue

            # Check for skipped output
            if isinstance(parsed, dict) and parsed.get("skipped"):
                logger.info(f"Prompt {prompt_name} returned skipped output")
                continue

            # Split into atomic MemoryUnits
            try:
                units = split_extraction_output(
                    extraction_result=parsed,
                    unit_type=unit_type,
                    source_task_id=task_id,
                    source_task_query=trajectory_data.query,
                    task_outcome=task_outcome,
                    extraction_model=str(
                        getattr(self.model, "model_id", "unknown")
                    ),
                )
            except Exception as e:
                logger.error(
                    f"split_extraction_output failed for {prompt_name}: {e}"
                )
                continue

            # Dedup by signature and compute embeddings
            existing_sigs = {u.signature for u in self.memory_units}
            for unit in units:
                if unit.signature in existing_sigs:
                    logger.debug(
                        f"Duplicate signature {unit.signature}, skipping"
                    )
                    continue

                # Compute embedding
                text_for_emb = unit.content_text()
                if text_for_emb and self.embedding_model is not None:
                    unit.embedding = self.embedding_model.encode(
                        text_for_emb, convert_to_numpy=True
                    )

                new_units.append(unit)
                existing_sigs.add(unit.signature)

            prompts_used.append(prompt_name)

        # Append to memory and persist
        if new_units:
            self.memory_units.extend(new_units)
            self._rebuild_embeddings()
            self._save_db()

        msg = (
            f"Extracted {len(new_units)} memory units from "
            f"{len(prompts_used)} prompts ({', '.join(prompts_used)})"
        )
        logger.info(msg)
        return True, msg

    # ------------------------------------------------------------------
    # Memory retrieval
    # ------------------------------------------------------------------

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        """Retrieve top-k relevant memories via semantic similarity."""
        empty_response = MemoryResponse(
            memories=[],
            memory_type=self.memory_type,
            total_count=0,
            request_id=str(uuid.uuid4()),
        )

        # Only provide at BEGIN phase
        if request.status != MemoryStatus.BEGIN:
            return empty_response

        # Filter active units that have embeddings
        active_units = [
            u for u in self.memory_units if u.is_active and u.embedding is not None
        ]
        if not active_units:
            return empty_response

        if self.embedding_model is None:
            return empty_response

        # Compute query embedding
        query_emb = self.embedding_model.encode(
            request.query, convert_to_numpy=True
        ).reshape(1, -1)

        # Build embedding matrix from active units
        emb_matrix = np.vstack([u.embedding for u in active_units])

        # Cosine similarity
        sims = cosine_similarity(query_emb, emb_matrix)[0]
        top_k = min(self.top_k, len(active_units))
        top_indices = sims.argsort()[-top_k:][::-1]

        # Format guidance text
        guidance_parts = []
        for idx in top_indices:
            unit = active_units[idx]
            score = float(sims[idx])
            formatted = self._format_memory_unit(unit, score)
            if formatted:
                guidance_parts.append(formatted)
            unit.record_access()

        if not guidance_parts:
            return empty_response

        guidance_text = "\n\n".join(guidance_parts)

        logger.info(
            f"provide_memory: returning {len(guidance_parts)} memories "
            f"(query='{request.query[:60]}...', active_units={len(active_units)})"
        )

        memory_item = MemoryItem(
            id=f"prompt_based_{uuid.uuid4()}",
            content=guidance_text,
            metadata={
                "num_sources": len(guidance_parts),
                "enabled_prompts": self.enabled_prompts,
                "total_memory_units": len(self.memory_units),
            },
            score=float(np.mean([sims[i] for i in top_indices])),
        )

        return MemoryResponse(
            memories=[memory_item],
            memory_type=self.memory_type,
            total_count=1,
            request_id=str(uuid.uuid4()),
        )

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_memory_unit(unit: MemoryUnit, score: float) -> Optional[str]:
        """Format a single MemoryUnit into readable guidance text."""
        c = unit.content
        tag = unit.type.value.upper()
        score_str = f"(relevance: {score:.2f})"

        if unit.type == MemoryUnitType.TIP:
            topic = c.get("topic", "")
            principle = c.get("principle", "")
            micro = c.get("micro_example", "")
            return (
                f"[TIP] {topic} {score_str}\n"
                f"  Principle: {principle}\n"
                f"  Example: {micro}"
            )

        elif unit.type == MemoryUnitType.WORKFLOW:
            parts = [f"[WORKFLOW] {score_str}"]
            for wf_key in ("agent_workflow", "search_workflow"):
                steps = c.get(wf_key, [])
                if not steps:
                    continue
                parts.append(f"  {wf_key}:")
                for s in steps:
                    step_num = s.get("step", "?")
                    action = s.get("action", s.get("query_formulation", ""))
                    parts.append(f"    Step {step_num}: {action}")
            return "\n".join(parts)

        elif unit.type == MemoryUnitType.INSIGHT:
            root = c.get("root_cause_conclusion", "")
            mismatch = c.get("state_mismatch_analysis", "")
            return (
                f"[INSIGHT] {score_str}\n"
                f"  Root cause: {root}\n"
                f"  Mismatch: {mismatch}"
            )

        elif unit.type == MemoryUnitType.TRAJECTORY:
            steps = c.get("steps", [])
            n = len(steps)
            parts = [f"[TRAJECTORY] {n} steps {score_str}"]
            for s in steps[:5]:  # Show at most 5 steps
                sid = s.get("step_id", "?")
                action = s.get("action", "")
                parts.append(f"  {sid}: {action}")
            if n > 5:
                parts.append(f"  ... ({n - 5} more steps)")
            return "\n".join(parts)

        elif unit.type == MemoryUnitType.SHORTCUT:
            name = c.get("name", "unnamed")
            desc = c.get("description", "")
            precond = c.get("precondition", "")
            return (
                f"[SHORTCUT] {name} {score_str}\n"
                f"  Description: {desc}\n"
                f"  Precondition: {precond}"
            )

        return f"[{tag}] {score_str}\n  {unit.content_text()[:200]}"

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_db(self) -> None:
        """Persist all memory units to db_path."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        data = [u.to_dict() for u in self.memory_units]
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _rebuild_embeddings(self) -> None:
        """Rebuild the numpy embedding matrix from unit embeddings."""
        embs = [u.embedding for u in self.memory_units if u.embedding is not None]
        if embs:
            self.embeddings = np.vstack(embs)
        else:
            self.embeddings = None
