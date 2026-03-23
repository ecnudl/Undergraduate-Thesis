#!/usr/bin/env python
# coding=utf-8

"""Run LongMemEval with no-memory or memory-backed evaluation flows."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import threading
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from FlashOAgents import OpenAIServerModel
from EvolveLab.config import get_memory_config
from EvolveLab.memory_schema import MemoryUnit, MemoryUnitType
from EvolveLab.memory_types import (
    MemoryRequest,
    MemoryStatus,
    MemoryType,
    PROVIDER_MAPPING,
    TrajectoryData,
)
from eval_utils import (
    TaskTimer,
    TokenCounter,
    capture_memory_metrics,
    create_run_directory,
    enrich_result_with_metrics,
    generate_unified_report,
    save_task_result,
)
from utils import write_jsonl


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

load_dotenv(override=False)
os.environ.setdefault("FORCE_STREAM", "1")

FULL_CONTEXT_SYSTEM_PROMPT = (
    "You answer questions using only the provided conversation history. "
    "Do not invent facts. Return a short direct answer with no explanation."
)
MEMORY_SYSTEM_PROMPT = (
    "You answer questions using only the retrieved memory snippets. "
    "If the snippets are insufficient, answer with the most faithful short answer you can infer and do not add explanation."
)

QUESTION_TYPE_TAGS = {
    "multi-session": ["lookup_factual", "multi_step_reasoning"],
    "temporal-reasoning": ["date_time", "multi_step_reasoning"],
    "knowledge-update": ["lookup_factual", "multi_step_reasoning"],
    "single-session-user": ["text_extraction", "lookup_factual"],
    "single-session-assistant": ["text_extraction", "lookup_factual"],
    "single-session-preference": ["text_extraction", "lookup_factual"],
}

QUESTION_TYPE_FAILURE_PATTERN = {
    "multi-session": "scope_error",
    "temporal-reasoning": "logic_error",
    "knowledge-update": "wrong_value",
    "single-session-user": "wrong_entity",
    "single-session-assistant": "wrong_entity",
    "single-session-preference": "wrong_entity",
}


def parse_task_indices(indices_str: Optional[str]) -> Optional[set[int]]:
    if not indices_str:
        return None

    indices: set[int] = set()
    for part in indices_str.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            if start > end:
                raise ValueError(f"Invalid range: {token}")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(token))
    return indices


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"'`]", "", text)
    text = re.sub(r"[\.,;:!?()\[\]{}]", "", text)
    return text


def truncate_text(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def load_longmemeval(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"LongMemEval file must contain a list: {path}")

    normalized = []
    for idx, item in enumerate(data, start=1):
        record = dict(item)
        record["_global_index"] = idx
        normalized.append(record)
    return normalized


def load_selection_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Selection file must contain a JSON object: {path}")
    return payload


def format_session_messages(session: List[Dict]) -> List[str]:
    lines: List[str] = []
    for message in session:
        role = str(message.get("role", "unknown")).strip().capitalize()
        content = message.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(x) for x in content)
        lines.append(f"{role}: {str(content).strip()}")
    return lines


def build_long_context_prompt(item: Dict) -> tuple[str, Dict[str, int]]:
    session_dates = item.get("haystack_dates", [])
    session_ids = item.get("haystack_session_ids", [])
    sessions = item.get("haystack_sessions", [])

    blocks: List[str] = []
    total_messages = 0
    for idx, session in enumerate(sessions, start=1):
        date = session_dates[idx - 1] if idx - 1 < len(session_dates) else "unknown"
        session_id = session_ids[idx - 1] if idx - 1 < len(session_ids) else f"session_{idx}"
        message_lines = format_session_messages(session if isinstance(session, list) else [])
        total_messages += len(message_lines)
        block = "\n".join(
            [
                f"[Session {idx}]",
                f"Date: {date}",
                f"Session ID: {session_id}",
                *message_lines,
            ]
        )
        blocks.append(block)

    context = "\n\n".join(blocks)
    prompt = (
        "Below is the user's conversation history.\n\n"
        f"{context}\n\n"
        f"Question date: {item.get('question_date', 'unknown')}\n"
        f"Question type: {item.get('question_type', 'unknown')}\n"
        f"Question: {item.get('question', '')}\n\n"
        "Answer using only the conversation history. Return only the final answer."
    )
    stats = {
        "num_sessions": len(sessions),
        "num_messages": total_messages,
        "context_characters": len(context),
        "prompt_characters": len(prompt),
    }
    return prompt, stats


def session_transcript_text(session: List[Dict], date: str, session_id: str) -> str:
    lines = format_session_messages(session)
    return "\n".join([f"Date: {date}", f"Session ID: {session_id}", *lines])


def build_memory_answer_prompt(item: Dict, memory_text: str) -> tuple[str, Dict[str, int]]:
    prompt = (
        f"Question date: {item.get('question_date', 'unknown')}\n"
        f"Question type: {item.get('question_type', 'unknown')}\n"
        f"Question: {item.get('question', '')}\n\n"
        "Retrieved memory snippets:\n"
        f"{memory_text or '[No retrieved memory snippets]'}\n\n"
        "Answer using only the retrieved memory snippets. Return only the final answer."
    )
    stats = {
        "num_sessions": len(item.get("haystack_sessions", [])),
        "num_messages": sum(len(session or []) for session in item.get("haystack_sessions", [])),
        "retrieved_memory_characters": len(memory_text or ""),
        "prompt_characters": len(prompt),
    }
    return prompt, stats


def judge_longmemeval_answer(
    question: str,
    golden_answer: str,
    pred_answer: str,
    question_type: str,
    judge_model: Optional[str],
) -> str:
    normalized_pred = normalize_answer(pred_answer)
    normalized_gold = normalize_answer(golden_answer)

    if normalized_pred == normalized_gold:
        return "correct"

    if not judge_model:
        return "incorrect"

    model_config = {
        "model_id": judge_model,
        "custom_role_conversions": {"tool-call": "assistant", "tool-response": "user"},
        "max_completion_tokens": 256,
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "api_base": os.environ.get("JUDGE_API_BASE") or os.environ.get("OPENAI_API_BASE", "https://api-vip.codex-for.me/v1"),
    }
    judge = OpenAIServerModel(**model_config)
    judge.reset_total_counts()

    prompt = f"""Determine whether the predicted answer is correct.

Question type: {question_type}
Question: {question}
Gold answer: {golden_answer}
Predicted answer: {pred_answer}

Judge as correct if the answers are semantically equivalent, including short paraphrases and equivalent numeric/date expressions.
Return JSON only:
{{"judgement": "correct" or "incorrect"}}"""

    try:
        response = judge(
            [
                {"role": "system", "content": [{"type": "text", "text": "You are a strict but fair evaluation judge."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
        )
        content = (response.content or "").strip()
        result = json.loads(content)
        judgement = str(result.get("judgement", "incorrect")).strip().lower()
        return judgement if judgement in {"correct", "incorrect"} else "incorrect"
    except Exception as exc:
        logger.warning("Judge model failed, fallback to exact-match incorrect: %s", exc)
        return "incorrect"


@contextmanager
def temporary_env(overrides: Dict[str, str]):
    previous = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def load_memory_provider(memory_type_str: str, model: OpenAIServerModel, item_storage_dir: Path):
    memory_type = MemoryType(memory_type_str)
    class_name, module_name = PROVIDER_MAPPING[memory_type]
    module = __import__(f"EvolveLab.providers.{module_name}", fromlist=[class_name])
    provider_class = getattr(module, class_name)
    config = get_memory_config(memory_type)
    config["model"] = model

    env_overrides: Dict[str, str] = {}
    if memory_type == MemoryType.MODULAR:
        config["storage_dir"] = str(item_storage_dir)
        env_overrides["MODULAR_STORAGE_DIR"] = str(item_storage_dir)
    elif memory_type == MemoryType.SILICONFRIEND:
        config["store_dir"] = str(item_storage_dir)
        config["memory_file"] = "longmemeval_memory.json"
        config["index_dir"] = str(item_storage_dir / "index")
        config["user_name"] = "longmemeval_eval"
        env_overrides.update(
            {
                "SILICONFRIEND_STORE_DIR": str(item_storage_dir),
                "SILICONFRIEND_MEMORY_FILE": "longmemeval_memory.json",
                "SILICONFRIEND_INDEX_DIR": str(item_storage_dir / "index"),
                "SILICONFRIEND_USER_NAME": "longmemeval_eval",
            }
        )

    with temporary_env(env_overrides):
        provider = provider_class(config=config)
        if provider.initialize():
            return provider

    if memory_type == MemoryType.MODULAR:
        retry_env = dict(env_overrides)
        retry_env["MEMORY_EMBEDDING_DEVICE"] = "cpu"
        config["embedding_device"] = "cpu"
        logger.warning("Retrying modular LongMemEval provider initialization on CPU embedding device.")
        with temporary_env(retry_env):
            provider = provider_class(config=config)
            if provider.initialize():
                return provider

    if memory_type == MemoryType.SILICONFRIEND:
        retry_env = dict(env_overrides)
        retry_env["SILICONFRIEND_EMBEDDING_DEVICE"] = "cpu"
        config["embedding_device"] = "cpu"
        logger.warning("Retrying SiliconFriend LongMemEval provider initialization on CPU embedding device.")
        with temporary_env(retry_env):
            provider = provider_class(config=config)
            if provider.initialize():
                return provider

    raise RuntimeError(f"Failed to initialize memory provider: {memory_type_str}")


def derive_topic(session_lines: List[str], fallback: str) -> str:
    for line in session_lines:
        if line.lower().startswith("user:"):
            return truncate_text(line[5:].strip(), 80) or fallback
    return truncate_text(session_lines[0], 80) if session_lines else fallback


def build_tip_unit(
    item: Dict,
    session_idx: int,
    date: str,
    session_id: str,
    transcript_lines: List[str],
) -> MemoryUnit:
    summary = truncate_text(" | ".join(transcript_lines), 520)
    user_lines = [line[5:].strip() for line in transcript_lines if line.lower().startswith("user:")]
    assistant_lines = [line[10:].strip() for line in transcript_lines if line.lower().startswith("assistant:")]
    topic = derive_topic(transcript_lines, f"{item.get('question_type', 'session')} memory")
    principle = truncate_text(summary, 280)
    micro_example = truncate_text(user_lines[0] if user_lines else (assistant_lines[0] if assistant_lines else summary), 140)
    applicability = truncate_text(
        f"When answering {item.get('question_type', 'longmem')} questions about details mentioned in session {session_id} on {date}.",
        120,
    )
    content = {
        "topic": topic,
        "principle": principle,
        "micro_example": micro_example,
        "counterfactual": "Relevant dialogue details may be missed if this session is not retrieved.",
        "task_type_tags": QUESTION_TYPE_TAGS.get(str(item.get("question_type", "")), ["lookup_factual"]),
        "applicability": applicability,
        "session_id": session_id,
        "session_date": date,
        "session_summary": summary,
    }
    unit = MemoryUnit(
        type=MemoryUnitType.TIP,
        content=content,
        source_task_id=f"{item.get('question_id')}::{session_id}",
        source_task_query=f"LongMemEval session memory for {item.get('question_id')} (session {session_idx + 1})",
        task_outcome="context",
        extraction_model="longmemeval_session_adapter",
    )
    unit.compute_signature()
    unit.token_estimate()
    return unit


def build_insight_unit(
    item: Dict,
    session_idx: int,
    date: str,
    session_id: str,
    transcript_lines: List[str],
) -> MemoryUnit:
    summary = truncate_text(" | ".join(transcript_lines), 520)
    topic = derive_topic(transcript_lines, f"{item.get('question_type', 'session')} memory")
    triples = []
    for line in transcript_lines[:3]:
        if ":" in line:
            role, content = line.split(":", 1)
            triples.append([role.strip(), "mentioned", truncate_text(content.strip(), 80)])
    if not triples:
        triples = [["session", "contains", truncate_text(summary, 80)]]
    content = {
        "root_cause_conclusion": truncate_text(
            f"Session {session_id} on {date} records durable conversational details about: {topic}. {summary}",
            320,
        ),
        "state_mismatch_analysis": truncate_text(
            f"Expected: recall details from session {session_id}; Actual: this session contains {summary}",
            320,
        ),
        "divergence_point": f"[Session {session_idx + 1} - Dialogue]: {session_id} on {date}",
        "failure_knowledge_graph": triples,
        "task_type_tags": QUESTION_TYPE_TAGS.get(str(item.get("question_type", "")), ["lookup_factual"]),
        "failure_pattern": QUESTION_TYPE_FAILURE_PATTERN.get(str(item.get("question_type", "")), "scope_error"),
        "applicability": truncate_text(
            "When the answer depends on locating the correct dialogue among many similar sessions.",
            120,
        ),
        "session_id": session_id,
        "session_date": date,
        "session_summary": summary,
    }
    unit = MemoryUnit(
        type=MemoryUnitType.INSIGHT,
        content=content,
        source_task_id=f"{item.get('question_id')}::{session_id}::insight",
        source_task_query=f"LongMemEval session insight for {item.get('question_id')} (session {session_idx + 1})",
        task_outcome="context",
        extraction_model="longmemeval_session_adapter",
    )
    unit.compute_signature()
    unit.token_estimate()
    return unit


def build_longmemeval_units(item: Dict, session_idx: int) -> List[MemoryUnit]:
    date = item.get("haystack_dates", [])[session_idx]
    session_id = item.get("haystack_session_ids", [])[session_idx]
    session = item.get("haystack_sessions", [])[session_idx]
    transcript_lines = format_session_messages(session if isinstance(session, list) else [])
    if not transcript_lines:
        transcript_lines = [f"Session ID: {session_id}", "No dialogue content recorded."]

    return [
        build_tip_unit(item, session_idx, date, session_id, transcript_lines),
        build_insight_unit(item, session_idx, date, session_id, transcript_lines),
    ]


def ingest_modular_sessions(provider, item: Dict) -> None:
    all_units: List[MemoryUnit] = []
    sessions = item.get("haystack_sessions", [])
    for session_idx in range(len(sessions)):
        all_units.extend(build_longmemeval_units(item, session_idx))

    if provider.embedding_model is not None:
        for unit in all_units:
            text = unit.content_text()
            if text:
                unit.embedding = provider.embedding_model.encode(text, convert_to_numpy=True)

    inserted = provider.store.add(all_units) if all_units else 0
    provider._experiment_metrics["num_extracted"] += len(all_units)
    provider._experiment_metrics["num_inserted"] += inserted
    provider._experiment_metrics["num_deduped"] += max(len(all_units) - inserted, 0)

    if provider.manager is not None and all_units:
        try:
            on_insert_result = provider.manager.run_on_insert(all_units, context={"benchmark": "longmemeval"})
            provider._record_management_results([on_insert_result])
            periodic_result = provider.manager.run_periodic({"benchmark": "longmemeval"})
            provider._record_management_results([periodic_result])
        except Exception as exc:
            logger.warning("LongMemEval modular management failed: %s", exc)

    provider._update_memory_totals()


def ingest_siliconfriend_sessions(provider, item: Dict) -> None:
    sessions = item.get("haystack_sessions", [])
    dates = item.get("haystack_dates", [])
    session_ids = item.get("haystack_session_ids", [])
    for session_idx, session in enumerate(sessions):
        date = dates[session_idx] if session_idx < len(dates) else "unknown"
        session_id = session_ids[session_idx] if session_idx < len(session_ids) else f"session_{session_idx + 1}"
        transcript = session_transcript_text(session if isinstance(session, list) else [], date, session_id)
        trajectory = [{"name": "summary", "value": transcript}]
        summary_text = truncate_text(transcript, 1200)
        trajectory_data = TrajectoryData(
            query=f"Conversation memory for session {session_id} on {date}",
            trajectory=trajectory,
            result=summary_text,
            metadata={
                "session_id": session_id,
                "session_date": date,
                "question_type": item.get("question_type"),
            },
        )
        provider.take_in_memory(trajectory_data)


def provide_memory_text(provider, item: Dict) -> str:
    request = MemoryRequest(
        query=str(item.get("question", "")),
        context=f"Question date: {item.get('question_date', 'unknown')} | Type: {item.get('question_type', 'unknown')}",
        status=MemoryStatus.BEGIN,
        additional_params={"question_type": item.get("question_type")},
    )
    response = provider.provide_memory(request)
    if not response.memories:
        return ""
    return "\n\n".join(str(memory.content) for memory in response.memories if memory.content)


def process_memory_item(
    item: Dict,
    model_config: Dict,
    judge_model: Optional[str],
    item_index: int,
    memory_type_str: str,
    item_storage_root: Path,
) -> Dict:
    task_model = OpenAIServerModel(**model_config)
    task_model.reset_total_counts()
    timer = TaskTimer()
    timer.start()
    provider = None

    try:
        provider = load_memory_provider(memory_type_str, task_model, item_storage_root / f"task_{item_index}")

        if memory_type_str == "modular":
            ingest_modular_sessions(provider, item)
        elif memory_type_str == "siliconfriend":
            ingest_siliconfriend_sessions(provider, item)
        else:
            raise ValueError(f"Unsupported LongMemEval memory provider: {memory_type_str}")

        memory_text = provide_memory_text(provider, item)
        prompt, context_stats = build_memory_answer_prompt(item, memory_text)
        response = task_model(
            [
                {"role": "system", "content": [{"type": "text", "text": MEMORY_SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
        )
        pred_answer = (response.content or "").strip()
        golden_answer = str(item.get("answer", "")).strip()
        judgement = judge_longmemeval_answer(
            question=str(item.get("question", "")),
            golden_answer=golden_answer,
            pred_answer=pred_answer,
            question_type=str(item.get("question_type", "")),
            judge_model=judge_model,
        )
        success = judgement == "correct"

        if memory_type_str == "modular" and provider.manager is not None:
            try:
                post_result = provider.manager.run_post_task(
                    {
                        "task_succeeded": success,
                        "used_unit_ids": list(getattr(provider, "_last_provided_ids", [])),
                        "benchmark": "longmemeval",
                    }
                )
                provider._record_management_results([post_result])
                provider._update_memory_totals()
            except Exception as exc:
                logger.warning("LongMemEval modular post-task management failed: %s", exc)

        result = {
            "item_index": item_index,
            "task_id": item.get("question_id"),
            "question": item.get("question"),
            "question_date": item.get("question_date"),
            "question_type": item.get("question_type"),
            "golden_answer": golden_answer,
            "pred_answer": pred_answer,
            "judgement": judgement,
            "task_score": 1.0 if success else 0.0,
            "success": success,
            "status": "success",
            "answer_session_ids": item.get("answer_session_ids", []),
            "haystack_session_ids": item.get("haystack_session_ids", []),
            "context_stats": context_stats,
            "retrieved_memory_text": memory_text,
            "memory_metrics": capture_memory_metrics(provider),
        }
        timer.stop()
        return enrich_result_with_metrics(result, timer, TokenCounter.from_model(task_model))
    except Exception as exc:
        result = {
            "item_index": item_index,
            "task_id": item.get("question_id"),
            "question": item.get("question"),
            "question_date": item.get("question_date"),
            "question_type": item.get("question_type"),
            "golden_answer": item.get("answer"),
            "pred_answer": None,
            "judgement": "incorrect",
            "task_score": 0.0,
            "success": False,
            "status": "error",
            "error": str(exc),
            "memory_metrics": capture_memory_metrics(provider),
        }
        timer.stop()
        return enrich_result_with_metrics(result, timer, TokenCounter.from_model(task_model))


def process_no_memory_item(
    item: Dict,
    model_config: Dict,
    judge_model: Optional[str],
    item_index: int,
) -> Dict:
    task_model = OpenAIServerModel(**model_config)
    task_model.reset_total_counts()
    timer = TaskTimer()
    timer.start()

    try:
        prompt, context_stats = build_long_context_prompt(item)
        response = task_model(
            [
                {"role": "system", "content": [{"type": "text", "text": FULL_CONTEXT_SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
        )
        pred_answer = (response.content or "").strip()
        golden_answer = str(item.get("answer", "")).strip()
        judgement = judge_longmemeval_answer(
            question=str(item.get("question", "")),
            golden_answer=golden_answer,
            pred_answer=pred_answer,
            question_type=str(item.get("question_type", "")),
            judge_model=judge_model,
        )
        success = judgement == "correct"

        result = {
            "item_index": item_index,
            "task_id": item.get("question_id"),
            "question": item.get("question"),
            "question_date": item.get("question_date"),
            "question_type": item.get("question_type"),
            "golden_answer": golden_answer,
            "pred_answer": pred_answer,
            "judgement": judgement,
            "task_score": 1.0 if success else 0.0,
            "success": success,
            "status": "success",
            "answer_session_ids": item.get("answer_session_ids", []),
            "haystack_session_ids": item.get("haystack_session_ids", []),
            "context_stats": context_stats,
            "memory_metrics": {},
        }
        timer.stop()
        return enrich_result_with_metrics(result, timer, TokenCounter.from_model(task_model))
    except Exception as exc:
        result = {
            "item_index": item_index,
            "task_id": item.get("question_id"),
            "question": item.get("question"),
            "question_date": item.get("question_date"),
            "question_type": item.get("question_type"),
            "golden_answer": item.get("answer"),
            "pred_answer": None,
            "judgement": "incorrect",
            "task_score": 0.0,
            "success": False,
            "status": "error",
            "error": str(exc),
            "memory_metrics": {},
        }
        timer.stop()
        return enrich_result_with_metrics(result, timer, TokenCounter.from_model(task_model))


def select_items(data: List[Dict], args: argparse.Namespace) -> List[Dict]:
    selected = data
    if args.selection_file:
        manifest = load_selection_file(args.selection_file)
        question_ids = set(manifest.get("selected_question_ids", []))
        if not question_ids:
            raise ValueError(f"Selection file has no selected_question_ids: {args.selection_file}")
        before = len(selected)
        selected = [item for item in selected if item.get("question_id") in question_ids]
        logger.info("Selection file applied: %s -> %s/%s items", args.selection_file, len(selected), before)

    if args.question_type:
        question_type = args.question_type.strip().lower()
        before = len(selected)
        selected = [item for item in selected if str(item.get("question_type", "")).strip().lower() == question_type]
        logger.info("Question-type filter applied: %s -> %s/%s", question_type, len(selected), before)

    if args.task_indices:
        indices = parse_task_indices(args.task_indices)
        selected = [selected[i - 1] for i in sorted(indices) if 0 < i <= len(selected)]
        logger.info("Selected %s tasks from indices: %s", len(selected), args.task_indices)
    elif args.sample_num is not None:
        selected = selected[: args.sample_num]
        logger.info("Limited to first %s tasks", args.sample_num)

    return selected


def main(args: argparse.Namespace) -> int:
    random.seed(args.seed)

    if args.memory_provider and args.concurrency != 1:
        raise SystemExit("LongMemEval memory-backed runs require --concurrency 1 so each item gets an isolated memory store.")
    if args.memory_provider and args.shared_memory_provider:
        raise SystemExit("LongMemEval does not support --shared_memory_provider because each item requires an isolated memory store.")

    model_config = {
        "model_id": args.model or os.environ.get("DEFAULT_MODEL", "gpt-5"),
        "custom_role_conversions": {"tool-call": "assistant", "tool-response": "user"},
        "max_completion_tokens": args.token_budget,
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "api_base": os.environ.get("OPENAI_API_BASE", "https://api-vip.codex-for.me/v1"),
    }

    data = load_longmemeval(args.infile)
    logger.info("Loaded %s items from %s", len(data), args.infile)
    data_to_run = select_items(data, args)
    logger.info("Total LongMemEval items to process: %s", len(data_to_run))

    if args.direct_output_dir:
        run_dir = args.direct_output_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        out_dir = os.path.dirname(args.outfile) or "."
        base_name = os.path.splitext(os.path.basename(args.outfile))[0]
        run_dir = create_run_directory(out_dir, base_name, use_timestamp=True)
    logger.info("Run directory: %s", run_dir)

    item_storage_root = Path(run_dir).resolve().parent / "memory_storage"
    if args.memory_provider:
        item_storage_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    file_lock = threading.Lock()

    def safe_write(result: Dict) -> None:
        with file_lock:
            filename = f"{result.get('item_index')}.json" if result.get("item_index") is not None else None
            save_task_result(result, run_dir, filename)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for idx, item in enumerate(data_to_run):
            item_index = int(item.get("_global_index", idx + 1))
            if args.memory_provider:
                futures.append(
                    executor.submit(
                        process_memory_item,
                        item,
                        model_config,
                        args.judge_model,
                        item_index,
                        args.memory_provider,
                        item_storage_root,
                    )
                )
            else:
                futures.append(
                    executor.submit(
                        process_no_memory_item,
                        item,
                        model_config,
                        args.judge_model,
                        item_index,
                    )
                )

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            safe_write(result)
            logger.info(
                "Task done [%s/%s]: %s | judgement=%s | tokens=%s",
                len(results),
                len(data_to_run),
                str(result.get("question", ""))[:80],
                result.get("judgement"),
                result.get("metrics", {}).get("total_tokens", 0),
            )

    results.sort(key=lambda x: int(x.get("item_index", 0)))
    write_jsonl(args.outfile, results)
    logger.info("Results saved to %s", args.outfile)

    report_path = os.path.join(run_dir, "report.txt")
    generate_unified_report(
        results,
        report_path,
        dataset_name="LongMemEval",
        has_levels=True,
        level_key="question_type",
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LongMemEval with no-memory or memory-backed flows")
    parser.add_argument("--infile", type=str, required=True, help="LongMemEval JSON file")
    parser.add_argument("--outfile", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--model", type=str, default=os.environ.get("DEFAULT_MODEL", "gpt-5"))
    parser.add_argument("--judge_model", type=str, default=os.environ.get("DEFAULT_JUDGE_MODEL", "gpt-5"))
    parser.add_argument("--sample_num", type=int, default=None)
    parser.add_argument("--task_indices", type=str, default=None)
    parser.add_argument("--selection_file", type=str, default=None)
    parser.add_argument("--question_type", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--token_budget", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--summary_interval", type=int, default=8)
    parser.add_argument("--prompts_type", type=str, default="default")
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument("--direct_output_dir", type=str, default=None)
    parser.add_argument("--memory_provider", type=str, default=None)
    parser.add_argument("--enable_memory_evolution", action="store_true")
    parser.add_argument("--disable_memory_evolution", action="store_true")
    parser.add_argument("--shared_memory_provider", action="store_true")
    parsed = parser.parse_args()

    if parsed.memory_provider and parsed.memory_provider not in {"modular", "siliconfriend"}:
        raise SystemExit("LongMemEval runner supports only memory_provider=modular or siliconfriend.")

    raise SystemExit(main(parsed))
