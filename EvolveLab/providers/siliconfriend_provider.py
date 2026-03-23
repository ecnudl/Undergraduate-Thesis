"""SiliconFriend memory baseline adapted for GAIA evaluation.

This provider preserves the MemoryBank-SiliconFriend memory JSON schema and a
retrieval-only vector memory flow, but removes its brittle old langchain stack
so it can run inside the current Flash-Searcher environment.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional runtime fallback
    faiss = None

from ..base_memory import BaseMemoryProvider
from ..memory_types import (
    MemoryItem,
    MemoryItemType,
    MemoryRequest,
    MemoryResponse,
    MemoryStatus,
    MemoryType,
    TrajectoryData,
)

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL_MAP = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec-base": "shibing624/text2vec-base-chinese",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "multilingual-mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "minilm-l6": "sentence-transformers/all-MiniLM-L6-v2",
    "minilm-l12": "sentence-transformers/all-MiniLM-L12-v2",
    "multi-qa": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "alephbert": "imvladikon/sentence-transformers-alephbert",
    "sbert-cn": "uer/sbert-base-chinese-nli",
}


def _resolve_model_name(name: str) -> str:
    return _EMBEDDING_MODEL_MAP.get(name, name)


def _load_embedding_model(model_name: str, cache_dir: str, device: str) -> SentenceTransformer:
    os.makedirs(cache_dir, exist_ok=True)
    resolved = _resolve_model_name(model_name)
    local_path = os.path.join(cache_dir, resolved.replace("/", "_"))

    def _make_model(path: str) -> SentenceTransformer:
        return SentenceTransformer(path, device=device)

    try:
        if os.path.exists(local_path) and os.listdir(local_path):
            return _make_model(local_path)
    except Exception as exc:
        logger.warning("Failed to load cached SiliconFriend embedding model: %s", exc)
        if device != "cpu":
            logger.warning("Retrying cached SiliconFriend embedding model on CPU.")
            return SentenceTransformer(local_path, device="cpu")

    model = _make_model(resolved)
    model.save(local_path)
    return model


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vecs / norms


def _strip_memory_prefix(text: str) -> str:
    stripped = text.strip()
    for prefix in (
        "Conversation content on ",
        "The summary of the conversation on ",
        "时间",
        "日期",
    ):
        if stripped.startswith(prefix):
            parts = stripped.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return stripped


class SiliconFriendProvider(BaseMemoryProvider):
    """MemoryBank-SiliconFriend retrieval baseline for GAIA."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(MemoryType.SILICONFRIEND, config or {})

        self.model = self.config.get("model")
        self.store_dir = os.environ.get("SILICONFRIEND_STORE_DIR") or self.config.get(
            "store_dir", "./storage/siliconfriend"
        )
        memory_file = os.environ.get("SILICONFRIEND_MEMORY_FILE") or self.config.get(
            "memory_file", "gaia_memory.json"
        )
        self.user_name = os.environ.get("SILICONFRIEND_USER_NAME") or self.config.get(
            "user_name", "gaia_eval"
        )
        self.top_k = int(
            os.environ.get("SILICONFRIEND_TOP_K") or self.config.get("top_k", 3)
        )
        self.language = os.environ.get("SILICONFRIEND_LANGUAGE") or self.config.get(
            "language", "en"
        )
        self.embedding_model_name = os.environ.get(
            "SILICONFRIEND_EMBEDDING_MODEL"
        ) or self.config.get("embedding_model", "minilm-l6")
        self.embedding_device = os.environ.get(
            "SILICONFRIEND_EMBEDDING_DEVICE"
        ) or self.config.get("embedding_device", "cpu")
        self.embedding_cache_dir = self.config.get(
            "embedding_cache_dir", "./storage/models"
        )
        self.response_mode = os.environ.get(
            "SILICONFRIEND_RESPONSE_MODE"
        ) or self.config.get("response_mode", "trajectory_summary")

        self.db_path = (
            memory_file
            if os.path.isabs(memory_file)
            else os.path.join(self.store_dir, memory_file)
        )

        self.embedding_model: Optional[SentenceTransformer] = None
        self._documents: List[Dict[str, Any]] = []
        self._embeddings: Optional[np.ndarray] = None
        self._faiss_index = None
        self._experiment_metrics: Dict[str, Any] = {}
        self.reset_experiment_metrics()

    def reset_experiment_metrics(self) -> None:
        self._experiment_metrics = {
            "num_memory_units": self._count_memory_units(),
            "num_inserted": 0,
            "num_deduped": 0,
            "num_retrieved": 0,
            "retrieval_calls": 0,
            "management_ops_triggered": 0,
            "graph_nodes": None,
            "graph_edges": None,
        }

    def get_experiment_metrics(self) -> dict:
        metrics = dict(self._experiment_metrics)
        metrics["num_memory_units"] = self._count_memory_units()
        return metrics

    def initialize(self) -> bool:
        try:
            os.makedirs(self.store_dir, exist_ok=True)
            if not os.path.exists(self.db_path):
                self._save_memory_bank({})

            self.embedding_model = _load_embedding_model(
                self.embedding_model_name,
                self.embedding_cache_dir,
                self.embedding_device,
            )
            self._rebuild_index()
            logger.info(
                "SiliconFriendProvider initialized: store_dir=%s user=%s top_k=%s language=%s device=%s",
                self.store_dir,
                self.user_name,
                self.top_k,
                self.language,
                self.embedding_device,
            )
            return True
        except Exception as exc:
            logger.exception("Failed to initialize SiliconFriendProvider: %s", exc)
            return False

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        if request.status != MemoryStatus.BEGIN:
            return MemoryResponse(
                memories=[],
                memory_type=self.memory_type,
                total_count=0,
                request_id=str(uuid.uuid4()),
            )

        self._experiment_metrics["retrieval_calls"] += 1
        self._experiment_metrics["num_retrieved"] = 0

        if self.embedding_model is None or not self._documents:
            return MemoryResponse(
                memories=[],
                memory_type=self.memory_type,
                total_count=0,
                request_id=str(uuid.uuid4()),
            )

        try:
            indices_scores = self._search(request.query)
        except Exception as exc:
            logger.exception("SiliconFriend retrieval failed: %s", exc)
            return MemoryResponse(
                memories=[],
                memory_type=self.memory_type,
                total_count=0,
                request_id=str(uuid.uuid4()),
            )

        memories: List[MemoryItem] = []
        for rank, (idx, score) in enumerate(indices_scores, start=1):
            doc = self._documents[idx]
            content = (
                "[SiliconFriend MemoryBank Retrieval]\n"
                f"Relevant memory {rank} (date: {doc['date']}):\n"
                f"{_strip_memory_prefix(doc['text'])}"
            )
            memories.append(
                MemoryItem(
                    id=str(uuid.uuid4()),
                    content=content,
                    metadata={"date": doc["date"], "rank": rank},
                    score=float(score),
                    type=MemoryItemType.TEXT,
                )
            )

        self._experiment_metrics["num_retrieved"] = len(memories)
        return MemoryResponse(
            memories=memories,
            memory_type=self.memory_type,
            total_count=len(memories),
            request_id=str(uuid.uuid4()),
        )

    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        try:
            memory_bank = self._load_memory_bank()
            user_memory = memory_bank.setdefault(self.user_name, {"name": self.user_name})
            history = user_memory.setdefault("history", {})
            date_key = datetime.utcnow().strftime("%Y-%m-%d")
            history.setdefault(date_key, [])

            memory_response = self._format_memory_response(trajectory_data)
            history[date_key].append(
                {"query": trajectory_data.query, "response": memory_response}
            )

            self._save_memory_bank(memory_bank)
            self._rebuild_index()
            self._experiment_metrics["num_inserted"] += 1
            self._experiment_metrics["num_memory_units"] = self._count_memory_units()
            return True, memory_response
        except Exception as exc:
            logger.exception("SiliconFriend take_in_memory failed: %s", exc)
            return False, f"Failed to add memory: {exc}"

    def _format_memory_response(self, trajectory_data: TrajectoryData) -> str:
        if self.response_mode == "final_answer":
            return f"Task answer: {trajectory_data.result}"

        parts = [
            f"Task: {trajectory_data.query}",
            f"Final answer: {trajectory_data.result}",
        ]
        if trajectory_data.metadata:
            is_correct = trajectory_data.metadata.get("is_correct")
            if is_correct is not None:
                parts.append(f"Outcome: {'correct' if is_correct else 'incorrect'}")

        steps: List[str] = []
        for idx, step in enumerate(trajectory_data.trajectory or []):
            name = step.get("name") or step.get("type") or "step"
            if name == "plan":
                value = step.get("value", "")
                if value:
                    steps.append(f"Step {idx} plan: {str(value)[:300]}")
            elif name == "action":
                tool_calls = step.get("tool_calls", [])
                obs = step.get("obs", "")
                tc_names = ", ".join(tc.get("name", "tool") for tc in tool_calls[:3])
                text = f"Step {idx} action"
                if tc_names:
                    text += f" tools={tc_names}"
                if obs:
                    text += f" obs={str(obs)[:300]}"
                steps.append(text)
            elif name == "summary":
                value = step.get("value", "")
                if value:
                    steps.append(f"Step {idx} summary: {str(value)[:300]}")
            if len(steps) >= 4:
                break

        if steps:
            parts.append("Key trajectory:")
            parts.extend(steps)

        return "\n".join(part for part in parts if part)

    def _load_memory_bank(self) -> Dict[str, Any]:
        if not os.path.exists(self.db_path):
            return {}
        with open(self.db_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_memory_bank(self, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _count_memory_units(self) -> int:
        try:
            memory_bank = self._load_memory_bank()
        except Exception:
            return 0
        user_memory = memory_bank.get(self.user_name, {})
        history = user_memory.get("history", {})
        return sum(len(items) for items in history.values())

    def _iter_memory_docs(self) -> List[Dict[str, Any]]:
        memory_bank = self._load_memory_bank()
        user_memory = memory_bank.get(self.user_name, {})
        history = user_memory.get("history", {})
        documents: List[Dict[str, Any]] = []

        for date in sorted(history.keys()):
            content = history[date]
            for dialog in content:
                query = str(dialog.get("query", "")).strip()
                response = str(dialog.get("response", "")).strip()
                if self.language == "cn":
                    text = f"时间{date}的对话内容：[|用户|]：{query}; [|AI恋人|]：{response}"
                else:
                    text = f"Conversation content on {date}: [|User|]: {query}; [|AI|]: {response}"
                documents.append({"date": date, "text": text})

            summaries = user_memory.get("summary", {})
            if date in summaries:
                summary = str(summaries[date]).strip()
                if self.language == "cn":
                    text = f"时间{date}的对话总结为：{summary}"
                else:
                    text = f"The summary of the conversation on {date} is: {summary}"
                documents.append({"date": date, "text": text})

        return documents

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if self.embedding_model is None or not texts:
            return np.empty((0, 0), dtype=np.float32)
        vecs = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        vecs = np.asarray(vecs, dtype=np.float32)
        return _normalize(vecs)

    def _rebuild_index(self) -> None:
        self._documents = self._iter_memory_docs()
        if not self._documents:
            self._embeddings = None
            self._faiss_index = None
            return

        self._embeddings = self._embed_texts([doc["text"] for doc in self._documents])
        if self._embeddings.size == 0:
            self._faiss_index = None
            return

        if faiss is not None:
            index = faiss.IndexFlatIP(self._embeddings.shape[1])
            index.add(self._embeddings)
            self._faiss_index = index
        else:
            self._faiss_index = None

    def _search(self, query: str) -> List[Tuple[int, float]]:
        if self._embeddings is None or self._embeddings.size == 0:
            return []

        qvec = self._embed_texts([query])
        if qvec.size == 0:
            return []

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(qvec, min(self.top_k, len(self._documents)))
            results: List[Tuple[int, float]] = []
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0:
                    results.append((int(idx), float(score)))
            return results

        sims = np.dot(self._embeddings, qvec[0])
        top_indices = np.argsort(-sims)[: min(self.top_k, len(self._documents))]
        return [(int(idx), float(sims[idx])) for idx in top_indices]
