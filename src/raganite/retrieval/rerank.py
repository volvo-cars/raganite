#
# Copyright 2026 Volvo Car Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reorder passages with a LangChain document compressor."""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


def order_by_documents(passages: list[dict[str, Any]], documents: list, *, top_k: int) -> list[dict]:
    """Apply a compressor's order to the rerank window and keep every passage."""
    window = passages[:top_k]
    ordered: list[dict[str, Any]] = []
    seen: set[int] = set()
    for document in documents:
        index = (getattr(document, "metadata", None) or {}).get("raganite_index")
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        if not 0 <= index < len(window) or index in seen:
            continue
        ordered.append(window[index])
        seen.add(index)
    ordered.extend(window[index] for index in range(len(window)) if index not in seen)
    return ordered + passages[len(window) :]


class CompressorReranker:
    """Call ``compress_documents`` and keep the original passage objects."""

    def __init__(self, compressor: Any):
        self._compressor = compressor
        self._lock = threading.Lock()

    def compress(self, query: str, passages: list[dict[str, Any]]) -> list:
        from langchain_core.documents import Document

        documents = [
            Document(
                page_content=str(passage.get("text") or "")[:2000],
                metadata={"raganite_index": index, "title": passage.get("title") or ""},
            )
            for index, passage in enumerate(passages)
        ]
        with self._lock:
            return list(self._compressor.compress_documents(documents, query))


def langchain_cross_encoder(model_name: str, top_n: int) -> Any:
    """Local cross-encoder through LangChain's compressor, not a direct model call."""
    try:
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "Cross-encoder reranking needs the rerank extra: uv sync --extra rerank"
        ) from exc
    logger.warning("Loading cross-encoder %s. The first run downloads its weights.", model_name)
    try:
        model = HuggingFaceCrossEncoder(model_name=model_name)
    except Exception as exc:
        logger.warning("Loading cross-encoder %s failed: %s", model_name, exc)
        raise
    logger.warning("Loaded cross-encoder %s.", model_name)
    return CrossEncoderReranker(model=model, top_n=top_n)


def as_compressor(value: Any) -> Any:
    """Accept a compressor or an object that already exposes ``compress``."""
    if value is None:
        return None
    if hasattr(value, "compress"):
        return value
    if hasattr(value, "compress_documents"):
        return CompressorReranker(value)
    raise TypeError("A passage reranker must implement compress_documents")
