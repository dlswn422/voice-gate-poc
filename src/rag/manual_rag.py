# src/rag/manual_rag.py
from __future__ import annotations

import os
import glob
import json
import math
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Iterable

import requests


DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# 실행 위치가 src/이든 루트든, MANUAL_DIR은 보통 "manuals"
MANUAL_DIR = os.getenv("MANUAL_DIR", "manuals")
TOP_K = int(os.getenv("RAG_TOP_K", "3"))

CACHE_FILENAME = os.getenv("RAG_CACHE_FILE", ".rag_cache.json")


@dataclass
class ManualChunk:
    doc_id: str
    chunk_id: str
    text: str
    embedding: List[float]


def _abs_dir(path: str) -> str:
    return os.path.abspath(path)


def _read_all_manual_texts(manual_dir: str) -> List[Tuple[str, str]]:
    base = _abs_dir(manual_dir)
    paths = sorted(glob.glob(os.path.join(base, "*.md")))
    out: List[Tuple[str, str]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            out.append((os.path.basename(p), f.read()))
    return out


def _chunk_text(doc_id: str, text: str) -> List[Tuple[str, str]]:
    # 빈 줄 기준 문단 분리
    paras = [t.strip() for t in text.split("\n\n") if t.strip()]
    chunks: List[Tuple[str, str]] = []
    for i, para in enumerate(paras):
        chunk_id = f"{doc_id}::p{i+1}"
        chunks.append((chunk_id, para))
    return chunks


def _ollama_embed(text: str, base_url: str, model: str, timeout_sec: int = 60) -> List[float]:
    url = base_url.rstrip("/") + "/api/embeddings"
    payload = {"model": model, "prompt": text}
    r = requests.post(url, json=payload, timeout=timeout_sec)
    r.raise_for_status()
    data = r.json()
    return data["embedding"]


def _cosine(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _hash_key(embed_model: str, doc_id: str, chunk_id: str, text: str) -> str:
    raw = f"{embed_model}::{doc_id}::{chunk_id}::{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class ManualRAG:
    """
    - md → 문단 chunk → 임베딩 → in-memory cosine 검색
    - 캐시: manuals/.rag_cache.json
    - A안: preferred_docs로 후보 제한 + hard_filter 지원
    """

    def __init__(
        self,
        manual_dir: str = MANUAL_DIR,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        top_k: int = TOP_K,
    ):
        self.manual_dir = manual_dir
        self.base_url = base_url
        self.embed_model = embed_model
        self.top_k = top_k

        self._chunks: List[ManualChunk] = []
        self._built = False

        self._cache_path = os.path.join(_abs_dir(self.manual_dir), CACHE_FILENAME)
        self._cache: Dict[str, List[float]] = {}

        # 마지막 검색 결과(best doc) 추적용
        self.last_best_doc: Optional[str] = None

    def _load_cache(self, debug: bool = False):
        if os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                if debug:
                    print(f"[RAG] cache_path={self._cache_path} cache_items={len(self._cache)}")
            except Exception as e:
                if debug:
                    print(f"[RAG] cache load failed: {e}")
                self._cache = {}
        else:
            if debug:
                print(f"[RAG] cache not found: {self._cache_path}")
            self._cache = {}

    def _save_cache(self, debug: bool = False):
        try:
            os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False)
            if debug:
                print(f"[RAG] cache saved: {self._cache_path} (items={len(self._cache)})")
        except Exception as e:
            if debug:
                print(f"[RAG] cache save failed: {e}")

    def build(self, debug: bool = False):
        if self._built:
            return

        self._load_cache(debug=debug)

        texts = _read_all_manual_texts(self.manual_dir)
        if debug:
            print(f"[RAG] manual_dir={self.manual_dir} files={len(texts)} embed_model={self.embed_model}")

        chunks: List[ManualChunk] = []
        cache_hit = 0
        cache_miss = 0

        for doc_id, full in texts:
            for chunk_id, chunk_text in _chunk_text(doc_id, full):
                key = _hash_key(self.embed_model, doc_id, chunk_id, chunk_text)

                if key in self._cache:
                    emb = self._cache[key]
                    cache_hit += 1
                else:
                    emb = _ollama_embed(chunk_text, self.base_url, self.embed_model)
                    self._cache[key] = emb
                    cache_miss += 1

                chunks.append(ManualChunk(doc_id=doc_id, chunk_id=chunk_id, text=chunk_text, embedding=emb))

        self._chunks = chunks
        self._built = True

        if cache_miss > 0:
            self._save_cache(debug=debug)

        if debug:
            print(f"[RAG] indexed_chunks={len(self._chunks)} (cache_hit={cache_hit}, cache_miss={cache_miss})")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        *,
        preferred_docs: Optional[Iterable[str]] = None,
        hard_filter: bool = False,
        prefer_boost: float = 0.35,
        debug: bool = False,
    ) -> List[ManualChunk]:
        """
        - preferred_docs: 후보 문서 리스트(파일명)
        - hard_filter=True: 후보 문서 안에서만 검색(A안 핵심)
        - hard_filter=False: 전체 검색 + 후보 문서 점수 가산
        """
        if not self._built:
            self.build(debug=debug)

        if top_k is None:
            top_k = self.top_k

        preferred_set = set(preferred_docs or [])
        q_emb = _ollama_embed(query, self.base_url, self.embed_model)

        scored: List[Tuple[float, ManualChunk]] = []
        for c in self._chunks:
            if preferred_set and hard_filter and c.doc_id not in preferred_set:
                continue

            s = _cosine(q_emb, c.embedding)

            if preferred_set and (not hard_filter) and c.doc_id in preferred_set:
                s += prefer_boost

            scored.append((s, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        hits = [c for _, c in scored[:top_k]]

        # best_doc 추적
        self.last_best_doc = hits[0].doc_id if hits else None

        if debug:
            if preferred_set:
                print(f"[RAG] preferred_docs={list(preferred_set)} hard_filter={hard_filter} prefer_boost={prefer_boost}")
            if self.last_best_doc:
                print(f"[RAG] best_doc={self.last_best_doc}")
            print("[RAG] top hits:")
            for i, c in enumerate(hits, 1):
                print(f"  {i}) {c.doc_id} / {c.chunk_id}")

        return hits
