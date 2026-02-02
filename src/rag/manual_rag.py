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
    md → 문단 chunk → 임베딩 → in-memory cosine 검색
    캐시: manuals/.rag_cache.json
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
        prefer_boost: float = 0.45,
        debug: bool = False,
    ) -> List[ManualChunk]:
        """
        preferred_docs:
          - hard_filter=True  -> 후보 문서 안에서만 검색
          - hard_filter=False -> 전체 검색 + 후보 문서에 가산점
        개선:
          - preferred_docs의 "순서"가 의미있도록, 앞쪽 문서일수록 더 큰 가산점을 준다.
            (예: HELP_REQUEST 후보에서 exit_gate_not_open.md가 gate_not_open.md보다 더 잘 1등으로 뜨게)
        """
        if not self._built:
            self.build(debug=debug)

        if not self._chunks:
            self.last_best_doc = None
            if debug:
                print(f"[RAG] no chunks indexed (manual_dir={self.manual_dir}).")
            return []

        if top_k is None:
            top_k = self.top_k

        preferred_list = list(preferred_docs or [])
        preferred_set = set(preferred_list)

        # 순서 기반 doc별 boost 테이블 생성
        # - 앞에 있을수록 boost가 큼
        # - 총합은 prefer_boost 범위 안에서 부드럽게 감소
        doc_boost: Dict[str, float] = {}
        if preferred_list and not hard_filter:
            n = len(preferred_list)
            # 예: n=3이면 weights ~ [1.0, 0.7, 0.4]
            for idx, doc_id in enumerate(preferred_list):
                if n == 1:
                    w = 1.0
                else:
                    # 1.0 → 0.4로 선형 감소 (너무 급격히 줄지 않게)
                    w = 1.0 - (0.6 * (idx / (n - 1)))
                doc_boost[doc_id] = prefer_boost * w

        q_emb = _ollama_embed(query, self.base_url, self.embed_model)

        scored: List[Tuple[float, ManualChunk]] = []
        for c in self._chunks:
            if preferred_set and hard_filter and c.doc_id not in preferred_set:
                continue

            s = _cosine(q_emb, c.embedding)

            if preferred_set and (not hard_filter):
                s += doc_boost.get(c.doc_id, 0.0)

            scored.append((s, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        hits = [c for _, c in scored[:top_k]]

        self.last_best_doc = hits[0].doc_id if hits else None

        if debug:
            if preferred_set:
                print(f"[RAG] preferred_docs={preferred_list} hard_filter={hard_filter} base_prefer_boost={prefer_boost}")
                if doc_boost:
                    print(f"[RAG] doc_boost_table={doc_boost}")
            if self.last_best_doc:
                print(f"[RAG] best_doc={self.last_best_doc}")
            print("[RAG] top hits:")
            for i, c in enumerate(hits, 1):
                print(f"  {i}) {c.doc_id} / {c.chunk_id}")

        return hits
