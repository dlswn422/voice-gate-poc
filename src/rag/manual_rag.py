# src/rag/manual_rag.py
from __future__ import annotations

import os
import glob
import json
import math
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import requests


DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# ✅ 너 프로젝트에서 실제 사용 중인 기본값(현재 로그 기준)
MANUAL_DIR = os.getenv("MANUAL_DIR", "manuals")
TOP_K = int(os.getenv("RAG_TOP_K", "3"))

# ✅ (추가) 캐시 파일 경로 (환경변수로 오버라이드 가능)
# - manuals/.rag_cache.json 로 저장됨(프로젝트 실행 위치가 어디든 manual_dir 기준으로 잡힘)
RAG_CACHE_PATH = os.getenv("RAG_CACHE_PATH", os.path.join(MANUAL_DIR, ".rag_cache.json"))


@dataclass
class ManualChunk:
    doc_id: str
    chunk_id: str
    text: str
    embedding: List[float]


def _read_all_manual_texts(manual_dir: str) -> List[Tuple[str, str]]:
    """(doc_id, full_text) 리스트"""
    paths = sorted(glob.glob(os.path.join(manual_dir, "*.md")))
    out: List[Tuple[str, str]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            out.append((os.path.basename(p), f.read()))
    return out


def _chunk_text(doc_id: str, text: str) -> List[Tuple[str, str]]:
    """
    아주 단순 chunking:
    - 빈 줄 기준으로 문단을 나눔
    """
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


# ==================================================
# ✅ (추가) 캐시 유틸
# ==================================================

def _fingerprint(embed_model: str, text: str) -> str:
    """
    임베딩은 텍스트(+embed_model)에 의해 결정 → doc_id/chunk_id는 영향 없음
    """
    h = hashlib.sha1()
    h.update(embed_model.encode("utf-8"))
    h.update(b"\n")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _load_cache(cache_path: str, embed_model: str, debug: bool = False) -> Dict[str, List[float]]:
    """
    cache 파일 구조:
    {
      "version": 1,
      "embed_model": "nomic-embed-text",
      "items": { "<fp>": [0.1, 0.2, ...], ... }
    }
    """
    if not os.path.exists(cache_path):
        if debug:
            print(f"[RAG] cache not found: {cache_path}")
        return {}

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if obj.get("version") != 1:
            if debug:
                print("[RAG] cache version mismatch → ignore cache")
            return {}

        if obj.get("embed_model") != embed_model:
            # 임베딩 모델이 바뀌면 캐시 재사용 불가
            if debug:
                print(f"[RAG] embed_model changed(cache={obj.get('embed_model')} now={embed_model}) → ignore cache")
            return {}

        items = obj.get("items", {})
        if not isinstance(items, dict):
            return {}
        # fp -> embedding(list)
        return items

    except Exception as e:
        if debug:
            print(f"[RAG] cache load failed → ignore cache: {e}")
        return {}


def _save_cache(cache_path: str, embed_model: str, items: Dict[str, List[float]], debug: bool = False) -> None:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    tmp_path = cache_path + ".tmp"
    obj = {
        "version": 1,
        "embed_model": embed_model,
        "items": items,
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

    # 원자적 교체(가능한 범위에서)
    os.replace(tmp_path, cache_path)

    if debug:
        print(f"[RAG] cache saved: {cache_path} (items={len(items)})")


class ManualRAG:
    """
    - 메뉴얼(md) → 문단 chunk → 임베딩 → in-memory 검색
    - ✅ 캐시(.rag_cache.json)로 임베딩 재사용
    """

    def __init__(
        self,
        manual_dir: str = MANUAL_DIR,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        cache_path: str = RAG_CACHE_PATH,  # ✅ (추가)
    ):
        self.manual_dir = manual_dir
        self.base_url = base_url
        self.embed_model = embed_model
        self.cache_path = cache_path

        self._chunks: List[ManualChunk] = []
        self._built = False

        # ✅ (추가) 캐시 로딩(최초 1회)
        self._cache_items: Dict[str, List[float]] = {}
        self._cache_loaded = False

    def _ensure_cache_loaded(self, debug: bool = False):
        if self._cache_loaded:
            return
        self._cache_items = _load_cache(self.cache_path, self.embed_model, debug=debug)
        self._cache_loaded = True

    def build(self, debug: bool = False):
        if self._built:
            return

        self._ensure_cache_loaded(debug=debug)

        texts = _read_all_manual_texts(self.manual_dir)
        if debug:
            print(f"[RAG] manual_dir={self.manual_dir} files={len(texts)} embed_model={self.embed_model}")
            print(f"[RAG] cache_path={self.cache_path} cache_items={len(self._cache_items)}")

        chunks: List[ManualChunk] = []

        cache_hit = 0
        cache_miss = 0

        for doc_id, full in texts:
            for chunk_id, chunk_text in _chunk_text(doc_id, full):
                fp = _fingerprint(self.embed_model, chunk_text)

                if fp in self._cache_items:
                    emb = self._cache_items[fp]
                    cache_hit += 1
                else:
                    emb = _ollama_embed(chunk_text, self.base_url, self.embed_model)
                    self._cache_items[fp] = emb
                    cache_miss += 1

                chunks.append(
                    ManualChunk(doc_id=doc_id, chunk_id=chunk_id, text=chunk_text, embedding=emb)
                )

        self._chunks = chunks
        self._built = True

        # ✅ (추가) 캐시 저장(새로 임베딩한 게 있으면)
        if cache_miss > 0:
            _save_cache(self.cache_path, self.embed_model, self._cache_items, debug=debug)

        if debug:
            print(f"[RAG] indexed_chunks={len(self._chunks)} (cache_hit={cache_hit}, cache_miss={cache_miss})")

    def retrieve(self, query: str, top_k: int = TOP_K, debug: bool = False) -> List[ManualChunk]:
        if not self._built:
            self.build(debug=debug)

        q_emb = _ollama_embed(query, self.base_url, self.embed_model)

        # 1) 문서(doc_id)별 최고 점수 계산
        doc_best: Dict[str, float] = {}
        doc_chunks: Dict[str, List[Tuple[float, ManualChunk]]] = {}

        for c in self._chunks:
            s = _cosine(q_emb, c.embedding)
            doc_best[c.doc_id] = max(doc_best.get(c.doc_id, -1.0), s)
            doc_chunks.setdefault(c.doc_id, []).append((s, c))

        # 2) 가장 관련 높은 문서 1개 선택
        best_doc = max(doc_best.items(), key=lambda x: x[1])[0] if doc_best else None
        if not best_doc:
            return []

        # 3) 그 문서 안에서 top_k chunk만 뽑기
        scored = sorted(doc_chunks[best_doc], key=lambda x: x[0], reverse=True)
        hits = [c for _, c in scored[:top_k]]

        if debug:
            print(f"[RAG] best_doc={best_doc}")
            print("[RAG] top hits:")
            for i, c in enumerate(hits, 1):
                print(f"  {i}) {c.doc_id} / {c.chunk_id}")

        return hits

