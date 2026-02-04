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

# ✅ CWD에 의존하지 않고, 항상 "src/manuals"를 기본으로 잡는다.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/rag
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))       # .../src
_DEFAULT_MANUAL_DIR = os.path.join(_SRC_DIR, "manuals")         # .../src/manuals

MANUAL_DIR = os.getenv("MANUAL_DIR", _DEFAULT_MANUAL_DIR)
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
        doc_id = os.path.basename(p)
        with open(p, "r", encoding="utf-8") as f:
            out.append((doc_id, f.read()))
    return out


def _split_into_chunks(text: str, max_chars: int = 450) -> List[str]:
    # 빈 줄 기준 문단 분리 후 max_chars로 합치기
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paras:
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= max_chars:
            buf = buf + "\n\n" + p
        else:
            chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


def _ollama_embed(text: str, base_url: str, model: str, timeout_sec: int = 60) -> List[float]:
    """임베딩 호출 (Ollama native + OpenAI-compatible 둘 다 지원)

    - Ollama native: POST {base}/api/embeddings  {model, prompt}
    - OpenAI compat: POST {base}/v1/embeddings   {model, input}

    ✅ /api/embeddings 가 404/405면 /v1/embeddings 로 fallback
    """
    b = (base_url or "").rstrip("/")

    # base_url이 이미 /v1면 OpenAI 호환을 우선 시도
    prefer_openai = b.endswith("/v1") or ("/v1/" in b)

    def _try_ollama() -> List[float]:
        url = b + "/api/embeddings"
        payload = {"model": model, "prompt": text}
        r = requests.post(url, json=payload, timeout=timeout_sec)
        r.raise_for_status()
        data = r.json()
        emb = data.get("embedding")
        if not emb:
            raise ValueError(f"no embedding field in response: {data}")
        return emb

    def _try_openai() -> List[float]:
        url = (b if b.endswith("/v1") else b + "/v1") + "/embeddings"
        payload = {"model": model, "input": text}
        r = requests.post(url, json=payload, timeout=timeout_sec)
        r.raise_for_status()
        data = r.json()
        arr = data.get("data") or []
        if not arr or not isinstance(arr, list) or not arr[0].get("embedding"):
            raise ValueError(f"no data[0].embedding in response: {data}")
        return arr[0]["embedding"]

    tries = [_try_openai, _try_ollama] if prefer_openai else [_try_ollama, _try_openai]
    last_err = None

    for fn in tries:
        try:
            return fn()
        except requests.HTTPError as e:
            last_err = e
            status = getattr(e.response, "status_code", None)
            # 404/405는 엔드포인트 불일치 가능성이 높아서 fallback
            if status in (404, 405):
                continue
            raise
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"embedding request failed for base_url={base_url}: {last_err}")


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
    캐시: <manual_dir>/.rag_cache.json
    """

    def __init__(
        self,
        manual_dir: str = MANUAL_DIR,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        top_k: int = TOP_K,
        cache_filename: str = CACHE_FILENAME,
    ):
        self.manual_dir = manual_dir
        self.base_url = base_url
        self.embed_model = embed_model
        self.top_k = top_k

        # ✅ 캐시 파일은 manual_dir 내부로 고정
        self.cache_path = os.path.join(_abs_dir(self.manual_dir), cache_filename)

        self.chunks: List[ManualChunk] = []
        self.cache: Dict[str, List[float]] = {}
        self.last_best_doc: Optional[str] = None

        self._load_cache()
        self.build()

    def _load_cache(self):
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
        except Exception:
            self.cache = {}

    def _save_cache(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False)
        except Exception:
            pass

    def build(self):
        self.chunks = []
        manual_texts = _read_all_manual_texts(self.manual_dir)

        for doc_id, full_text in manual_texts:
            parts = _split_into_chunks(full_text)
            for i, chunk_text in enumerate(parts):
                chunk_id = str(i)
                key = _hash_key(self.embed_model, doc_id, chunk_id, chunk_text)

                if key in self.cache:
                    emb = self.cache[key]
                else:
                    emb = _ollama_embed(
                        chunk_text,
                        base_url=self.base_url,
                        model=self.embed_model,
                        timeout_sec=60,
                    )
                    self.cache[key] = emb

                self.chunks.append(
                    ManualChunk(doc_id=doc_id, chunk_id=chunk_id, text=chunk_text, embedding=emb)
                )

        self._save_cache()

    def retrieve(
        self,
        query: str,
        *,
        preferred_docs: Optional[List[str]] = None,
        hard_filter: bool = False,
        prefer_boost: float = 0.45,
        debug: bool = False,
    ) -> List[ManualChunk]:
        """
        - preferred_docs: 특정 문서들만 우선/또는 강제
        - hard_filter=True면 preferred_docs 밖은 제외(정확도↑)
        - prefer_boost: preferred_docs에 속한 chunk 점수 가중치
        """
        self.last_best_doc = None

        q_emb = _ollama_embed(
            query,
            base_url=self.base_url,
            model=self.embed_model,
            timeout_sec=60,
        )

        scored: List[Tuple[float, ManualChunk]] = []
        for c in self.chunks:
            if preferred_docs and hard_filter and (c.doc_id not in preferred_docs):
                continue

            score = _cosine(q_emb, c.embedding)

            if preferred_docs and (c.doc_id in preferred_docs):
                score += prefer_boost

            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self.top_k]

        if top:
            self.last_best_doc = top[0][1].doc_id

        if debug:
            print(f"[RAG] manual_dir={_abs_dir(self.manual_dir)} cache={self.cache_path}")
            print(f"[RAG] preferred_docs={preferred_docs} hard_filter={hard_filter} boost={prefer_boost}")
            for s, c in top:
                print(f"[RAG] score={s:.4f} doc={c.doc_id} chunk={c.chunk_id}")

        return [c for _, c in top]
