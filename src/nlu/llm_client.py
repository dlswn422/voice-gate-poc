from __future__ import annotations

import json
import os
import re
import time
import traceback
import requests

from src.nlu.intent_schema import IntentResult, Intent


# ==================================================
# Ollama Native Chat API 설정 (Intent-1 전용)
# ==================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv(
    "OLLAMA_INTENT_MODEL",
    os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
)

# ⏱ timeout 단축 (tail latency 방지)
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "30"))

OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"


# ==================================================
# 1차 의도 분류 시스템 프롬프트 (경량화 버전)
# ==================================================
SYSTEM_PROMPT_INTENT = (
    "너는 주차장 키오스크 음성 시스템의 1차 의도 분류기다.\n"
    "사용자 발화를 Level-1 주제로만 분류한다.\n\n"
    "의도 목록:\n"
    "ENTRY, EXIT, PAYMENT, REGISTRATION, TIME_PRICE, FACILITY, COMPLAINT, NONE\n\n"
    "규칙:\n"
    "- 해결 방법 제시 금지\n"
    "- 세부 원인 구분 금지\n"
    "- 주제 기준으로만 분류\n\n"
    "출력(JSON only):\n"
    "{\"intent\": \"INTENT_NAME\"}"
)


# ==================================================
# JSON 추출 유틸 (성능 최적화)
# ==================================================
def _extract_json(text: str) -> dict:
    """
    LLM 출력에서 intent JSON을 안전하게 추출한다.
    빠른 경로 → 실패 시 방어 로직
    """
    if not text:
        raise ValueError("Empty LLM output")

    t = text.strip()

    # 🚀 Fast-path: 순수 JSON인 경우 (대부분)
    if t.startswith("{") and t.endswith("}"):
        return json.loads(t)

    # 1) 가장 큰 JSON 블록
    start = t.find("{")
    end = t.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(t[start:end])
        except Exception:
            pass

    # 2) 짧은 JSON 블록
    m = re.search(r"\{.*?\}", t, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # 3) fallback: intent 키만 추출
    m = re.search(r'"intent"\s*:\s*"([A-Z_]+)"', t)
    if m:
        return {"intent": m.group(1)}

    raise ValueError(f"JSON not found in output: {t}")


# ==================================================
# 1차 의도 분류 (INTENT ONLY)
# ==================================================
def detect_intent_llm(text: str, debug: bool = True) -> IntentResult:
    """
    1차(Level-1) 의도 분류 전용 함수

    - 입력: STT 확정 발화
    - 출력: IntentResult(intent, confidence=0.0)

    ⚠️ 주의
    - 이 함수는 절대 해결하지 않는다
    - confidence 계산은 AppEngine 책임
    """
    if not text or not text.strip():
        return IntentResult(intent=Intent.NONE, confidence=0.0)

    if debug:
        print(f"[LLM] (Intent-1) Input text: {text}")
        print(f"[LLM] (Intent-1) model={OLLAMA_MODEL}")

    prompt = SYSTEM_PROMPT_INTENT + "\n\n[사용자 발화]\n" + text

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            # 🔒 분류 안정성 유지
            "temperature": 0.0,

            # 🚀 토큰 생성 최소화
            "num_predict": 12,

            # 🚀 탐색 공간 축소
            "top_k": 20,

            # 🚀 context window 축소
            "num_ctx": 512,
        },
    }

    try:
        if debug:
            print("[LLM] ⏳ Intent-1 inference started...")
        start_ts = time.time()

        r = requests.post(
            OLLAMA_CHAT_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        r.raise_for_status()

        elapsed_ms = (time.time() - start_ts) * 1000
        if debug:
            print(f"[LLM] ✅ Intent-1 inference finished ({elapsed_ms:.0f} ms)")

        data = r.json()
        content = (data.get("message") or {}).get("content", "") or ""

        if debug:
            print("[LLM] (Intent-1) Raw output:")
            print(content)

        obj = _extract_json(content)
        intent_str = str(obj.get("intent", "NONE")).strip()

        try:
            intent = Intent(intent_str)
        except Exception:
            intent = Intent.NONE

        if debug:
            print(f"[LLM] 🎯 Intent-1 classified: {intent.name}")

        return IntentResult(intent=intent, confidence=0.0)

    except Exception as e:
        print("[LLM] ❌ Intent-1 inference failed")
        if debug:
            print(repr(e))
            traceback.print_exc()

        # 실패 시에도 시스템은 멈추지 않는다
        return IntentResult(intent=Intent.NONE, confidence=0.0)
