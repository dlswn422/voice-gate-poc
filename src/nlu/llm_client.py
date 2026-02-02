# src/nlu/llm_client.py
from __future__ import annotations

import json
import os
import re
import time
import traceback
import requests

from src.nlu.intent_schema import IntentResult, Intent

# ==================================================
# Ollama Native Chat API 설정 (확정)
# ==================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv(
    "OLLAMA_INTENT_MODEL",
    os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
)
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "20"))

OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"

_JSON_RE = re.compile(r"\{[\s\S]*?\}")

# ==================================================
# 1차 의도 분류 시스템 프롬프트 (INTENT ONLY)
# ==================================================
SYSTEM_PROMPT_INTENT = (
    "너는 '주차장 키오스크 CX' 전용 음성 의도 분류 AI다.\n\n"
    "사용자의 발화를 아래 의도 중 하나로만 분류하라.\n"
    "이 분류는 실행이나 제어 판단이 아니라,\n"
    "사용자가 처한 상황 / 문제 / 문의 유형을 구분하기 위한 것이다.\n\n"
    "[의도 목록]\n"
    "- ENTRY_FLOW_ISSUE\n"
    "- EXIT_FLOW_ISSUE\n"
    "- PAYMENT_ISSUE\n"
    "- REGISTRATION_ISSUE\n"
    "- TIME_ISSUE\n"
    "- PRICE_INQUIRY\n"
    "- HOW_TO_EXIT\n"
    "- HOW_TO_REGISTER\n"
    "- COMPLAINT\n"
    "- NONE\n\n"
    "[분류 규칙]\n"
    "- 명령처럼 보여도 상황 설명으로 본다\n"
    "- 입차 문제와 출차 문제를 문맥으로 구분한다\n"
    "- 애매해도 반드시 하나의 의도만 선택한다\n"
    "- 해결 방법이나 실행 판단은 절대 하지 않는다\n\n"
    "[출력 규칙]\n"
    "- 반드시 JSON만 출력한다\n"
    "- 형식: {\"intent\": \"INTENT_NAME\"}\n"
    "- 다른 텍스트 출력 금지\n"
)

# ==================================================
# JSON 추출 유틸
# ==================================================
def _extract_json(text: str) -> dict:
    text = (text or "").strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return json.loads(text[start:end])

    m = _JSON_RE.search(text)
    if m:
        return json.loads(m.group(0))

    raise ValueError(f"JSON not found in output: {text}")

# ==================================================
# 1차 의도 분류 (INTENT ONLY)
# ==================================================
def detect_intent_llm(text: str, debug: bool = True) -> IntentResult:
    """
    - 1차 intent 분류 전용
    - confidence는 AppEngine에서 계산
    """

    if not text or not text.strip():
        return IntentResult(intent=Intent.NONE, confidence=0.0)

    if debug:
        print(f"[LLM] (Ollama) Input text: {text}")
        print(f"[LLM] (Ollama) base_url={OLLAMA_BASE_URL}")
        print(f"[LLM] (Ollama) model={OLLAMA_MODEL}")

    prompt = (
        SYSTEM_PROMPT_INTENT
        + "\n\n[사용자 발화]\n"
        + text
    )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 32,
        },
    }

    try:
        print("[LLM] ⏳ Intent inference started...")
        start_ts = time.time()

        r = requests.post(
            OLLAMA_CHAT_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        r.raise_for_status()

        elapsed_ms = (time.time() - start_ts) * 1000
        print(f"[LLM] ✅ Intent inference finished ({elapsed_ms:.0f} ms)")

        data = r.json()
        content = data.get("message", {}).get("content", "")

        if debug:
            print("[LLM] (Ollama) Raw output:")
            print(content)

        obj = _extract_json(content)
        intent_str = str(obj.get("intent", "NONE")).strip()

        try:
            intent = Intent(intent_str)
        except Exception:
            intent = Intent.NONE

        print(f"[LLM] 🎯 Parsed intent: {intent.name}")

        return IntentResult(
            intent=intent,
            confidence=0.0,
        )

    except Exception as e:
        print("[LLM] ❌ Intent inference failed")
        if debug:
            print(repr(e))
            traceback.print_exc()

        return IntentResult(
            intent=Intent.NONE,
            confidence=0.0,
        )
