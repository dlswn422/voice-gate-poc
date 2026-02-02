# src/nlu/llm_client.py
from __future__ import annotations

import json
import os
import re
import traceback
from typing import Optional

import requests

from src.nlu.intent_schema import IntentResult, Intent

# ==================================================
# Ollama 설정 (2차(dialog_llm_client.py)와 동일한 스타일)
# ==================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
# 1차 intent 전용 모델을 분리하고 싶으면 OLLAMA_INTENT_MODEL을 .env에 추가해서 쓰면 됨.
# 없으면 OLLAMA_MODEL(2차용) 값을 그대로 fallback.
OLLAMA_MODEL = os.getenv("OLLAMA_INTENT_MODEL", os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "15"))

_JSON_RE = re.compile(r"\{[\s\S]*\}")

# ==================================================
# 1차 의도 분류용 시스템 프롬프트 (intent만 JSON으로)
# ==================================================
SYSTEM_PROMPT_INTENT = (
    "너는 '주차장 키오스크 CX' 전용 음성 의도 분류 AI다.\n\n"
    "사용자의 발화를 아래 의도 중 하나로 분류하라.\n"
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
    "- '문 열어', '나가야 돼요' 같은 표현은 명령이 아니라 상황 설명으로 본다\n"
    "- 문제 상황인지, 방법 문의인지, 감정 표현인지 구분한다\n"
    "- 입차 문제와 출차 문제를 문맥상 명확히 구분한다\n"
    "- 애매하더라도 반드시 가장 가까운 하나의 의도를 선택한다\n"
    "- 실행 여부나 해결 방법은 절대 판단하지 않는다\n\n"
    "[출력 규칙]\n"
    "- 반드시 JSON만 출력한다\n"
    "- 형식: {\"intent\": \"INTENT_NAME\"}\n"
    "- 다른 문장/설명 금지\n"
)

# ==================================================
# JSON 추출 유틸 (기존 로직 유지)
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
# 1차 의도 분류 (INTENT ONLY) - Ollama HTTP 호출 버전
# ==================================================
def detect_intent_llm(text: str, debug: bool = True) -> IntentResult:
    """
    [역할]
    - 사용자 발화의 '의도(intent)'만 분류한다
    - 실행 판단 ❌
    - confidence 판단 ❌ (AppEngine에서 계산)

    [중요]
    - DB 구조 유지를 위해 confidence는 항상 0.0으로 반환
    """
    if debug:
        print(f"[LLM] (Ollama) Input text: {text}")
        print(f"[LLM] (Ollama) base_url={OLLAMA_BASE_URL} model={OLLAMA_MODEL}")

    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_INTENT},
            {"role": "user", "content": text},
        ],
        "stream": False,
        "options": {
            # 1차 분류는 최대한 흔들리지 않게
            "temperature": 0.0,
            # 길게 말 못 하게 제한 (Ollama에서 모델마다 옵션 지원이 조금 다를 수 있음)
            "num_predict": 32,
        },
    }

    try:
        r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()

        data = r.json()
        content = (data.get("message") or {}).get("content", "") or ""

        if debug:
            print("[LLM] (Ollama) Raw output:")
            print(content)

        obj = _extract_json(content)
        intent_str = str(obj.get("intent", "NONE")).strip()

        try:
            intent = Intent(intent_str)
        except Exception:
            intent = Intent.NONE

        if debug:
            print(f"[LLM] (Ollama) Parsed intent: {intent.name}")

        return IntentResult(intent=intent, confidence=0.0)

    except Exception as e:
        if debug:
            print("[LLM] (Ollama) Inference/Parse error:", repr(e))
            traceback.print_exc()

        return IntentResult(intent=Intent.NONE, confidence=0.0)
