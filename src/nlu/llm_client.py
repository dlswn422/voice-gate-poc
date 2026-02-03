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
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "20"))

OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"


# ==================================================
# 1차 의도 분류 시스템 프롬프트 (LEVEL-1 ONLY)
# ==================================================
SYSTEM_PROMPT_INTENT = (
    "너는 주차장 키오스크 음성 시스템의 1차 의도 분류기다.\n\n"
    "역할:\n"
    "- 사용자의 발화를 '주제 단위(Level-1 Intent)'로만 분류한다\n"
    "- 해결 방법 제시, 실행 판단, 대화 생성은 절대 하지 않는다\n"
    "- HOW_TO, ISSUE, ERROR 같은 세부 원인은 고려하지 않는다\n\n"
    "[의도 목록]\n"
    "- ENTRY        (입차 관련)\n"
    "- EXIT         (출차 관련)\n"
    "- PAYMENT      (요금/결제/정산 관련)\n"
    "- REGISTRATION (방문자/차량 등록 관련)\n"
    "- TIME_PRICE   (시간/요금 정책 문의)\n"
    "- FACILITY     (차단기/기기 이상)\n"
    "- COMPLAINT    (불만/짜증/혼란 표현)\n"
    "- NONE         (주차장과 무관)\n\n"
    "[분류 규칙]\n"
    "- 명령처럼 보여도 '행동'이 아닌 '주제'로 분류한다\n"
    "- 문제 상황과 방법 문의를 구분하지 않는다\n"
    "- 애매해도 반드시 하나의 의도를 선택한다\n\n"
    "[출력 규칙]\n"
    "- 반드시 JSON만 출력한다\n"
    "- 형식: {\"intent\": \"INTENT_NAME\"}\n"
    "- 다른 텍스트는 절대 출력하지 않는다\n"
)

# ==================================================
# 재시도 전용 프롬프트 (Intent.NONE 방지용)
# ==================================================
SYSTEM_PROMPT_INTENT_RETRY = (
    SYSTEM_PROMPT_INTENT
    + "\n\n"
    "⚠️ 주의:\n"
    "아래 발화는 음성 인식 결과라 문장이 불완전하거나 어색할 수 있다.\n"
    "그래도 가장 가까운 의도 하나를 반드시 선택하라.\n"
)


# ==================================================
# JSON 추출 유틸 (방어적)
# ==================================================
def _extract_json(text: str) -> dict:
    """
    LLM 출력에서 intent JSON을 최대한 안전하게 추출한다.
    """
    if not text:
        raise ValueError("Empty LLM output")

    text = text.strip()

    # 1️⃣ 코드블록 제거
    text = re.sub(r"```.*?```", "", text, flags=re.S)

    # 2️⃣ JSON 객체 추출
    m = re.search(r"\{[^{}]*\}", text)
    if m:
        return json.loads(m.group(0))

    # 3️⃣ fallback: intent 키만 강제 추출
    m = re.search(r'"intent"\s*:\s*"([A-Z_]+)"', text)
    if m:
        return {"intent": m.group(1)}

    raise ValueError(f"JSON not found in output: {text}")


# ==================================================
# 내부 호출 함수 (단일 시도)
# ==================================================
def _classify_once(prompt: str, debug: bool) -> Intent:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 16,
        },
    }

    r = requests.post(
        OLLAMA_CHAT_URL,
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )
    r.raise_for_status()

    data = r.json()
    content = data.get("message", {}).get("content", "")

    if debug:
        print("[LLM] (Intent-1) Raw output:")
        print(content)

    obj = _extract_json(content)
    intent_str = str(obj.get("intent", "NONE")).strip()

    try:
        return Intent(intent_str)
    except Exception:
        return Intent.NONE


# ==================================================
# 1차 의도 분류 (INTENT ONLY, retry 1회)
# ==================================================
def detect_intent_llm(text: str, debug: bool = True) -> IntentResult:
    """
    1차(Level-1) 의도 분류 전용 함수

    정책:
    - 1회 시도
    - Intent.NONE이면 프롬프트 변경 후 1회 재시도
    - 그래도 실패하면 NONE 확정
    """

    if not text or not text.strip():
        return IntentResult(intent=Intent.NONE, confidence=0.0)

    if debug:
        print(f"[LLM] (Intent-1) Input text: {text}")
        print(f"[LLM] (Intent-1) model={OLLAMA_MODEL}")

    try:
        print("[LLM] ⏳ Intent-1 inference started...")
        start_ts = time.time()

        # ------------------------------
        # 1차 시도
        # ------------------------------
        prompt = SYSTEM_PROMPT_INTENT + "\n\n[사용자 발화]\n" + text
        intent = _classify_once(prompt, debug)

        # ------------------------------
        # Intent.NONE → 재시도 1회
        # ------------------------------
        if intent == Intent.NONE:
            if debug:
                print("[LLM] 🔁 Intent.NONE → retry once with relaxed prompt")

            retry_prompt = (
                SYSTEM_PROMPT_INTENT_RETRY
                + "\n\n[사용자 발화]\n"
                + text
            )
            intent = _classify_once(retry_prompt, debug)

        elapsed_ms = (time.time() - start_ts) * 1000
        print(f"[LLM] ✅ Intent-1 inference finished ({elapsed_ms:.0f} ms)")
        print(f"[LLM] 🎯 Intent-1 classified: {intent.name}")

        return IntentResult(
            intent=intent,
            confidence=0.0,  # AppEngine에서 계산
        )

    except Exception as e:
        print("[LLM] ❌ Intent-1 inference failed")
        if debug:
            print(repr(e))
            traceback.print_exc()

        return IntentResult(
            intent=Intent.NONE,
            confidence=0.0,
        )
