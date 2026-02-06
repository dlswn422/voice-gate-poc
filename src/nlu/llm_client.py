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
# Ollama Native Chat API 설정 (Intent-1 전용)
# ==================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv(
    "OLLAMA_INTENT_MODEL",
    os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
)
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))

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
# JSON 추출 유틸 (방어적)
# ==================================================
def _extract_json(text: str) -> dict:
    """
    LLM 출력에서 intent JSON을 최대한 안전하게 추출한다.

    허용 케이스:
    - 순수 JSON
    - 코드블록 포함 JSON (```json ... ```)
    - 설명 + JSON
    - JSON이 조금 깨졌지만 intent 키는 존재
    """
    if not text:
        raise ValueError("Empty LLM output")

    t = text.strip()

    # 1) 가장 큰 JSON 블록(첫 '{' ~ 마지막 '}') 시도
    start = t.find("{")
    end = t.rfind("}") + 1
    if start != -1 and end > start:
        cand = t[start:end].strip()
        try:
            return json.loads(cand)
        except Exception:
            pass

    # 2) 가장 첫 JSON 객체(짧은 {...})라도 찾아보기
    m = re.search(r"\{.*?\}", t, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # 3) fallback: intent 키만 강제 추출
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

    입력:
        - STT로 확정된 사용자 발화

    출력:
        - IntentResult(intent, confidence=0.0)

    ⚠️ 주의
    - 이 함수는 절대 해결하지 않는다
    - confidence는 AppEngine에서 계산한다
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
            # 분류는 흔들리면 안 됨
            "temperature": 0.0,
            # JSON 하나만 출력하면 충분
            "num_predict": 32,
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
