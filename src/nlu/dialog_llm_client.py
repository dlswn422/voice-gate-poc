"""2차 대화형 LLM (Llama 3.1 Instruct) 클라이언트

기본값은 Ollama 로컬 서버(http://localhost:11434)로 llama3.1:8b를 호출한다.

요구사항
- Ollama 설치
- `ollama pull llama3.1:8b`

이 모듈은 "자유 대화"를 유지하되, 엔진이 상태를 제어할 수 있게
LLM 응답을 JSON으로 받는다.

출력(JSON) 스키마:
{
  "assistant": "사용자에게 보여줄 자연어 응답",
  "state": "CONTINUE" | "END",
  "suggested_intent": "OPEN_GATE" | "CLOSE_GATE" | "NONE",
  "confidence": 0.0~1.0,
  "confirm": true|false,
  "confirm_prompt": "(confirm=true일 때) 실행 전 사용자에게 물을 문장"
}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import requests


DialogState = Literal["CONTINUE", "END"]
SuggestedIntent = Literal["OPEN_GATE", "CLOSE_GATE", "NONE"]


@dataclass
class DialogLLMResponse:
    assistant: str
    state: DialogState = "CONTINUE"
    suggested_intent: SuggestedIntent = "NONE"
    confidence: float = 0.0
    confirm: bool = False
    confirm_prompt: Optional[str] = None


DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("DIALOG_MODEL", "llama3.1:8b")


SYSTEM_PROMPT = (
    "너는 주차장 출입/차단봉 관련 고객센터 상담사다.\n"
    "사용자의 발화를 이해하고, 해결을 위해 필요한 정보를 자연스럽게 질문하고,\n"
    "가능하면 단계별 해결 방법을 제시해라.\n\n"
    "중요 규칙(반드시 지켜):\n"
    "1) 출력은 반드시 JSON만. 다른 텍스트 금지.\n"
    "2) JSON 필드: assistant, state, suggested_intent, confidence, confirm, confirm_prompt\n"
    "3) assistant는 한국어로, 친절하고 간결하게.\n"
    "4) 추가 정보가 필요하면 질문은 한 번에 1개만.\n"
    "5) 사용자가 '열어줘/올려줘'처럼 명시적으로 차단봉 제어를 요청한 경우에만\n"
    "   suggested_intent를 OPEN_GATE 또는 CLOSE_GATE로 설정. 그 외는 NONE.\n"
    "6) suggested_intent가 OPEN_GATE/CLOSE_GATE이면 confirm=true로 설정하고\n"
    "   confirm_prompt에 '차단봉을 올릴까요? 예/아니오로 답해주세요.' 같은 문장을 넣어라.\n"
    "7) 대화가 해결되었거나 더 이상 물을 게 없으면 state='END'로 설정.\n\n"
    "예시(JSON만 출력):\n"
    '{"assistant":"결제가 승인되셨나요? 승인됐다면 어느 출입구인지 알려주세요.",'
    '"state":"CONTINUE","suggested_intent":"NONE","confidence":0.72,'
    '"confirm":false,"confirm_prompt":null}'
)


def _extract_json_object(text: str) -> Dict[str, Any]:
    """모델이 JSON 외 텍스트를 섞어도 최대한 JSON 객체를 복구한다."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return json.loads(text[start : end + 1])


def _coerce_response(data: Dict[str, Any]) -> DialogLLMResponse:
    assistant = str(data.get("assistant", "")).strip()
    state = data.get("state", "CONTINUE")
    if state not in ("CONTINUE", "END"):
        state = "CONTINUE"

    suggested_intent = data.get("suggested_intent", "NONE")
    if suggested_intent not in ("OPEN_GATE", "CLOSE_GATE", "NONE"):
        suggested_intent = "NONE"

    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))

    confirm = bool(data.get("confirm", False))
    confirm_prompt = data.get("confirm_prompt", None)
    if confirm_prompt is not None:
        confirm_prompt = str(confirm_prompt)

    if not assistant:
        assistant = "죄송합니다. 다시 한 번 말씀해 주시겠어요?"

    return DialogLLMResponse(
        assistant=assistant,
        state=state,
        suggested_intent=suggested_intent,
        confidence=confidence,
        confirm=confirm,
        confirm_prompt=confirm_prompt,
    )


def dialog_llm_chat(
    messages: List[Dict[str, str]],
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    model: str = DEFAULT_OLLAMA_MODEL,
    timeout_sec: int = 120,
) -> DialogLLMResponse:
    """Ollama /api/chat 호출.

    messages: [{role: system|user|assistant, content: str}, ...]
    """
    url = base_url.rstrip("/") + "/api/chat"

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        # 자유대화 품질이 흔들리지 않게 temperature는 낮게
        "options": {"temperature": 0.2},
    }

    r = requests.post(url, json=payload, timeout=timeout_sec)
    r.raise_for_status()
    content = r.json().get("message", {}).get("content", "")

    data = _extract_json_object(content)
    return _coerce_response(data)


def make_dialog_system_messages() -> List[Dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]
