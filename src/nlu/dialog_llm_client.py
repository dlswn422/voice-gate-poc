"""
2차 대화형 LLM (Llama 3.1 Instruct) 클라이언트 + (B안) Manual RAG

- Ollama 로컬 서버(http://localhost:11434)의 Llama 3.1 모델을 호출
- ManualRAG로 manuals 폴더의 md를 검색해서 "참고 메뉴얼"로 모델에 강제 주입
- 엔진이 파싱하기 쉽도록 JSON 출력 유도
- DONE 키워드 강제 처리 + 배웅 멘트 고정

환경변수(선택):
- OLLAMA_BASE_URL: 기본 "http://localhost:11434"
- OLLAMA_MODEL: 기본 "llama3.1:8b"
- OLLAMA_TIMEOUT: 기본 "30" (초)
- MANUAL_DIR: 기본 "manuals"
- RAG_TOP_K: 기본 "3"
- EMBED_MODEL: 기본 "nomic-embed-text"
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal

import requests

from nlu.intent_schema import Intent
from rag.manual_rag import ManualRAG


# -------------------------
# 반환 스키마
# -------------------------
DialogAction = Literal["ASK", "SOLVE", "PROPOSE_OPEN", "PROPOSE_CLOSE", "DONE", "FAILSAFE"]


@dataclass
class DialogResult:
    reply: str = ""
    action: DialogAction = "ASK"
    suggested_intent: Intent = Intent.NONE
    confidence: float = 0.5
    slots: Dict[str, Any] = None
    need_confirmation: bool = False
    confirm_prompt: Optional[str] = None
    raw: Optional[str] = None

    def __post_init__(self):
        if self.slots is None:
            self.slots = {}


# -------------------------
# Ollama 설정
# -------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "30"))


# -------------------------
# DONE 키워드(이중 안전장치)
# -------------------------
DONE_KEYWORDS = [
    "됐어요", "되었습니다", "해결", "괜찮아요", "그만", "종료", "끝", "마칠게", "고마워", "감사", "안녕",
    "이제 됐", "됐습니다", "해결됐", "정상", "문제없", "됐어", "다 됐", "이만", "끊을게",
]


def _normalize(text: str) -> str:
    t = text.strip().lower()
    # 공백/구두점 제거해서 매칭 안정화
    t = re.sub(r"[\s\.\,\!\?\u3002\uFF0E\uFF0C\uFF01\uFF1F]+", "", t)
    return t


def _is_done_utterance(text: str) -> bool:
    t = _normalize(text)
    return any(_normalize(k) in t for k in DONE_KEYWORDS)


FAREWELL_TEXT = "네, 해결되셨다니 다행입니다. 이용해 주셔서 감사합니다. 안전운전하세요."


# -------------------------
# System Prompt (JSON 강제 + 매뉴얼 근거 강제)
# -------------------------
SYSTEM_PROMPT = """
너는 '주차장 키오스크 고객센터 상담사'다.
사용자의 문제를 파악하고, 제공된 '매뉴얼 컨텍스트'가 있으면 그 내용에 근거해 단계별로 안내한다.

[절대 규칙]
1) 한국어로 답한다.
2) 출력은 반드시 JSON만 출력한다(추가 텍스트/설명/마크다운 금지).
3) 가능하면 질문은 1개만 한다(추가 정보가 꼭 필요할 때만).
4) 사용자가 해결/종료 의사를 밝히면 action="DONE"으로 설정하고, reply는 짧은 배웅 멘트로 마무리한다.
5) 아래에 [MANUAL_CONTEXT_BEGIN ... END]가 주어지면:
   - 그 컨텍스트 '범위 안'에서만 답한다(추측/지어내기 금지).
   - 답변은 우선순위 조치 순서대로 1) 2) 3) 형식으로 안내한다.
   - 컨텍스트에 없는 내용이면, 임의 해결책을 말하지 말고 추가 질문(ASK)으로 전환한다.
6) 차단기 제어 요청이 명확할 때만:
   - action="PROPOSE_OPEN" 또는 "PROPOSE_CLOSE"
   - suggested_intent는 OPEN_GATE / CLOSE_GATE
   - need_confirmation=true + confirm_prompt 반드시 포함

[출력 JSON 스키마]
{
  "reply": "사용자에게 보여줄 문장",
  "action": "ASK|SOLVE|PROPOSE_OPEN|PROPOSE_CLOSE|DONE|FAILSAFE",
  "suggested_intent": "OPEN_GATE|CLOSE_GATE|NONE",
  "confidence": 0.0~1.0,
  "need_confirmation": true|false,
  "confirm_prompt": "예/아니오 확인 질문(필요 시)",
  "slots": { ... }
}

[주의]
- reply는 길게 장황하지 않게. '지금 당장 할 수 있는 조치' 위주.
- 매뉴얼이 있으면 "다시 시도해 보세요" 같은 일반론만 말하지 말고,
  매뉴얼 단계(예: 카드 방향 확인 → IC칩 닦기 → 다른 카드 …)를 따라 구체적으로 안내해라.
""".strip()


# -------------------------
# Manual RAG (B안)
# -------------------------
_rag = ManualRAG()  # env(MANUAL_DIR/RAG_TOP_K/EMBED_MODEL/OLLAMA_BASE_URL)을 manual_rag.py가 읽음


def _build_manual_context(user_text: str, debug: bool = False) -> str:
    """
    RAG로 찾아온 메뉴얼 chunk들을 LLM이 '근거'로 쓰기 좋게 합쳐서 반환
    - 구획(BEGIN/END)으로 감싸서 모델이 컨텍스트 범위를 명확히 인식하게 함
    """
    hits = _rag.retrieve(user_text, debug=debug)
    if not hits:
        return ""

    lines = []
    lines.append("[MANUAL_CONTEXT_BEGIN]")
    lines.append("아래는 참고 메뉴얼 컨텍스트다. 반드시 이 범위 안에서만 답하라.")
    for i, c in enumerate(hits, 1):
        # 문서/청크 메타는 모델이 근거를 "구분"하는 데 도움(사용자에게 그대로 말하라는 뜻 아님)
        lines.append(f"(HIT {i}) doc={c.doc_id} chunk={c.chunk_id}")
        lines.append(c.text.strip())
        lines.append("")  # 가독성용 빈줄
    lines.append("[MANUAL_CONTEXT_END]")
    return "\n".join(lines).strip()


def _build_messages(
    user_text: str,
    history: Optional[List[Dict[str, str]]] = None,
    context: Optional[Dict[str, Any]] = None,
    manual_context: str = "",
) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if context:
        msgs.append({"role": "system", "content": f"context: {json.dumps(context, ensure_ascii=False)}"})

    if manual_context:
        msgs.append({"role": "system", "content": manual_context})

    if history:
        msgs.extend(history)

    msgs.append({"role": "user", "content": user_text})
    return msgs


def _parse_json_only(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("empty response")
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError("no json object found")
    return json.loads(text[start:end])


def dialog_llm_chat(
    user_text: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
    context: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> DialogResult:
    # ✅ DONE 키워드면 LLM 호출 없이 바로 종료 (강제)
    if _is_done_utterance(user_text):
        return DialogResult(
            reply=FAREWELL_TEXT,
            action="DONE",
            suggested_intent=Intent.NONE,
            confidence=1.0,
            need_confirmation=False,
            confirm_prompt=None,
            raw=None,
        )

    manual_context = _build_manual_context(user_text, debug=debug)

    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": _build_messages(user_text, history=history, context=context, manual_context=manual_context),
        "stream": False,
        # 매뉴얼 기반 "정확/일관" 목적 → temperature 낮게
        "options": {"temperature": 0.2},
    }

    # (디버그) 실제로 매뉴얼이 프롬프트에 들어가는지 확인하고 싶을 때
    if debug:
        has_manual = bool(manual_context)
        print(f"[DIALOG] manual_context_injected={has_manual} manual_len={len(manual_context)}")

    try:
        r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        if debug:
            print(f"❌ [DIALOG] Llama 호출 실패: {e}")
        return DialogResult(
            reply="현재 상담 응답을 생성하지 못했어요. 잠시 후 다시 말씀해 주세요.",
            action="FAILSAFE",
            suggested_intent=Intent.NONE,
            confidence=0.0,
        )

    data = r.json()
    content = (data.get("message") or {}).get("content", "") or ""
    if debug:
        print("🧾 [DIALOG RAW OUTPUT]")
        print(content)

    try:
        obj = _parse_json_only(content)

        reply = str(obj.get("reply", "")).strip()
        action = str(obj.get("action", "ASK")).strip()

        suggested = str(obj.get("suggested_intent", "NONE")).strip()
        try:
            suggested_intent = Intent(suggested)
        except Exception:
            suggested_intent = Intent.NONE

        conf = obj.get("confidence", 0.5)
        try:
            confidence = float(conf)
        except Exception:
            confidence = 0.5
        confidence = max(0.0, min(confidence, 1.0))

        need_confirmation = bool(obj.get("need_confirmation", False))
        confirm_prompt = obj.get("confirm_prompt", None)
        slots = obj.get("slots", {}) or {}

        # action 유효성
        if action not in ("ASK", "SOLVE", "PROPOSE_OPEN", "PROPOSE_CLOSE", "DONE", "FAILSAFE"):
            action = "ASK"

        # ✅ DONE 키워드는 파싱 후에도 다시 한번 강제(모델이 규칙을 어길 때 대비)
        if _is_done_utterance(user_text):
            action = "DONE"
            reply = FAREWELL_TEXT
            suggested_intent = Intent.NONE
            need_confirmation = False
            confirm_prompt = None
            confidence = 1.0

        # suggested_intent 보정(제안 액션이 아니면 NONE)
        if action not in ("PROPOSE_OPEN", "PROPOSE_CLOSE"):
            suggested_intent = Intent.NONE

        # confirm_prompt 자동 생성(제안 시)
        if action in ("PROPOSE_OPEN", "PROPOSE_CLOSE"):
            need_confirmation = True
            if not confirm_prompt:
                confirm_prompt = "차단기를 실행할까요? (예/아니오)"

        # ✅ DONE일 때 배웅 문구 고정(모델이 이상한 문구 주는 것 방지)
        if action == "DONE":
            reply = FAREWELL_TEXT
            suggested_intent = Intent.NONE
            need_confirmation = False
            confirm_prompt = None

        if not reply:
            reply = "확인을 위해 한 가지만 더 여쭤볼게요."

        return DialogResult(
            reply=reply,
            action=action,  # type: ignore
            suggested_intent=suggested_intent,
            confidence=confidence,
            slots=slots if isinstance(slots, dict) else {},
            need_confirmation=need_confirmation,
            confirm_prompt=confirm_prompt,
            raw=content,
        )

    except Exception:
        # JSON 파싱 실패 시
        return DialogResult(
            reply=content.strip() or "무슨 문제가 있는지 조금 더 자세히 말씀해 주세요.",
            action="ASK",
            suggested_intent=Intent.NONE,
            confidence=0.5,
            raw=content,
        )
