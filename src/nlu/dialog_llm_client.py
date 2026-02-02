# src/nlu/dialog_llm_client.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Iterable

import requests

from nlu.intent_schema import Intent
from rag.manual_rag import ManualRAG


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


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "30"))

DONE_KEYWORDS = [
    "됐어요", "되었습니다", "해결", "괜찮아요", "그만", "종료", "끝", "마칠게", "고마워", "감사", "안녕",
    "이제 됐", "됐습니다", "해결됐", "정상", "문제없", "됐어", "다 됐", "이만", "끊을게",
]

FAREWELL_TEXT = "네, 해결되셨다니 다행입니다. 이용해 주셔서 감사합니다. 안전운전하세요."


def _normalize(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"[\s\.\,\!\?\u3002\uFF0E\uFF0C\uFF01\uFF1F]+", "", t)
    return t


def _is_done_utterance(text: str) -> bool:
    t = _normalize(text)
    return any(_normalize(k) in t for k in DONE_KEYWORDS)


SYSTEM_PROMPT = """
너는 '주차장 키오스크 고객센터 상담사'다.

목표:
- 사용자의 상황(결제/입차/출차/등록/네트워크/물리 고장 등)을 파악하고,
- 아래 [MANUAL_CONTEXT_BEGIN ... END]가 제공되면 "그 내용"을 참고해서 실제 조치 방법을 안내한다.

중요 규칙:
1) 한국어로 답한다.
2) 출력은 반드시 JSON만 출력한다. (추가 텍스트/마크다운 금지)
3) 질문이 필요하면 1개만 한다.
4) 매뉴얼 컨텍스트가 있으면:
   - 단순히 "문제 같아요" 처럼 라벨링만 하지 말고,
   - 컨텍스트에 있는 '조치/확인/재시도/안내' 중 최소 1개 이상을 구체적으로 포함해서 답해라.
   - 컨텍스트에 없는 내용은 지어내지 말고, 필요한 정보 1개를 ASK로 질문해라.
5) 사용자가 해결/종료 의사를 밝히면 action="DONE"으로 설정하고 reply는 짧은 배웅으로 마무리한다.
6) 차단기 제어 요청이 명확할 때만:
   - action="PROPOSE_OPEN" 또는 "PROPOSE_CLOSE"
   - suggested_intent는 OPEN_GATE / CLOSE_GATE
   - need_confirmation=true + confirm_prompt 포함

출력 JSON 스키마:
{
  "reply": "사용자에게 보여줄 문장",
  "action": "ASK|SOLVE|PROPOSE_OPEN|PROPOSE_CLOSE|DONE|FAILSAFE",
  "suggested_intent": "OPEN_GATE|CLOSE_GATE|NONE",
  "confidence": 0.0~1.0,
  "need_confirmation": true|false,
  "confirm_prompt": "예/아니오 확인 질문(필요 시)",
  "slots": { ... }
}
""".strip()


# ---------- A안: intent → manuals 파일 후보 매핑 ----------
# ⚠️ 네 프로젝트 실제 파일명 기준으로 맞춰둠 (로그에 나온 10개)
INTENT_TO_DOCS: Dict[str, List[str]] = {
    "PAYMENT_ISSUE": ["payment_card_fail.md", "discount_free_time_issue.md"],
    "PRICE_INQUIRY": ["price_inquiry.md"],
    "TIME_ISSUE": ["discount_free_time_issue.md"],
    "REGISTRATION_ISSUE": ["visit_registration_fail.md"],
    "NETWORK_ISSUE": ["network_terminal_down.md"],
    "ENTRY_FLOW_ISSUE": ["entry_gate_not_open.md", "lpr_mismatch_or_no_entry_record.md"],
    "EXIT_FLOW_ISSUE": ["exit_gate_not_open.md", "lpr_mismatch_or_no_entry_record.md"],
    "BARRIER_PHYSICAL_FAULT": ["barrier_physical_fault.md"],
    # 혹시 HELP_REQUEST 같은 넓은 라벨이 오면 failsafe로
    "HELP_REQUEST": ["failsafe_done.md"],
}

# RAG
_rag = ManualRAG()


def _preferred_docs_from_context(context: Optional[Dict[str, Any]]) -> List[str]:
    if not context:
        return []
    first_intent = (context.get("first_intent") or "").strip()
    if not first_intent:
        return []
    return INTENT_TO_DOCS.get(first_intent, [])


def _build_manual_context(
    user_text: str,
    *,
    preferred_docs: Optional[Iterable[str]] = None,
    hard_filter: bool = True,
    debug: bool = False,
) -> str:
    hits = _rag.retrieve(
        user_text,
        preferred_docs=preferred_docs,
        hard_filter=hard_filter,
        prefer_boost=0.35,
        debug=debug,
    )
    if not hits:
        return ""

    lines: List[str] = []
    lines.append("[MANUAL_CONTEXT_BEGIN]")
    lines.append("아래는 참고 매뉴얼 발췌다. 이 내용을 참고해서 '구체 조치'를 안내하라.")
    for i, c in enumerate(hits, 1):
        lines.append(f"(HIT {i}) doc={c.doc_id} chunk={c.chunk_id}")
        lines.append(c.text.strip())
        lines.append("")
    lines.append("[MANUAL_CONTEXT_END]")
    return "\n".join(lines).strip()


def _build_messages(
    user_text: str,
    *,
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
    # DONE 강제
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

    preferred_docs = _preferred_docs_from_context(context)
    manual_context = _build_manual_context(
        user_text,
        preferred_docs=preferred_docs if preferred_docs else None,
        hard_filter=True if preferred_docs else False,  # first_intent 없으면 전체검색
        debug=debug,
    )

    if debug:
        print(f"[RAG] first_intent={(context or {}).get('first_intent')} preferred_docs={preferred_docs}")
        print(f"[DIALOG] manual_context_injected={bool(manual_context)} manual_len={len(manual_context)}")

    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": _build_messages(user_text, history=history, context=context, manual_context=manual_context),
        "stream": False,
        "options": {"temperature": 0.2},
    }

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
        # suggested_intent는 OPEN/CLOSE/NONE만 허용
        if suggested not in ("OPEN_GATE", "CLOSE_GATE", "NONE"):
            suggested = "NONE"

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

        if action not in ("ASK", "SOLVE", "PROPOSE_OPEN", "PROPOSE_CLOSE", "DONE", "FAILSAFE"):
            action = "ASK"

        # DONE 재강제
        if _is_done_utterance(user_text) or action == "DONE":
            return DialogResult(
                reply=FAREWELL_TEXT,
                action="DONE",
                suggested_intent=Intent.NONE,
                confidence=1.0,
                need_confirmation=False,
                confirm_prompt=None,
                raw=content,
            )

        # PROPOSE_*일 때만 intent 유지
        if action not in ("PROPOSE_OPEN", "PROPOSE_CLOSE"):
            suggested_intent = Intent.NONE

        if action in ("PROPOSE_OPEN", "PROPOSE_CLOSE"):
            need_confirmation = True
            if not confirm_prompt:
                confirm_prompt = "차단기를 실행할까요? (예/아니오)"

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
        return DialogResult(
            reply=content.strip() or "무슨 문제가 있는지 조금 더 자세히 말씀해 주세요.",
            action="ASK",
            suggested_intent=Intent.NONE,
            confidence=0.5,
            raw=content,
        )
