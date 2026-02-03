from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Iterable, Tuple

import requests

from src.nlu.intent_schema import Intent
from src.rag.manual_rag import ManualRAG


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
FAREWELL_TEXT = "이용해 주셔서 감사합니다. 안전운전하세요."


def _normalize(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"[\s\.\,\!\?\u3002\uFF0E\uFF0C\uFF01\uFF1F]+", "", t)
    return t


def _is_done_utterance(text: str) -> bool:
    t = _normalize(text)
    return any(_normalize(k) in t for k in DONE_KEYWORDS)


def _sanitize_reply(reply: str) -> str:
    if not reply:
        return reply
    reply = reply.replace("\r\n", "\n")
    reply = re.sub(r"[ \t]+", " ", reply)
    reply = re.sub(r" *\n *", "\n", reply)
    return reply.strip()


def _strip_markdown_noise(s: str) -> str:
    """
    RAG 컨텍스트에서 '# 제목 - TAG' 같은 헤더가 답변으로 튀는 걸 방지하기 위해
    매뉴얼 chunk 텍스트에서 헤더/구분 라인 등을 제거한다.
    """
    lines = []
    for ln in (s or "").splitlines():
        t = ln.strip()
        if not t:
            continue
        # markdown heading 제거
        if t.startswith("#"):
            continue
        # 구분선류
        if re.fullmatch(r"[-=]{3,}", t):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


def _extract_allowed_actions(hits_text: str, limit: int = 10) -> List[str]:
    """
    매뉴얼 발췌에서 '사용자가 따라할 수 있는 조치 문장' 후보를 뽑는다.
    LLM이 매뉴얼을 참고하도록 강제하는 장치(=답변에 최소 1개 포함 유도).
    """
    if not hits_text:
        return []

    actions: List[str] = []
    seen = set()

    for raw_line in hits_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # 헤더/메타 제거
        if line.startswith("#") or line.startswith("(HIT") or line.startswith("[MANUAL_CONTEXT_"):
            continue

        # bullet/번호/조치: 형태
        if re.match(r"^[-•\*]\s+", line) or re.match(r"^\d+[.)]\s+", line) or re.match(r"^(조치|확인|안내|재시도)\s*[:：]", line):
            cand = re.sub(r"^[-•\*]\s+", "", line)
            cand = re.sub(r"^\d+[.)]\s+", "", cand)
            cand = re.sub(r"^(조치|확인|안내|재시도)\s*[:：]\s*", "", cand)
            cand = cand.strip()
        else:
            # 명령/권고 느낌 문장만
            if len(line) < 8:
                continue
            if any(k in line for k in ["가능", "추정", "같", "의심"]):
                continue
            if not any(k in line for k in ["하세요", "해 주세요", "확인", "점검", "재시도", "문의", "출력", "등록", "재결제", "결제", "버튼"]):
                continue
            cand = line

        cand = _sanitize_reply(cand)
        if not cand:
            continue
        if cand in seen:
            continue
        seen.add(cand)
        actions.append(cand)
        if len(actions) >= limit:
            break

    return actions


SYSTEM_PROMPT = """
너는 '주차장 키오스크 고객센터 상담사'다.

목표:
- 사용자의 상황을 파악하고,
- 아래 [MANUAL_CONTEXT_BEGIN ... END]가 제공되면 그 내용을 "참고"해서
  사용자가 바로 따라할 수 있는 조치 안내를 만든다.

중요 규칙:
1) 한국어로 답한다.
2) 출력은 반드시 JSON 한 개만 출력한다(추가 텍스트/마크다운 금지).
3) 매뉴얼 컨텍스트가 있으면:
   - reply에 [ALLOWED_ACTIONS]에서 최소 1개 이상을 반드시 포함해라.
   - 매뉴얼 제목/헤더(# ...)를 그대로 복사해서 reply로 내보내지 마라.
   - 매뉴얼에 없는 내용(예: 카드번호/CVV 입력 등)을 지어내지 마라.
4) 질문이 필요하면 action="ASK"로 하고 질문은 1개만 한다.
5) 종료/해결 의사면 action="DONE" + 배웅 멘트.
6) suggested_intent는 OPEN_GATE/CLOSE_GATE/NONE 중 하나만 사용한다.
   - 차단기 제어 요청이 명확할 때만 PROPOSE_OPEN/PROPOSE_CLOSE를 사용.
   - 그 외에는 suggested_intent="NONE"로 고정한다.

출력 JSON 스키마:
{
  "reply": "문장",
  "action": "ASK|SOLVE|PROPOSE_OPEN|PROPOSE_CLOSE|DONE|FAILSAFE",
  "suggested_intent": "OPEN_GATE|CLOSE_GATE|NONE",
  "confidence": 0.0~1.0,
  "need_confirmation": true|false,
  "confirm_prompt": null 또는 문자열,
  "slots": {}
}
""".strip()


# ✅ 세션의 첫 intent(ENTRY/EXIT/PAYMENT/...)를 받아
#   그 intent에 매핑된 "문서 후보"만 RAG 하드필터로 검색
INTENT_TO_DOCS: Dict[str, List[str]] = {
    "PAYMENT": ["payment_card_fail.md", "discount_free_time_issue.md"],
    "TIME_PRICE": ["discount_free_time_issue.md", "price_inquiry.md"],
    "REGISTRATION": ["visit_registration_fail.md"],
    "ENTRY": ["entry_gate_not_open.md", "lpr_mismatch_or_no_entry_record.md"],
    "EXIT": ["exit_gate_not_open.md", "lpr_mismatch_or_no_entry_record.md"],
    "FACILITY": ["barrier_physical_fault.md", "network_terminal_down.md", "failsafe_done.md"],
    "COMPLAINT": [],  # complaint는 세션 intent로 들어와도, doc 후보는 사용자 발화로 RAG가 고르도록 (하드필터 X)
    "NONE": [],
}

_rag = ManualRAG()


def _preferred_docs_from_context(context: Optional[Dict[str, Any]]) -> List[str]:
    if not context:
        return []
    first_intent = (context.get("first_intent") or "").strip()
    if not first_intent:
        return []
    return INTENT_TO_DOCS.get(first_intent, [])


def _build_manual_context(
    hits: List[Any],
) -> Tuple[str, List[str]]:
    """
    MANUAL_CONTEXT + ALLOWED_ACTIONS를 함께 구성해서
    모델이 매뉴얼을 '참고'하도록 강제한다.
    """
    if not hits:
        return "", []

    # chunk 원문 합치기(허용 조치 추출용)
    all_text = "\n".join([getattr(c, "text", "") or "" for c in hits])
    allowed = _extract_allowed_actions(all_text, limit=10)

    lines: List[str] = []
    lines.append("[MANUAL_CONTEXT_BEGIN]")
    lines.append("아래는 참고 매뉴얼 발췌다. 이 내용을 참고해서 답하라.")
    lines.append("주의: 제목(# ...)이나 태그를 그대로 복사해 답변으로 내지 말 것.")

    for i, c in enumerate(hits, 1):
        raw = getattr(c, "text", "") or ""
        cleaned = _strip_markdown_noise(raw)
        if not cleaned:
            continue
        lines.append(f"(HIT {i}) doc={c.doc_id} chunk={c.chunk_id}")
        lines.append(cleaned)
        lines.append("")

    lines.append("[ALLOWED_ACTIONS_BEGIN]")
    if allowed:
        for i, a in enumerate(allowed, 1):
            lines.append(f"{i}. {a}")
    else:
        lines.append("NONE")
    lines.append("[ALLOWED_ACTIONS_END]")
    lines.append("[MANUAL_CONTEXT_END]")

    return "\n".join(lines).strip(), allowed


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


def _coerce(obj: Dict[str, Any]) -> Dict[str, Any]:
    reply = _sanitize_reply(str(obj.get("reply", "") or ""))
    action = str(obj.get("action", "ASK") or "ASK").strip().upper()

    if action not in ("ASK", "SOLVE", "PROPOSE_OPEN", "PROPOSE_CLOSE", "DONE", "FAILSAFE"):
        action = "ASK"

    # suggested_intent는 오직 OPEN/CLOSE/NONE만 허용
    suggested = str(obj.get("suggested_intent", "NONE") or "NONE").strip().upper()
    if suggested not in ("OPEN_GATE", "CLOSE_GATE", "NONE"):
        suggested = "NONE"

    conf = obj.get("confidence", 0.5)
    try:
        confidence = float(conf)
    except Exception:
        confidence = 0.5
    confidence = max(0.0, min(confidence, 1.0))

    need_confirmation = bool(obj.get("need_confirmation", False))
    confirm_prompt = obj.get("confirm_prompt", None)
    slots = obj.get("slots", {}) or {}

    # PROPOSE_*가 아니면 confirmation/intent 제거
    if action not in ("PROPOSE_OPEN", "PROPOSE_CLOSE"):
        need_confirmation = False
        confirm_prompt = None
        suggested = "NONE"

    if action == "PROPOSE_OPEN":
        suggested = "OPEN_GATE"
        need_confirmation = True
        if not confirm_prompt:
            confirm_prompt = "차단기를 열까요? (예/아니오)"
    elif action == "PROPOSE_CLOSE":
        suggested = "CLOSE_GATE"
        need_confirmation = True
        if not confirm_prompt:
            confirm_prompt = "차단기를 닫을까요? (예/아니오)"

    # reply가 매뉴얼 헤더처럼 나오면 제거
    if reply.lstrip().startswith("#"):
        reply = ""

    return {
        "reply": reply,
        "action": action,
        "suggested_intent": suggested,
        "confidence": confidence,
        "need_confirmation": need_confirmation,
        "confirm_prompt": confirm_prompt,
        "slots": slots if isinstance(slots, dict) else {},
    }


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
        )

    preferred_docs = _preferred_docs_from_context(context)

    # RAG
    manual_context = ""
    allowed_actions: List[str] = []
    rag_best_doc = None

    try:
        hits = _rag.retrieve(
            user_text,
            preferred_docs=preferred_docs if preferred_docs else None,
            # 세션 intent가 명확하면 그 문서 안에서만 검색(정확도↑)
            hard_filter=True if preferred_docs else False,
            prefer_boost=0.45,
            debug=debug,
        )
        rag_best_doc = _rag.last_best_doc
        manual_context, allowed_actions = _build_manual_context(hits) if hits else ("", [])
    except Exception as e:
        if debug:
            print(f"[RAG] failed: {e}")
        manual_context = ""
        allowed_actions = []

    if debug:
        print(f"[DIALOG] first_intent={(context or {}).get('first_intent')} preferred_docs={preferred_docs}")
        print(f"[DIALOG] manual_context_injected={bool(manual_context)} manual_len={len(manual_context)}")
        print(f"[DIALOG] rag_best_doc={rag_best_doc}")

    # LLM 호출
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
        data = r.json()
        content = (data.get("message") or {}).get("content", "") or ""
    except Exception as e:
        if debug:
            print(f"❌ [DIALOG] Llama 호출 실패: {e}")
        return DialogResult(
            reply="현재 상담 응답을 생성하지 못했어요. 잠시 후 다시 말씀해 주세요.",
            action="FAILSAFE",
            suggested_intent=Intent.NONE,
            confidence=0.0,
        )

    if debug:
        print("🧾 [DIALOG RAW OUTPUT]")
        print(content)

    # 파싱 + 보정
    try:
        obj = _parse_json_only(content)
        obj = _coerce(obj)

        # DONE 재확인
        if obj["action"] == "DONE":
            return DialogResult(
                reply=FAREWELL_TEXT,
                action="DONE",
                suggested_intent=Intent.NONE,
                confidence=1.0,
                need_confirmation=False,
                confirm_prompt=None,
                raw=content,
            )

        # ✅ 매뉴얼이 있는데도 reply가 비었거나 너무 일반적이면(헤더 복붙 방지 후 공백 등)
        #    allowed_actions에서 1개를 최소로 채워준다.
        if manual_context and (not obj["reply"]):
            if allowed_actions:
                obj["reply"] = allowed_actions[0]
                obj["action"] = "SOLVE"
            else:
                obj["reply"] = "화면에 표시되는 오류 문구가 무엇인가요?"
                obj["action"] = "ASK"

        # suggested_intent enum 처리 (Intent 타입은 기존 구조 유지)
        try:
            suggested_intent = Intent(obj["suggested_intent"])
        except Exception:
            suggested_intent = Intent.NONE

        return DialogResult(
            reply=obj["reply"],
            action=obj["action"],  # type: ignore
            suggested_intent=suggested_intent,
            confidence=obj["confidence"],
            need_confirmation=obj["need_confirmation"],
            confirm_prompt=obj["confirm_prompt"],
            slots=obj["slots"],
            raw=content,
        )

    except Exception:
        # JSON 파싱 실패 시 안전 ASK
        return DialogResult(
            reply="확인을 위해 한 가지만 여쭤볼게요. 화면에 표시되는 오류 문구가 무엇인가요?",
            action="ASK",
            suggested_intent=Intent.NONE,
            confidence=0.5,
            raw=content,
        )
