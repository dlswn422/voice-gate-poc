# src/nlu/dialog_llm_client.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal

import requests

from nlu.intent_schema import Intent
from rag.manual_rag import ManualRAG


# ✅ 로딩되는 파일이 맞는지 확인용
BUILD_ID = "dlg-2026-02-02-manual-brief-v2"
print(f"[DIALOG_MODULE] loaded dialog_llm_client BUILD_ID={BUILD_ID}")


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

# ✅ manuals 폴더 위치(기본: src 실행 기준 "manuals")
MANUAL_DIR = os.getenv("MANUAL_DIR", "manuals")

DONE_KEYWORDS = [
    "됐어요", "되었습니다", "해결", "괜찮아요", "그만", "종료", "끝", "마칠게", "고마워", "감사", "안녕",
    "이제 됐", "됐습니다", "해결됐", "정상", "문제없", "됐어", "다 됐", "이만", "끊을게",
]
FAREWELL_TEXT = "이용해 주셔서 감사합니다. 안전운전하세요."


# ---------------------------
# Utils
# ---------------------------
def _normalize(text: str) -> str:
    t = (text or "").strip().lower()
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

    # 자주 보인 케이스만 보정
    reply = reply.replace("버 튼", "버튼")
    reply = reply.replace("출 력", "출력")
    reply = reply.replace("시 도", "시도")

    return reply.strip()


def _looks_like_question(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    return ("?" in t) or t.endswith("요?") or t.endswith("까요?") or t.endswith("인가요?") or t.endswith("있나요?")


def _is_retry_utterance(user_text: str) -> bool:
    t = _normalize(user_text)
    return any(k in t for k in ["다시", "재시도", "해도안", "했는데안", "계속안", "여전히안", "또안"])


# ---------------------------
# Manual file loading
# ---------------------------
def _load_manual_doc_text(doc_id: str) -> str:
    """
    ✅ RAG chunk가 아니라 '문서 전체'에서 조치 순서를 파싱해야 안정적이다.
    """
    if not doc_id:
        return ""
    path = os.path.join(MANUAL_DIR, doc_id)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


# ---------------------------
# Manual parsing helpers
# ---------------------------
def _extract_section_lines(md_text: str, header: str) -> List[str]:
    """
    md_text에서 "## {header}" 섹션의 라인들을 추출.
    다음 "## "가 나오기 전까지.
    """
    if not md_text:
        return []
    lines = md_text.splitlines()
    out: List[str] = []
    in_section = False
    target = f"## {header}".strip()
    for ln in lines:
        s = ln.strip()
        if s.startswith("## ") and in_section:
            break
        if s == target:
            in_section = True
            continue
        if in_section:
            out.append(ln)
    return out


def _parse_manual_question(md_text: str) -> Optional[str]:
    lines = _extract_section_lines(md_text, "확인 질문(필요 시 1개)")
    for ln in lines:
        s = ln.strip()
        if s.startswith("-"):
            q = s.lstrip("-").strip()
            q = _sanitize_reply(q)
            if q:
                return q
    return None


def _parse_manual_actions(md_text: str) -> List[str]:
    """
    "## 조치 순서(우선순위)"에서 1)~n) 항목을 추출.
    """
    lines = _extract_section_lines(md_text, "조치 순서(우선순위)")
    actions: List[str] = []
    for ln in lines:
        s = ln.strip()
        m = re.match(r"^\d+\)\s*(.+)$", s)
        if m:
            a = _sanitize_reply(m.group(1))
            if a:
                actions.append(a)
    return actions


def _parse_manual_escalation(md_text: str) -> List[str]:
    """
    "## 에스컬레이션(관리자/관제 안내)" bullet 추출
    """
    lines = _extract_section_lines(md_text, "에스컬레이션(관리자/관제 안내)")
    esc: List[str] = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("-"):
            e = _sanitize_reply(s.lstrip("-").strip())
            if e:
                esc.append(e)
    return esc


def _make_manual_brief_from_doc(doc_text: str, doc_id: str) -> Dict[str, Any]:
    q = _parse_manual_question(doc_text)
    steps = _parse_manual_actions(doc_text)
    esc = _parse_manual_escalation(doc_text)

    return {
        "doc": doc_id,
        "question": q,
        "steps": steps[:5],
        "escalation": esc[:3],
    }


def _needs_intent_specific_question(first_intent: str, user_text: str) -> bool:
    t = _normalize(user_text)
    if first_intent == "REGISTRATION_ISSUE":
        if any(k in t for k in ["키오스크", "모바일", "안내데스크"]):
            return False
        return True
    if first_intent == "PAYMENT_ISSUE":
        # "안돼"가 들어가도 질문이 유용한 케이스가 많아서,
        # 너무 빨리 False로 떨어지지 않게 '승인 실패/오류 문구'가 명시된 경우만 False
        if any(k in t for k in ["승인실패", "결제실패", "오류", "실패라고", "실패떠"]):
            return False
        return True
    if first_intent == "ENTRY_FLOW_ISSUE":
        return True
    return False


def _template_reply_from_manual(first_intent: str, manual_brief: Dict[str, Any], user_text: str) -> Dict[str, Any]:
    steps: List[str] = manual_brief.get("steps") or []
    esc: List[str] = manual_brief.get("escalation") or []
    q: Optional[str] = manual_brief.get("question")

    retry = _is_retry_utterance(user_text)

    chosen: List[str] = []
    if steps:
        if retry:
            chosen.extend(steps[:2])
            if len(steps) >= 4:
                chosen.append(steps[3])
            elif len(steps) >= 3:
                chosen.append(steps[2])
        else:
            chosen.extend(steps[:3])

    if len(chosen) < 3 and esc:
        for e in esc:
            if len(chosen) >= 3:
                break
            chosen.append(e)

    if not chosen:
        return {
            "reply": q or "확인을 위해 한 가지만 여쭤볼게요. 화면에 표시되는 오류 문구가 무엇인가요?",
            "action": "ASK",
            "suggested_intent": "NONE",
            "confidence": 0.7,
            "need_confirmation": False,
            "confirm_prompt": None,
            "slots": {},
        }

    add_question = False
    if q and _needs_intent_specific_question(first_intent, user_text):
        add_question = True

    lines = []
    for i, s in enumerate(chosen, 1):
        lines.append(f"{i}) {s}")

    if add_question:
        lines.append(f"추가 확인: {q}")

    return {
        "reply": _sanitize_reply("\n".join(lines)),
        "action": "SOLVE",
        "suggested_intent": "NONE",
        "confidence": 0.85,
        "need_confirmation": False,
        "confirm_prompt": None,
        "slots": {},
    }


# ---------------------------
# RAG / Prompt
# ---------------------------
INTENT_TO_DOCS: Dict[str, List[str]] = {
    "PAYMENT_ISSUE": ["payment_card_fail.md"],
    "REGISTRATION_ISSUE": ["visit_registration_fail.md"],
    "ENTRY_FLOW_ISSUE": ["entry_gate_not_open.md"],
}

_rag = ManualRAG()


def _preferred_docs_from_context(context: Optional[Dict[str, Any]]) -> List[str]:
    if not context:
        return []
    first_intent = (context.get("first_intent") or "").strip()
    if not first_intent:
        return []
    return INTENT_TO_DOCS.get(first_intent, [])


def _manual_context_with_brief(hits: List[Any]) -> (str, Dict[str, Any], str):
    """
    MANUAL_CONTEXT에 원문 발췌를 넣되,
    manual_brief는 ✅ doc 전체 원문에서 파싱한다.
    returns: (manual_context, manual_brief, best_doc_id)
    """
    if not hits:
        return "", {}, ""

    best = hits[0]
    best_doc = getattr(best, "doc_id", "") or "UNKNOWN"

    # ✅ 핵심: doc 전체 원문 로드해서 파싱
    doc_text = _load_manual_doc_text(best_doc)
    if not doc_text:
        # fallback: chunk 텍스트로라도 파싱 시도(최후)
        doc_text = getattr(best, "text", "") or ""

    brief = _make_manual_brief_from_doc(doc_text, best_doc)

    lines: List[str] = []
    lines.append("[MANUAL_CONTEXT_BEGIN]")
    lines.append("아래는 참고 매뉴얼 발췌다. 이 내용을 참고해서 사용자가 바로 따라할 수 있는 조치 안내를 작성하라.")
    for i, c in enumerate(hits, 1):
        lines.append(f"(HIT {i}) doc={c.doc_id} chunk={c.chunk_id}")
        lines.append((c.text or "").strip())
        lines.append("")

    lines.append("[MANUAL_BRIEF_BEGIN]")
    lines.append(json.dumps(brief, ensure_ascii=False))
    lines.append("[MANUAL_BRIEF_END]")
    lines.append("[MANUAL_CONTEXT_END]")
    return "\n".join(lines).strip(), brief, best_doc


SYSTEM_PROMPT = """
너는 '주차장 키오스크 고객센터 상담사'다.

목표:
- 사용자의 문제를 파악하고,
- [MANUAL_CONTEXT_BEGIN ... END]가 있으면 그 내용을 "참고"해서
  사용자가 바로 따라할 수 있는 조치 안내를 만든다.

중요 규칙:
1) 한국어로 답한다.
2) 출력은 반드시 JSON 한 개만 출력한다(추가 텍스트/마크다운 금지).
3) [MANUAL_BRIEF_BEGIN ... END]가 있으면:
   - reply는 사용자 행동 중심으로 3~5개의 조치(번호 목록)로 작성한다.
   - 조치는 추상 표현(예: "문제 확인")만 단독으로 쓰지 말고, 무엇을/어떻게 할지 구체화한다.
   - 매뉴얼에 없는 내용을 단정하지 않는다.
   - 필요할 때만 마지막 줄에 질문 1개를 추가한다(최대 1개).
   - 기본 action="SOLVE"를 우선하되, 정보가 정말 부족하면 action="ASK" + 질문 1개만.
4) 사용자가 "다시 했는데 안돼요/재시도 실패"라면:
   - 매뉴얼의 다음 단계(우선순위 뒤쪽/에스컬레이션)를 포함해 2개 이상 추가 조치를 제시한다.
5) suggested_intent는 OPEN_GATE/CLOSE_GATE/NONE 중 하나만.
   - 차단기 제어 요청이 명확한 경우에만 action=PROPOSE_OPEN/PROPOSE_CLOSE 사용.
6) action이 PROPOSE_*가 아니면 need_confirmation=false, confirm_prompt=null.

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


def _build_messages(
    user_text: str,
    *,
    context: Optional[Dict[str, Any]],
    manual_context: str,
    history: Optional[List[Dict[str, str]]] = None,
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


def _coerce_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    reply = _sanitize_reply(str(obj.get("reply", "")).strip())
    action = str(obj.get("action", "ASK")).strip()
    if action not in ("ASK", "SOLVE", "PROPOSE_OPEN", "PROPOSE_CLOSE", "DONE", "FAILSAFE"):
        action = "ASK"

    suggested = str(obj.get("suggested_intent", "NONE")).strip()
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

    return {
        "reply": reply,
        "action": action,
        "suggested_intent": suggested,
        "confidence": confidence,
        "need_confirmation": need_confirmation,
        "confirm_prompt": confirm_prompt,
        "slots": slots if isinstance(slots, dict) else {},
    }


def _reply_is_too_abstract(reply: str) -> bool:
    if not reply:
        return True
    lines = [ln.strip() for ln in reply.splitlines() if ln.strip()]
    numbered = sum(1 for ln in lines if re.match(r"^\d+\)\s+", ln))
    if numbered >= 2:
        return False
    if any(k in reply for k in ["문제를 확인", "확인해 주세요"]) and not any(
        k in reply for k in ["삽입", "닦", "재시도", "다른 카드", "입력", "정지선", "재진입", "호출 버튼", "안내번호", "채널"]
    ):
        return True
    return False


def dialog_llm_chat(
    user_text: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
    context: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> DialogResult:
    if _is_done_utterance(user_text):
        return DialogResult(reply=FAREWELL_TEXT, action="DONE", suggested_intent=Intent.NONE, confidence=1.0)

    preferred_docs = _preferred_docs_from_context(context)
    first_intent = (context or {}).get("first_intent") or ""

    manual_context = ""
    manual_brief: Dict[str, Any] = {}
    try:
        hits = _rag.retrieve(
            user_text,
            preferred_docs=preferred_docs if preferred_docs else None,
            hard_filter=True if preferred_docs else False,
            prefer_boost=0.45,
            debug=debug,
        )
        manual_context, manual_brief, best_doc = _manual_context_with_brief(hits) if hits else ("", {}, "")
    except Exception as e:
        if debug:
            print(f"[RAG] failed: {e}")
        manual_context, manual_brief = "", {}

    if debug:
        print(f"[DIALOG] BUILD_ID={BUILD_ID}")
        print(f"[DIALOG] manual_context_injected={bool(manual_context)} manual_len={len(manual_context)}")
        if manual_brief:
            print(f"[DIALOG] manual_brief={json.dumps(manual_brief, ensure_ascii=False)}")

    # ✅ 이제 manual_brief는 doc 전체 기반이라 steps가 채워져야 정상
    if not manual_brief or not manual_brief.get("steps"):
        q = manual_brief.get("question") if manual_brief else None
        return DialogResult(
            reply=q or "확인을 위해 한 가지만 여쭤볼게요. 화면에 표시되는 오류 문구가 무엇인가요?",
            action="ASK",
            suggested_intent=Intent.NONE,
            confidence=0.7,
        )

    # LLM 호출
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": _build_messages(user_text, context=context, manual_context=manual_context, history=history),
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
            print(f"❌ [DIALOG] llama call failed: {e}")
        obj = _template_reply_from_manual(first_intent, manual_brief, user_text)
        return DialogResult(
            reply=obj["reply"],
            action=obj["action"],  # type: ignore
            suggested_intent=Intent.NONE,
            confidence=obj["confidence"],
            need_confirmation=False,
            confirm_prompt=None,
            slots={},
            raw=None,
        )

    if debug:
        print("🧾 [DIALOG RAW OUTPUT]")
        print(content)

    # JSON 파싱
    try:
        obj = _parse_json_only(content)
        obj = _coerce_output(obj)
    except Exception:
        obj = _template_reply_from_manual(first_intent, manual_brief, user_text)

    # DONE 강제
    if obj.get("action") == "DONE" or _is_done_utterance(user_text):
        return DialogResult(reply=FAREWELL_TEXT, action="DONE", suggested_intent=Intent.NONE, confidence=1.0, raw=content)

    # 품질 방어
    if obj.get("action") == "ASK" and not _looks_like_question(obj.get("reply", "")):
        obj = _template_reply_from_manual(first_intent, manual_brief, user_text)

    if obj.get("action") == "SOLVE" and _reply_is_too_abstract(obj.get("reply", "")):
        obj = _template_reply_from_manual(first_intent, manual_brief, user_text)

    try:
        suggested_intent = Intent(obj.get("suggested_intent", "NONE"))
    except Exception:
        suggested_intent = Intent.NONE

    return DialogResult(
        reply=obj.get("reply", ""),
        action=obj.get("action", "ASK"),  # type: ignore
        suggested_intent=suggested_intent,
        confidence=float(obj.get("confidence", 0.5)),
        need_confirmation=bool(obj.get("need_confirmation", False)),
        confirm_prompt=obj.get("confirm_prompt", None),
        slots=obj.get("slots", {}) if isinstance(obj.get("slots", {}), dict) else {},
        raw=content,
    )
