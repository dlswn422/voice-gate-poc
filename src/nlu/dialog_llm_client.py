from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Tuple

from src.nlu.intent_schema import Intent
from src.rag.manual_rag import ManualRAG


# ==================================================
# 환경/정책
# ==================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("DIALOG_MODEL", os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))

STRICT_SLOT_GATING = os.getenv("STRICT_SLOT_GATING", "1").strip().lower() in ("1", "true", "yes")
USE_LLM_FOR_SOLVE = os.getenv("USE_LLM_FOR_SOLVE", "0").strip().lower() in ("1", "true", "yes")
DEFAULT_HARD_TURN_LIMIT = int(os.getenv("SECOND_STAGE_HARD_TURN_LIMIT", "6") or 6)

FAREWELL_TEXT = "이용해 주셔서 감사합니다. 안녕히 가세요."


# ==================================================
# 슬롯 정의 / 필수 슬롯
# ==================================================
SLOT_KEYS = ["symptom", "where", "when", "error_message", "attempted", "card_or_device"]

REQUIRED_SLOTS_BY_INTENT: Dict[str, List[str]] = {
    "PAYMENT": ["where", "symptom"],
    "EXIT": ["where", "symptom"],
    "ENTRY": ["where", "symptom"],
    "REGISTRATION": ["where", "symptom"],
    "TIME_PRICE": ["where", "symptom"],
    "FACILITY": ["where", "symptom"],
    "COMPLAINT": ["symptom"],
    "NONE": ["symptom"],
}


def _norm_intent_name(x: Any) -> str:
    if not x:
        return "NONE"
    s = str(x).strip().upper()
    if s.startswith("INTENT."):
        s = s.split(".", 1)[-1]
    return s if s in REQUIRED_SLOTS_BY_INTENT else "NONE"


def _merge_slots(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(dst or {})
    if not isinstance(src, dict):
        return out
    for k in SLOT_KEYS:
        if k in src:
            v = src.get(k)
            if isinstance(v, str) and not v.strip():
                v = None
            out[k] = v
    return out


def _missing_required_slots(intent_name: str, slots: Dict[str, Any]) -> List[str]:
    req = REQUIRED_SLOTS_BY_INTENT.get(_norm_intent_name(intent_name), ["symptom"])
    miss = []
    for k in req:
        v = (slots or {}).get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            miss.append(k)
    return miss


# ==================================================
# 종료 발화 감지
# ==================================================
DONE_KEYWORDS = ["됐어요", "되었습니다", "해결", "괜찮아요", "그만", "종료", "끝", "마칠게", "고마워", "감사", "안녕"]


def _normalize(text: str) -> str:
    return re.sub(r"[\s\.\,\!\?]+", "", (text or "").strip().lower())


def _is_done_utterance(text: str) -> bool:
    t = _normalize(text)
    return any(_normalize(k) in t for k in DONE_KEYWORDS)


# ==================================================
# 질문 템플릿(슬롯별)
# ==================================================
def _question_for_missing_slot(intent_name: str, slot: str) -> str:
    intent = _norm_intent_name(intent_name)
    if slot == "where":
        if intent == "PAYMENT":
            return "결제를 어디에서 진행하셨나요? (출구 정산기/사전 정산기(키오스크)/모바일·앱·QR)"
        return "문제가 발생한 위치/기기가 어디인가요? (출구/키오스크/차단기 등)"
    if slot == "symptom":
        if intent == "PAYMENT":
            return "결제에서 어떤 현상이 발생하시나요? (승인 실패/카드 인식 불가/결제 버튼 무반응/정산 반영 문제)"
        return "어떤 현상이 문제인가요? (예: 안 됨/오류 문구/무반응 등)"
    if slot == "error_message":
        return "화면에 표시되는 오류 문구가 있나요? 그대로 읽어주실 수 있을까요?"
    if slot == "attempted":
        return "이미 시도해 보신 조치가 있나요? (재시도/다른 카드/재부팅 등)"
    if slot == "card_or_device":
        return "어떤 결제수단(카드/모바일/QR) 또는 어떤 기기에서 문제가 발생했나요?"
    if slot == "when":
        return "언제부터 문제가 발생했나요? (방금/오늘/어제부터 등)"
    return "조금 더 자세히 말씀해 주실 수 있을까요?"


def _first_clarify_question(intent_name: str) -> str:
    intent = _norm_intent_name(intent_name)
    if intent == "PAYMENT":
        return "결제 문제 중 정확히 어떤 현상이신가요?"
    if intent in ("EXIT", "ENTRY"):
        return "차단기 문제 중 어떤 상황인지 먼저 확인해볼게요. 어떤 현상이신가요?"
    if intent == "REGISTRATION":
        return "등록에서 어떤 단계에서 막히셨나요?"
    if intent == "TIME_PRICE":
        return "시간/요금 중 어떤 부분이 궁금하신가요?"
    if intent == "FACILITY":
        return "기기에서 어떤 문제가 발생하셨나요?"
    return "어떤 도움을 원하시는지 조금 더 구체적으로 말씀해 주세요."


# ==================================================
# 휴리스틱 슬롯 추출 (STT 오탈자 방어)
# ==================================================
def _heuristic_extract_slots(user_text: str, intent_name: str) -> Dict[str, Any]:
    t = (user_text or "").strip()
    tt = _normalize(t)
    intent = _norm_intent_name(intent_name)

    out: Dict[str, Any] = {}

    # where
    if any(k in tt for k in ["출구", "출구정산기", "출구정산"]):
        out["where"] = "출구 정산기"
    elif any(k in tt for k in ["키오스크", "사전정산", "사전", "정산기"]):
        out["where"] = "사전 정산기(키오스크)"
    elif any(k in tt for k in ["모바일", "앱", "qr", "큐알"]):
        out["where"] = "모바일·앱·QR"

    # symptom
    if intent == "PAYMENT":
        # 승인 실패(승인/증인 오탈자 방어)
        if ("승인" in tt or "증인" in tt) and "실패" in tt:
            out["symptom"] = "승인 실패"
        elif "인식" in tt and ("안" in tt or "불가" in tt):
            out["symptom"] = "카드 인식 불가"
        elif ("버튼" in tt or "터치" in tt) and ("무반응" in tt or "안" in tt):
            out["symptom"] = "결제 버튼 무반응"
        elif "반영" in tt and ("안" in tt or "누락" in tt):
            out["symptom"] = "정산 반영 문제"

    # error_message
    if any(k in t for k in ["에러", "오류", "Error", "error", "코드", "code", ":"]):
        # "승인 실패가 떠요" 같은 문장도 오류 문구로 간주 가능
        out["error_message"] = t

    return out


# ==================================================
# manuals_v2 기반 RAG
# ==================================================
INTENT_TO_DOCS: Dict[str, List[str]] = {
    "PAYMENT": ["payment_card_fail.md", "mobile_payment_qr_issue.md", "network_terminal_down.md", "discount_free_time_issue.md"],
    "TIME_PRICE": ["price_inquiry.md", "discount_free_time_issue.md"],
    "REGISTRATION": ["visit_registration_fail.md"],
    "ENTRY": ["entry_gate_not_open.md", "lpr_mismatch_or_no_entry_record.md"],
    "EXIT": ["exit_gate_not_open.md", "lpr_mismatch_or_no_entry_record.md"],
    "FACILITY": ["barrier_physical_fault.md", "network_terminal_down.md", "failsafe_done.md"],
    "COMPLAINT": [],
    "NONE": [],
}

_rag = ManualRAG()


def _clean_line(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def _is_header_or_tag_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    # markdown header / code fence / tag-like
    if s.startswith("#"):
        return True
    if s.startswith("```") or s.endswith("```"):
        return True
    if s.startswith("tags:") or s.startswith("태그:"):
        return True
    return False


def _extract_section(text: str, section_title_keywords: List[str]) -> str:
    """
    text(청크)에서 특정 섹션(예: '해결 안내 문장 템플릿') 부분만 잘라낸다.
    - 다음 큰 섹션 시작(##, ###, ===, [..]) 또는 빈줄 연속 등을 만나면 종료.
    """
    if not text:
        return ""

    lines = text.splitlines()
    start_idx = -1

    # 섹션 시작 찾기
    for i, ln in enumerate(lines):
        s = _clean_line(ln)
        for kw in section_title_keywords:
            if kw in s:
                start_idx = i + 1
                break
        if start_idx != -1:
            break

    if start_idx == -1:
        return ""

    # 섹션 끝 찾기
    end_idx = len(lines)
    for j in range(start_idx, len(lines)):
        s = _clean_line(lines[j])
        # 다음 섹션의 전형적 시작 패턴
        if s.startswith("#") or s.startswith("[") and s.endswith("]"):
            end_idx = j
            break
        if re.match(r"^(필수 슬롯|추가 슬롯|진단 분기|조치 순서|에스컬레이션|첫 질문)\b", s):
            end_idx = j
            break

    return "\n".join(lines[start_idx:end_idx]).strip()


def _extract_solve_templates(chunk_text: str) -> List[str]:
    """
    ✅ '해결 안내 문장 템플릿' 섹션에서만 답변 템플릿 문장 추출
    """
    sec = _extract_section(chunk_text, ["해결 안내 문장 템플릿", "SOLVE_TEMPLATE"])
    if not sec:
        return []

    out: List[str] = []
    for ln in sec.splitlines():
        s = _clean_line(ln)
        if not s:
            continue
        if _is_header_or_tag_line(s):
            continue

        # bullet / dash / numbered 모두 허용
        s = re.sub(r"^[-•\*]\s+", "", s)
        s = re.sub(r"^\d+[.)]\s+", "", s).strip()
        if not s:
            continue

        # 템플릿은 문장형이므로 너무 짧으면 제외
        if len(s) < 10:
            continue

        out.append(s)

    # 중복 제거
    uniq: List[str] = []
    seen = set()
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def _extract_action_steps(chunk_text: str, limit: int = 5) -> List[str]:
    """
    fallback: '조치 순서' 섹션에서 bullet 문장 추출
    """
    sec = _extract_section(chunk_text, ["조치 순서", "조치", "우선순위"])
    if not sec:
        return []

    out: List[str] = []
    for ln in sec.splitlines():
        s = _clean_line(ln)
        if not s:
            continue
        if _is_header_or_tag_line(s):
            continue
        s = re.sub(r"^[-•\*]\s+", "", s)
        s = re.sub(r"^\d+[.)]\s+", "", s).strip()
        if not s:
            continue
        # 조치는 "하세요/확인/재시도/호출" 같은 행동형만
        if not any(k in s for k in ["하세요", "해 주세요", "확인", "재시도", "호출", "연락", "점검", "대기", "재부팅", "다른"]):
            continue
        if len(s) > 180:
            s = s[:180].rstrip() + "…"
        out.append(s)
        if len(out) >= limit:
            break

    uniq: List[str] = []
    seen = set()
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def _build_manual_guidance(hits: List[Any]) -> Tuple[List[str], List[str]]:
    """
    ✅ SOLVE_TEMPLATE를 최우선으로,
    없으면 조치 순서를 fallback으로 사용.
    """
    solve_templates: List[str] = []
    action_steps: List[str] = []

    for c in hits or []:
        raw = getattr(c, "text", "") or ""
        if not raw:
            continue

        if not solve_templates:
            solve_templates = _extract_solve_templates(raw)

        if not action_steps:
            action_steps = _extract_action_steps(raw, limit=5)

        if solve_templates and action_steps:
            break

    return solve_templates, action_steps


def _compose_solve_reply(intent_name: str, slots: Dict[str, Any], solve_templates: List[str], action_steps: List[str]) -> str:
    intent = _norm_intent_name(intent_name)
    where = (slots or {}).get("where") or ""
    symptom = (slots or {}).get("symptom") or ""

    header = f"확인 결과, **{where}**에서 **{symptom}** 상황으로 보입니다.".strip()
    if not where and symptom:
        header = f"확인 결과, **{symptom}** 상황으로 보입니다."
    if not header.endswith("."):
        header += "."

    body_lines: List[str] = []

    # ✅ 1순위: solve template 1개(상황에 맞는 걸 고르는 건 다음 단계에서 고도화 가능)
    if solve_templates:
        body_lines.append(solve_templates[0])

    # ✅ 2순위: 조치 단계 2~3개
    if action_steps:
        for a in action_steps[:3]:
            body_lines.append(f"- {a}")

    # ✅ fallback
    if not body_lines:
        if intent == "PAYMENT" and ("승인 실패" in symptom):
            body_lines = [
                "(승인 실패) 승인 실패는 카드사/카드 상태 또는 단말 통신 상태 문제일 수 있어요. 카드 방향/IC칩을 확인하고 다른 카드로도 재시도해 주세요.",
                "- 동일하면 관리자 호출을 요청해 단말기/회선 점검을 진행해 주세요.",
            ]
        else:
            body_lines = [
                "현재 정보로는 원인을 단정하기 어려워요. 잠시 후 재시도해 보시고, 동일하면 관리자 호출을 요청해 주세요."
            ]

    tail = "위 안내로 해결되셨나요? 다른 문제가 더 있으시면 말씀해 주세요."

    return "\n".join([header, "", *body_lines, "", tail]).strip()


# ==================================================
# SOLVE 문장 자연화 (옵션)
# ==================================================
SYSTEM_PROMPT_SOLVE_ONLY = """
너는 주차장 무인정산/차단기/출입 시스템 상담원이다.
아래 [DRAFT]를 한국어로 자연스럽게 다듬어라.
- 의미/조치 내용은 바꾸지 말 것
- 불필요하게 길게 늘리지 말 것
- 마지막에 '추가 문제가 있으면 말씀해 주세요'는 유지
출력은 그냥 문장만.
""".strip()


def _llm_polish_solve(draft: str) -> str:
    import requests

    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_SOLVE_ONLY},
            {"role": "user", "content": f"[DRAFT]\n{draft}"},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return ((data.get("message") or {}).get("content", "") or "").strip() or draft


# ==================================================
# 결과 타입
# ==================================================
DialogAction = Literal["ASK", "SOLVE", "DONE", "FAILSAFE", "ESCALATE_DONE"]


@dataclass
class DialogResult:
    reply: str = ""
    action: DialogAction = "ASK"
    confidence: float = 0.5
    slots: Dict[str, Any] = None
    new_intent: Optional[str] = None
    raw: Optional[str] = None

    def __post_init__(self):
        if self.slots is None:
            self.slots = {}


# ==================================================
# 메인 엔트리포인트
# ==================================================
def dialog_llm_chat(
    user_text: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
    context: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> DialogResult:
    if _is_done_utterance(user_text):
        return DialogResult(reply=FAREWELL_TEXT, action="DONE", confidence=1.0)

    ctx = context or {}
    hard_limit = int(ctx.get("hard_turn_limit", DEFAULT_HARD_TURN_LIMIT) or DEFAULT_HARD_TURN_LIMIT)
    turn_count_user = int(ctx.get("turn_count_user", 0) or 0)

    first_intent = _norm_intent_name(ctx.get("first_intent"))
    current_intent = _norm_intent_name(ctx.get("current_intent") or first_intent)

    ctx_slots = ctx.get("slots") or {}
    if not isinstance(ctx_slots, dict):
        ctx_slots = {}

    # 1) 턴 제한(관리자 호출+종료)
    if turn_count_user >= hard_limit:
        return DialogResult(
            reply="여러 번 확인했지만 현재 정보로는 문제 상황을 정확히 특정하기 어렵습니다. 관리자를 호출해 도움을 받아주세요. " + FAREWELL_TEXT,
            action="ESCALATE_DONE",
            confidence=1.0,
            slots=ctx_slots,
        )

    # 2) 휴리스틱 슬롯 선반영
    heuristic_slots = _heuristic_extract_slots(user_text, current_intent)
    merged_pre = _merge_slots(ctx_slots, heuristic_slots)

    # 3) 첫 2차 응답은 무조건 ASK
    if turn_count_user == 0:
        miss0 = _missing_required_slots(current_intent, merged_pre)
        q = _question_for_missing_slot(current_intent, miss0[0]) if miss0 else _first_clarify_question(current_intent)
        return DialogResult(reply=q, action="ASK", confidence=0.8, slots=merged_pre)

    # 4) 필수 슬롯 미충족이면 ASK 강제
    missing = _missing_required_slots(current_intent, merged_pre)
    if STRICT_SLOT_GATING and missing:
        q = _question_for_missing_slot(current_intent, missing[0])
        return DialogResult(reply=q, action="ASK", confidence=0.7, slots=merged_pre)

    # 5) ✅ 필수 슬롯 충족 → SOLVE(매뉴얼 기반)
    solve_templates: List[str] = []
    action_steps: List[str] = []
    try:
        preferred_docs = INTENT_TO_DOCS.get(current_intent, [])
        hits = _rag.retrieve(
            query=f"{current_intent} {merged_pre.get('where') or ''} {merged_pre.get('symptom') or ''} {user_text}",
            preferred_docs=preferred_docs,
            hard_filter=True if preferred_docs else False,
            debug=debug,
        )
        solve_templates, action_steps = _build_manual_guidance(hits or [])
    except Exception as e:
        if debug:
            print(f"[RAG] failed: {e}")

    draft = _compose_solve_reply(current_intent, merged_pre, solve_templates, action_steps)
    if USE_LLM_FOR_SOLVE:
        try:
            draft = _llm_polish_solve(draft)
        except Exception as e:
            if debug:
                print(f"[SOLVE_POLISH] failed: {e}")

    return DialogResult(reply=draft, action="SOLVE", confidence=0.9, slots=merged_pre)
