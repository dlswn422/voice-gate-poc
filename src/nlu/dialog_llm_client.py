from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Tuple

from src.nlu.intent_schema import Intent  # (사용 안 해도 무방, 유지)
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

# ✅ 핵심 수정:
# - REGISTRATION / TIME_PRICE는 "어디에서"가 없어도 해결이 가능하므로 where를 필수에서 제거
REQUIRED_SLOTS_BY_INTENT: Dict[str, List[str]] = {
    "PAYMENT": ["where", "symptom"],
    "EXIT": ["where", "symptom"],
    "ENTRY": ["where", "symptom"],
    "FACILITY": ["where", "symptom"],
    "REGISTRATION": ["symptom"],
    "TIME_PRICE": ["symptom"],
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
# ⚠️ 기존 버그:
# "안됐어요" 안에 "됐어요"가 포함 -> DONE으로 오인
# ✅ 해결: 부정 접두어(안/못/미/덜) 바로 앞이면 종료로 보지 않음 + 강한 종료 문구는 정확 매칭 위주
DONE_KEYWORDS = [
    "해결됐어요", "해결되었습니다",
    "끝", "종료", "그만", "마칠게",
    "고마워", "감사합니다", "감사", "안녕",
    "됐어요", "되었습니다", "해결", "괜찮아요",
]
_NEG_PREFIXES = ("안", "못", "미", "덜")


def _normalize(text: str) -> str:
    return re.sub(r"[\s\.\,\!\?\"“”'’]+", "", (text or "").strip().lower())


def _is_done_utterance(text: str) -> bool:
    t = _normalize(text)
    if not t:
        return False

    # 강한 종료는 단독/짧은 형태 위주
    strong_exact = ("끝", "종료", "그만", "마칠게", "안녕")
    if any(t == _normalize(k) for k in strong_exact):
        return True

    # "감사합니다/고마워"는 짧은 경우만 종료로 인정 (추가 문의가 붙은 긴 문장 방지)
    thanks = ("감사합니다", "감사", "고마워")
    if any(t == _normalize(k) or t.endswith(_normalize(k)) for k in thanks):
        if len(t) <= 12:
            return True

    # 나머지 키워드: 포함되더라도 부정 접두어(안/못/미/덜) 바로 앞이면 제외
    for kw in DONE_KEYWORDS:
        k = _normalize(kw)
        if not k:
            continue
        start = 0
        while True:
            idx = t.find(k, start)
            if idx == -1:
                break
            prev = t[idx - 1] if idx - 1 >= 0 else ""
            if prev in _NEG_PREFIXES:
                start = idx + len(k)
                continue
            # 단독 또는 문장 끝에 가까울 때만 종료로 인정 (오인 최소화)
            if t == k or t.endswith(k):
                return True
            start = idx + len(k)

    return False


# ==================================================
# 질문 템플릿(슬롯별)
# ==================================================
def _question_for_missing_slot(intent_name: str, slot: str) -> str:
    intent = _norm_intent_name(intent_name)

    if slot == "where":
        if intent == "PAYMENT":
            return "결제를 어디에서 진행하셨나요? (출구 정산기/사전 정산기(키오스크)/모바일·앱·QR)"
        if intent in ("EXIT", "ENTRY"):
            return "어느 위치에서 문제가 발생했나요? (출구 차단기/입구 차단기/키오스크 등)"
        if intent == "FACILITY":
            return "어느 기기에서 문제가 발생했나요? (키오스크/정산기/출구 차단기 등)"
        return "문제가 발생한 위치/기기가 어디인가요? (출구/키오스크/차단기 등)"

    if slot == "symptom":
        if intent == "PAYMENT":
            return "결제에서 어떤 현상이 발생하시나요? (승인 실패/카드 인식 불가/결제 버튼 무반응/정산 반영 문제)"
        if intent in ("EXIT", "ENTRY"):
            return "차단기에서 어떤 현상이신가요? (안 열림/무반응/차량 인식 불가/오류 문구)"
        if intent == "FACILITY":
            return "기기에서 어떤 현상이신가요? (먹통/무반응/통신 불가 문구/오류 화면/영수증 미출력)"
        if intent == "REGISTRATION":
            return "등록이 어느 단계에서 안 되나요? (번호 입력/인증/저장/완료 처리/등록 확인 등)"
        if intent == "TIME_PRICE":
            return "요금/시간 중 어떤 문제가 있나요? (무료시간 적용/할인/요금 조회/정산 내역 등)"
        return "어떤 현상이 문제인가요? (예: 안 됨/오류 문구/무반응 등)"

    if slot == "error_message":
        return "화면에 표시되는 오류 문구가 있나요? 그대로 읽어주실 수 있을까요?"
    if slot == "attempted":
        return "이미 시도해 보신 조치가 있나요? (재시도/재부팅/관리자 호출 등)"
    if slot == "card_or_device":
        return "어떤 결제수단(카드/모바일/QR) 또는 어떤 기기에서 문제가 발생했나요?"
    if slot == "when":
        return "언제부터 문제가 발생했나요? (방금/오늘/어제부터 등)"
    return "조금 더 자세히 말씀해 주실 수 있을까요?"


def _first_clarify_question(intent_name: str) -> str:
    intent = _norm_intent_name(intent_name)
    if intent == "PAYMENT":
        return "결제 문제 중 정확히 어떤 현상이신가요?"
    if intent == "EXIT":
        return "출구에서 어떤 문제가 발생하셨나요? (정산 완료 후 안 열림/무반응/오류 문구 등)"
    if intent == "ENTRY":
        return "입구에서 어떤 문제가 발생하셨나요? (차량 인식/차단기 안 열림/무반응 등)"
    if intent == "REGISTRATION":
        return "등록이 어느 단계에서 안 되나요? (번호 입력/인증/저장/완료 처리/등록 확인 등)"
    if intent == "TIME_PRICE":
        return "요금/시간 중 어떤 항목이 문제인가요? (무료시간 적용/할인/요금 조회 등)"
    if intent == "FACILITY":
        return "기기 문제를 확인해볼게요. 어느 기기에서 어떤 현상이신가요?"
    return "어떤 도움을 원하시는지 조금 더 구체적으로 말씀해 주세요."


# ==================================================
# 휴리스틱 슬롯 추출 (STT 오탈자 방어)
# ==================================================
def _heuristic_extract_slots(user_text: str, intent_name: str) -> Dict[str, Any]:
    t = (user_text or "").strip()
    tt = _normalize(t)
    intent = _norm_intent_name(intent_name)

    out: Dict[str, Any] = {}

    # --------------------------------------------------
    # where (정교하게)
    # --------------------------------------------------
    # 키오스크/정산기 우선
    if any(k in tt for k in ["키오스크", "사전정산", "사전", "정산기"]):
        out["where"] = "사전 정산기(키오스크)"
    if any(k in tt for k in ["출구정산기", "출구정산"]):
        out["where"] = "출구 정산기"

    # 출구/입구 차단기 (결제의 출구정산기와 충돌 방지: 이미 출구정산기로 잡혔으면 유지)
    if any(k in tt for k in ["출구차단기", "출구게이트", "출구쪽", "출구에서", "출구"]):
        out["where"] = out.get("where") or "출구 차단기"
    if any(k in tt for k in ["입구차단기", "입구게이트", "입구쪽", "입구에서", "입구"]):
        out["where"] = out.get("where") or "입구 차단기"

    # 모바일/앱/QR
    if any(k in tt for k in ["모바일", "앱", "qr", "큐알"]):
        out["where"] = "모바일·앱·QR"

    # 차단봉/차단바/봉도 차단기 계열
    if any(k in tt for k in ["차단봉", "차단바", "봉"]):
        out["where"] = out.get("where") or "차단기"

    # --------------------------------------------------
    # symptom
    # --------------------------------------------------
    if intent == "PAYMENT":
        if ("승인" in tt or "증인" in tt) and "실패" in tt:
            out["symptom"] = "승인 실패"
        elif ("카드" in tt or "ic" in tt) and ("인식" in tt and ("안" in tt or "불가" in tt)):
            out["symptom"] = "카드 인식 불가"
        elif ("버튼" in tt or "터치" in tt) and ("무반응" in tt or "안" in tt):
            out["symptom"] = "결제 버튼 무반응"
        elif "반영" in tt and ("안" in tt or "누락" in tt):
            out["symptom"] = "정산 반영 문제"
        elif "영수증" in tt and any(k in tt for k in ["안", "미출력", "안나와", "안나옴"]):
            out["symptom"] = "영수증 미출력"

    if intent == "FACILITY":
        # 서버/통신/네트워크
        if any(k in tt for k in ["서버와통신", "통신할수없", "통신불가", "서버연결", "연결할수없", "네트워크", "통신장애", "연결오류"]):
            out["symptom"] = "통신/네트워크 문제"
        # 먹통/무반응
        if any(k in tt for k in ["먹통", "무반응", "멈췄", "작동안", "안됨", "안돼", "안되"]):
            out["symptom"] = out.get("symptom") or "무반응/먹통"
        # 오류 문구
        if any(k in tt for k in ["오류문구", "오류", "에러", "error"]):
            out["symptom"] = out.get("symptom") or "오류 문구 표시"

    if intent in ("EXIT", "ENTRY"):
        if any(k in tt for k in ["안올라", "안올라가", "안올라감"]):
            out["symptom"] = "차단기 안 올라감"
        if any(k in tt for k in ["안열", "안열려", "안열림"]):
            out["symptom"] = out.get("symptom") or "차단기 안 열림"
        if any(k in tt for k in ["무반응", "먹통"]):
            out["symptom"] = out.get("symptom") or "무반응"
        # 정산 완료 + 출구 + 안열림/무반응
        if any(k in tt for k in ["정산", "결제"]) and any(k in tt for k in ["완료", "했", "했어요", "됐", "되었"]):
            if any(k in tt for k in ["출구", "출차"]) and any(k in tt for k in ["안열", "무반응", "안올라"]):
                out["where"] = out.get("where") or "출구 차단기"
                out["symptom"] = "정산 완료 후 출구 차단기 미개방"

    if intent == "TIME_PRICE":
        # 무료/할인/감면 + 적용 안됨
        if any(k in tt for k in ["무료", "할인", "감면"]) and any(k in tt for k in ["적용안", "적용안됐", "적용안되", "미적용", "누락", "안됐", "안되", "안돼"]):
            out["symptom"] = "무료/할인 적용 안 됨"
        elif any(k in tt for k in ["무료", "할인", "감면"]):
            out["symptom"] = out.get("symptom") or "무료/할인 문의"
        elif any(k in tt for k in ["요금", "주차비", "얼마", "금액"]):
            out["symptom"] = out.get("symptom") or "요금 조회"
        elif any(k in tt for k in ["시간", "몇시간", "언제", "주차시간"]):
            out["symptom"] = out.get("symptom") or "주차 시간 조회"

    if intent == "REGISTRATION":
        if any(k in tt for k in ["인증", "문자", "sms", "인증번호"]):
            out["symptom"] = "인증/인증번호 문제"
        elif any(k in tt for k in ["저장", "등록", "완료", "확인"]) and any(k in tt for k in ["안", "실패", "안됨", "안돼", "안되"]):
            out["symptom"] = out.get("symptom") or "등록 저장/완료/확인 실패"
        elif any(k in tt for k in ["번호", "차량번호", "차번"]):
            out["symptom"] = out.get("symptom") or "차량번호 등록 문제"
        elif any(k in tt for k in ["등록안", "등록이안", "등록이안되", "등록안되", "안되는중", "안됨"]):
            out["symptom"] = out.get("symptom") or "등록 진행 불가"

    # error_message: 실제 화면 문구/오류 포함시 저장 (RAG 쿼리 강화)
    if any(k in t for k in ["서버와 통신", "통신", "오류", "에러", "Error", "error", "코드", "code", "문구", "떠", "표시", ":"]):
        out["error_message"] = t

    # ✅ 정책: REGISTRATION/TIME_PRICE는 where를 굳이 요구/유지하지 않음 (혼선 방지)
    if intent in ("REGISTRATION", "TIME_PRICE"):
        out.pop("where", None)

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
    "FACILITY": ["barrier_physical_fault.md", "network_terminal_down.md", "failsafe_done.md", "kiosk_ui_device_issue.md"],
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
    if s.startswith("#"):
        return True
    if s.startswith("```") or s.endswith("```"):
        return True
    if s.startswith("tags:") or s.startswith("태그:"):
        return True
    return False


def _extract_section(text: str, section_title_keywords: List[str]) -> str:
    if not text:
        return ""

    lines = text.splitlines()
    start_idx = -1

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

    end_idx = len(lines)
    for j in range(start_idx, len(lines)):
        s = _clean_line(lines[j])
        if s.startswith("#") or (s.startswith("[") and s.endswith("]")):
            end_idx = j
            break
        if re.match(r"^(필수 슬롯|추가 슬롯|진단 분기|조치 순서|에스컬레이션|첫 질문)\b", s):
            end_idx = j
            break

    return "\n".join(lines[start_idx:end_idx]).strip()


def _extract_solve_templates(chunk_text: str) -> List[str]:
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
        s = re.sub(r"^[-•\*]\s+", "", s)
        s = re.sub(r"^\d+[.)]\s+", "", s).strip()
        if not s:
            continue
        if len(s) < 10:
            continue
        out.append(s)

    uniq: List[str] = []
    seen = set()
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def _extract_action_steps(chunk_text: str, limit: int = 5) -> List[str]:
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

    if solve_templates:
        body_lines.append(solve_templates[0])

    if action_steps:
        for a in action_steps[:3]:
            body_lines.append(f"- {a}")

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

    # ✅ 정책: REGISTRATION/TIME_PRICE는 where를 유지하지 않음(혼선 방지)
    if current_intent in ("REGISTRATION", "TIME_PRICE"):
        merged_pre.pop("where", None)

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
            query=f"{current_intent} {merged_pre.get('where') or ''} {merged_pre.get('symptom') or ''} {merged_pre.get('error_message') or ''} {user_text}",
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
