from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Tuple

import requests

from src.nlu.intent_schema import Intent
from src.rag.manual_rag import ManualRAG

DialogAction = Literal["ASK", "SOLVE", "DONE", "ESCALATE_DONE"]

# ==================================================
# 환경/정책
# ==================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("DIALOG_MODEL", os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "25"))

DEFAULT_HARD_TURN_LIMIT = int(os.getenv("SECOND_STAGE_HARD_TURN_LIMIT", "6") or 6)

STRICT_SLOT_GATING = os.getenv("STRICT_SLOT_GATING", "1").strip().lower() in ("1", "true", "yes")

FAREWELL_TEXT = "이용해 주셔서 감사합니다. 안녕히 가세요."


# ==================================================
# 슬롯 정의
# ==================================================
SLOT_KEYS = [
    "where",
    "symptom",
    "when",
    "error_message",
    "attempted",
    "card_or_device",
    "paid_status",        # ✅ EXIT에서 중요
]

REQUIRED_SLOTS_BY_INTENT: Dict[str, List[str]] = {
    "PAYMENT": ["where", "symptom"],
    "EXIT": ["where", "paid_status", "symptom"],  # ✅ 정산 여부 필수
    "ENTRY": ["where", "symptom"],
    "REGISTRATION": ["where", "symptom"],
    "TIME_PRICE": ["symptom"],                    # ✅ where 강제하면 루프가 늘어남
    "FACILITY": ["where", "symptom"],             # error_message는 조건부로 추가
    "COMPLAINT": ["symptom"],
    "NONE": ["symptom"],
}


# ==================================================
# 매뉴얼 후보 매핑
# ==================================================
INTENT_TO_DOCS: Dict[str, List[str]] = {
    "PAYMENT": [
        "payment_card_fail.md",
        "mobile_payment_qr_issue.md",
        "network_terminal_down.md",
        "discount_free_time_issue.md",
    ],
    "TIME_PRICE": [
        "discount_free_time_issue.md",
        "price_inquiry.md",
    ],
    "REGISTRATION": [
        "visit_registration_fail.md",
    ],
    "ENTRY": [
        "entry_gate_not_open.md",
        "lpr_mismatch_or_no_entry_record.md",
    ],
    "EXIT": [
        "exit_gate_not_open.md",
        "lpr_mismatch_or_no_entry_record.md",
    ],
    "FACILITY": [
        "kiosk_ui_device_issue.md",
        "network_terminal_down.md",
        "barrier_physical_fault.md",
        "failsafe_done.md",
    ],
    "COMPLAINT": [],
    "NONE": [],
}


@dataclass
class DialogResult:
    reply: str = ""
    action: DialogAction = "ASK"
    slots: Dict[str, Any] = None
    new_intent: Optional[str] = None
    raw: Optional[str] = None

    def __post_init__(self):
        if self.slots is None:
            self.slots = {}


_rag = ManualRAG()


# ==================================================
# 유틸
# ==================================================
def _normalize(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[\s\.\,\!\?\u3002\uFF0E\uFF0C\uFF01\uFF1F]+", "", t)
    return t


DONE_KEYWORDS = [
    "됐어요", "되었습니다", "해결", "괜찮아요",
    "그만", "종료", "끝", "마칠게",
    "고마워", "감사", "안녕",
]


def _is_done_utterance(text: str) -> bool:
    """
    ✅ '안됐어요' 같은 부정문이 DONE으로 오인되는 문제 차단
    """
    t = _normalize(text)
    neg_prefix = ("안", "못", "미")

    for kw in DONE_KEYWORDS:
        k = _normalize(kw)
        idx = t.find(k)
        if idx == -1:
            continue
        if idx > 0 and t[idx - 1] in neg_prefix:
            continue
        if t.startswith("안" + k) or t.startswith("못" + k) or t.startswith("미" + k):
            continue
        return True
    return False


def _norm_intent_name(x: Any) -> str:
    if not x:
        return "NONE"
    s = str(x).strip().upper()
    if s.startswith("INTENT."):
        s = s.split(".", 1)[-1]
    return s if s in REQUIRED_SLOTS_BY_INTENT else "NONE"


def _merge_slots(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    if not isinstance(incoming, dict):
        return out
    for k in SLOT_KEYS:
        if k not in incoming:
            continue
        v = incoming.get(k)
        if isinstance(v, str):
            v = v.strip()
            if not v:
                continue
        if v is None:
            continue
        out[k] = v
    return out


def _get_current_intent(ctx: Dict[str, Any]) -> str:
    cur = (ctx.get("current_intent") or "").strip()
    if cur:
        return _norm_intent_name(cur)
    first = (ctx.get("first_intent") or ctx.get("intent") or "").strip()
    return _norm_intent_name(first)


def _dynamic_required_slots(intent: str, slots: Dict[str, Any]) -> List[str]:
    """
    ✅ FACILITY에서 오류 문구가 핵심인 경우 error_message를 조건부 필수로 추가
    """
    req = list(REQUIRED_SLOTS_BY_INTENT.get(intent, ["symptom"]))

    if intent == "FACILITY":
        sym = str((slots or {}).get("symptom", "") or "")
        # 오류/통신 관련이면 error_message가 있으면 해결이 빨라짐
        if any(x in sym for x in ["오류", "통신", "서버", "연결"]) and not (slots or {}).get("error_message"):
            if "error_message" not in req:
                req.append("error_message")

    if intent == "PAYMENT":
        # 승인 실패/통신 문제면 attempted(재시도/다른 카드)도 있으면 좋지만 "필수"로 만들면 루프 생길 수 있어 보류
        pass

    return req


def _missing_required_slots(intent: str, slots: Dict[str, Any]) -> List[str]:
    req = _dynamic_required_slots(intent, slots)
    missing = []
    for k in req:
        v = (slots or {}).get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            missing.append(k)
    return missing


# ==================================================
# 질문 템플릿
# ==================================================
def _first_clarify_question(intent: str) -> str:
    if intent == "PAYMENT":
        return "결제를 어디에서 진행하셨나요? (출구 정산기/사전 정산기(키오스크)/모바일·앱·QR)"
    if intent == "EXIT":
        return "정산은 이미 완료하셨나요? (예: 정산 완료/아직/모름)"
    if intent == "ENTRY":
        return "입구에서 어떤 문제가 발생했나요? (차량 인식/차단기/오류 문구/무반응)"
    if intent == "REGISTRATION":
        return "등록을 어디에서 진행 중이신가요? (키오스크/입구/관리실 등)"
    if intent == "TIME_PRICE":
        return "시간/요금 문제 중 어떤 항목인가요? (무료 시간/요금 과다/시간 표시/할인 적용)"
    if intent == "FACILITY":
        return "문제가 발생한 기기/위치를 알려주세요. (키오스크/차단기/입출구 장비 등)"
    return "어떤 현상이 문제인지 한 문장으로 알려주세요."


def _question_for_missing_slot(intent: str, slot: str) -> str:
    if slot == "where":
        return "문제가 발생한 위치/기기가 어디인가요? (입구/출구/키오스크/정산기/차단기 등)"
    if slot == "paid_status":
        return "정산은 완료하셨나요? (정산 완료/아직/모름)"
    if slot == "symptom":
        if intent == "PAYMENT":
            return "결제에서 어떤 현상이 발생하시나요? (승인 실패/카드 인식 불가/결제 버튼 무반응/정산 반영 문제)"
        if intent == "TIME_PRICE":
            return "시간/요금에서 어떤 부분이 문제인가요? (무료 시간 미적용/요금 과다/시간이 다르게 표시 등)"
        if intent == "REGISTRATION":
            return "등록에서 어떤 단계가 안 되나요? (번호 입력/인증/저장/조회/완료가 안 뜸)"
        if intent in ("EXIT", "ENTRY"):
            return "차단기 동작이 어떤가요? (안 열림/무반응/오류 문구/차량 인식 안 됨 등)"
        if intent == "FACILITY":
            return "어떤 현상이 문제인가요? (먹통/무반응/오류 문구/통신 불가 등)"
        return "어떤 현상이 문제인가요?"
    if slot == "error_message":
        return "화면에 표시되는 오류 문구가 있다면 그대로 읽어주세요."
    if slot == "attempted":
        return "이미 시도해 보신 조치가 있나요? (재시도/다른 카드/재부팅 등)"
    if slot == "card_or_device":
        return "결제 수단/기기 상태를 알려주세요. (카드/모바일/QR, 카드 방향/IC칩 등)"
    if slot == "when":
        return "언제부터 발생했나요? (방금/오늘/어제부터 등)"
    return "조금 더 자세히 말씀해 주실 수 있을까요?"


# ==================================================
# 휴리스틱 슬롯 추출 (안정성 핵심)
# ==================================================
def _extract_quoted(text: str) -> Optional[str]:
    m = re.search(r"[\"“”'‘’]([^\"“”'‘’]{3,120})[\"“”'‘’]", text)
    if m:
        return m.group(1).strip()
    return None


def _heuristic_extract_slots(text: str, intent: str) -> Dict[str, Any]:
    t = (text or "").strip()
    tl = t.lower()
    tt = _normalize(t)

    out: Dict[str, Any] = {}

    # where
    if any(x in t for x in ["출구", "출차"]):
        out["where"] = "출구"
    if any(x in t for x in ["입구", "입차"]):
        out["where"] = "입구"
    if any(x in t for x in ["키오스크", "정산기", "사전정산"]):
        out["where"] = "키오스크/정산기"
    if any(x in t for x in ["출구 정산기", "출구정산기"]):
        out["where"] = "출구 정산기"
    if any(x in t for x in ["차단기", "차단봉", "차단바", "바리어"]):
        out.setdefault("where", "차단기")

    # paid_status (EXIT에서 특히)
    if any(x in t for x in ["정산 했", "정산했", "결제했", "결제 했", "정산 완료", "정산완료"]):
        out["paid_status"] = "정산 완료"
    if any(x in t for x in ["정산 안", "정산안", "결제 안", "결제안", "아직 안 했", "아직안했"]):
        out["paid_status"] = "미정산"

    # error_message
    q = _extract_quoted(t)
    if q:
        out["error_message"] = q

    if "서버와 통신할 수 없습니다" in t or ("서버" in t and "통신" in t and ("없" in t or "안" in t)):
        out["error_message"] = "서버와 통신할 수 없습니다"

    if any(x in t for x in ["통신할 수 없", "통신이 안", "네트워크가 안", "연결이 안"]):
        out.setdefault("error_message", "통신/연결 문제")

    # symptom - 공통
    if any(x in t for x in ["먹통", "무반응", "반응이 없", "버튼이 안", "터치가 안"]):
        out["symptom"] = "무반응/먹통"
    if any(x in t for x in ["오류", "에러", "error", "err"]):
        out.setdefault("symptom", "오류 문구 표시")
    if any(x in t for x in ["통신할 수 없", "통신이 안", "서버", "네트워크", "연결이 안"]):
        out.setdefault("symptom", "서버/통신 불가")
    if any(x in t for x in ["안 열", "안올라", "안 올라", "안 열려", "안열려"]):
        out.setdefault("symptom", "차단기 미동작/미개방")

    # intent별 symptom
    if intent == "PAYMENT":
        if ("승인" in t or "증인" in t) and "실패" in t:
            out["symptom"] = "승인 실패"
        elif "인식" in t and ("안" in t or "불가" in t):
            out["symptom"] = "카드 인식 불가"
        elif ("버튼" in t or "터치" in t) and ("무반응" in t or "안" in t):
            out["symptom"] = "결제 버튼 무반응"
        elif "반영" in t and ("안" in t or "누락" in t):
            out["symptom"] = "정산 반영 문제"

        if any(x in t for x in ["카드", "ic", "칩"]):
            out["card_or_device"] = "카드"
        if any(x in t for x in ["모바일", "앱", "qr"]):
            out["card_or_device"] = "모바일/QR"

    if intent == "TIME_PRICE":
        if "무료" in t and any(x in t for x in ["적용 안", "미적용", "왜 안", "안됐", "안 됐"]):
            out["symptom"] = "무료 시간 미적용"
        elif any(x in t for x in ["요금이 비싸", "요금 과다", "너무 많이", "과금"]):
            out["symptom"] = "요금 과다/할인 미반영"
        elif any(x in t for x in ["시간이 다르", "시간 표시", "시간이 안 맞"]):
            out["symptom"] = "시간 표시/계산 불일치"

    if intent == "REGISTRATION":
        if any(x in t for x in ["등록이 안", "등록 안", "등록 실패", "등록 불가"]):
            out["symptom"] = "등록 실패"
        if any(x in t for x in ["번호판", "인식", "lpr"]):
            out.setdefault("symptom", "번호판 인식 문제")
        out.setdefault("where", "키오스크/정산기")

    if intent in ("EXIT", "ENTRY", "FACILITY"):
        if any(x in t for x in ["차량 인식", "번호판 인식", "인식이 안"]):
            out.setdefault("symptom", "차량/번호판 인식 실패")

    # attempted
    if any(x in t for x in ["재시도", "다시 해", "다시해", "다시 시도"]):
        out["attempted"] = "재시도"
    if any(x in t for x in ["재부팅", "껐다", "다시 켰", "전원"]):
        out["attempted"] = "재부팅/전원"

    return out


# ==================================================
# 매뉴얼 컨텍스트 & 해결 문장(템플릿) 추출
# ==================================================
def _strip_markdown_noise(text: str) -> str:
    if not text:
        return ""
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        if re.match(r"^-\s*#\s*", s):
            continue
        if s.startswith("```"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _extract_section(text: str, section_title_keywords: List[str]) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    start = -1
    for i, ln in enumerate(lines):
        s = ln.strip()
        for kw in section_title_keywords:
            if kw in s:
                start = i + 1
                break
        if start != -1:
            break
    if start == -1:
        return ""

    end = len(lines)
    for j in range(start, len(lines)):
        s = lines[j].strip()
        if s.startswith("#"):
            end = j
            break
        if re.match(r"^(필수 슬롯|추가 슬롯|진단 분기|조치 순서|에스컬레이션|첫 질문)\b", s):
            end = j
            break
    return "\n".join(lines[start:end]).strip()


def _extract_solve_templates(chunk_text: str) -> List[str]:
    sec = _extract_section(chunk_text, ["해결 안내 문장 템플릿", "SOLVE_TEMPLATE"])
    if not sec:
        return []
    out: List[str] = []
    for ln in sec.splitlines():
        s = ln.strip()
        if not s:
            continue
        s = re.sub(r"^[-•\*]\s+", "", s)
        s = re.sub(r"^\d+\.\s+", "", s)
        s = s.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("```") or s.startswith("tags:"):
            continue
        out.append(s)
    # 중복 제거
    uniq = []
    seen = set()
    for x in out:
        k = _normalize(x)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(x)
    return uniq


def _score_template(tpl: str, slots: Dict[str, Any]) -> int:
    """
    ✅ symptom/error_message/where와 더 잘 맞는 템플릿 우선
    """
    score = 0
    sym = _normalize(str(slots.get("symptom", "") or ""))
    err = _normalize(str(slots.get("error_message", "") or ""))
    wh = _normalize(str(slots.get("where", "") or ""))

    ntpl = _normalize(tpl)

    if sym and sym in ntpl:
        score += 5
    if err and err in ntpl:
        score += 5
    if wh and wh in ntpl:
        score += 2

    # "(승인 실패)" 같이 괄호 라벨이 있으면 sym 매칭 가산
    if sym and any(k in ntpl for k in [sym]):
        score += 2
    return score


def _pick_best_template(templates: List[str], slots: Dict[str, Any]) -> Optional[str]:
    if not templates:
        return None
    scored = sorted((( _score_template(t, slots), t) for t in templates), key=lambda x: x[0], reverse=True)
    best_score, best = scored[0]
    return best if best_score > 0 else templates[0]


def _build_manual_context(hits: List[Any]) -> Tuple[str, List[str]]:
    if not hits:
        return "", []

    # 템플릿 우선 추출
    templates: List[str] = []
    for c in hits:
        raw = getattr(c, "text", "") or ""
        templates.extend(_extract_solve_templates(raw))

    ctx_lines: List[str] = []
    ctx_lines.append("[MANUAL_CONTEXT_BEGIN]")
    ctx_lines.append("아래는 참고 매뉴얼 발췌다. 이 내용으로만 답하라.")
    ctx_lines.append("주의: 제목(#...)이나 태그(- # ...)를 그대로 출력하지 말 것.")
    ctx_lines.append("")

    for i, c in enumerate(hits, 1):
        cleaned = _strip_markdown_noise(getattr(c, "text", "") or "")
        if not cleaned:
            continue
        ctx_lines.append(f"(HIT {i}) doc={c.doc_id} chunk={c.chunk_id}")
        ctx_lines.append(cleaned)
        ctx_lines.append("")

    ctx_lines.append("[MANUAL_CONTEXT_END]")

    # 중복 제거
    uniq = []
    seen = set()
    for t in templates:
        k = _normalize(t)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(t)

    return "\n".join(ctx_lines).strip(), uniq


def _preferred_docs(intent: str) -> List[str]:
    return INTENT_TO_DOCS.get(intent, [])


# ==================================================
# LLM 호출 (ASK 유지 + intent 전환 제안용)
# ==================================================
OUTPUT_SCHEMA = """
{
  "reply": "문장",
  "action": "ASK|SOLVE|DONE|ESCALATE_DONE",
  "new_intent": null 또는 "EXIT|ENTRY|PAYMENT|REGISTRATION|TIME_PRICE|FACILITY|COMPLAINT|NONE",
  "slots": {
    "where": null 또는 문자열,
    "symptom": null 또는 문자열,
    "when": null 또는 문자열,
    "error_message": null 또는 문자열,
    "attempted": null 또는 문자열,
    "card_or_device": null 또는 문자열,
    "paid_status": null 또는 문자열
  }
}
""".strip()


def _build_messages(user_text: str, *, history: Optional[List[Dict[str, str]]], ctx: Dict[str, Any], manual_context: str) -> List[Dict[str, str]]:
    intent = _get_current_intent(ctx)
    slots = ctx.get("slots", {}) if isinstance(ctx.get("slots", {}), dict) else {}
    req = _dynamic_required_slots(intent, slots)

    sys = (
        "너는 주차장 키오스크/차단기/정산 관련 2차 트러블슈팅 상담사다.\n"
        "규칙:\n"
        "1) required_slots가 모두 채워지기 전까지는 반드시 action='ASK'만.\n"
        "2) 질문은 한 번에 1개.\n"
        "3) required_slots가 채워지면 매뉴얼 템플릿을 근거로 SOLVE.\n"
        "4) new_intent는 사용자가 명확히 다른 이슈로 전환한 경우에만 제안.\n"
        "5) 출력은 JSON만.\n"
        f"\n- current_intent: {intent}\n"
        f"- required_slots: {req}\n"
        f"- slots_so_far: {slots}\n"
        "\n" + OUTPUT_SCHEMA
    )

    msgs: List[Dict[str, str]] = [{"role": "system", "content": sys}]
    if manual_context:
        msgs.append({"role": "system", "content": manual_context})

    if history:
        for m in history[-10:]:
            if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
                c = m.get("content", "")
                if isinstance(c, str) and c.strip():
                    msgs.append({"role": m["role"], "content": c})

    msgs.append({"role": "user", "content": user_text})
    return msgs


def _parse_json_only(text: str) -> Dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?", "", text or "", flags=re.IGNORECASE).replace("```", "").strip()
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not m:
        raise ValueError("no json")
    return json.loads(m.group(0))


def dialog_llm_chat(
    user_text: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
    context: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> DialogResult:
    # DONE 최우선
    if _is_done_utterance(user_text):
        return DialogResult(reply=FAREWELL_TEXT, action="DONE", slots={})

    ctx = context or {}
    intent = _get_current_intent(ctx)
    hard_limit = int(ctx.get("hard_turn_limit", DEFAULT_HARD_TURN_LIMIT) or DEFAULT_HARD_TURN_LIMIT)
    turn_count_user = int(ctx.get("turn_count_user", 0) or 0)

    # 턴 초과 -> 관리자 호출 종료
    if turn_count_user >= hard_limit:
        return DialogResult(
            reply="여러 번 확인했지만 현재 정보로는 문제 상황을 정확히 특정하기 어렵습니다. 관리자를 호출해 도움을 받아주세요. " + FAREWELL_TEXT,
            action="ESCALATE_DONE",
            slots=ctx.get("slots", {}) if isinstance(ctx.get("slots", {}), dict) else {},
        )

    # 슬롯 휴리스틱 선반영(루프 방지)
    base_slots = ctx.get("slots", {}) if isinstance(ctx.get("slots", {}), dict) else {}
    heuristic = _heuristic_extract_slots(user_text, intent)
    slots_pre = _merge_slots(base_slots, heuristic)

    # RAG
    manual_context = ""
    templates: List[str] = []
    try:
        hits = _rag.retrieve(
            user_text,
            preferred_docs=_preferred_docs(intent),
            hard_filter=True,
            prefer_boost=0.45,
            debug=debug,
        )
        manual_context, templates = _build_manual_context(hits or [])
    except Exception as e:
        if debug:
            print(f"[RAG] failed: {e}")
        manual_context, templates = "", []

    # 첫 2차 응답은 무조건 재질문
    if turn_count_user == 0:
        missing0 = _missing_required_slots(intent, slots_pre)
        q = _question_for_missing_slot(intent, missing0[0]) if missing0 else _first_clarify_question(intent)
        return DialogResult(reply=q, action="ASK", slots=slots_pre)

    # required 체크
    missing = _missing_required_slots(intent, slots_pre)

    # 아직 부족하면 -> ASK를 코드가 강제(LLM 호출 없이도 가능)
    if STRICT_SLOT_GATING and missing:
        return DialogResult(reply=_question_for_missing_slot(intent, missing[0]), action="ASK", slots=slots_pre)

    # 여기까지 왔으면 required 충족 -> SOLVE로 가는게 기본
    # 다만 intent 전환 감지를 위해 LLM 한번 쓰되, ASK 루프는 허용하지 않음
    try:
        url = f"{OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": OLLAMA_MODEL,
            "messages": _build_messages(user_text, history=history, ctx={**ctx, "slots": slots_pre}, manual_context=manual_context),
            "stream": False,
            "options": {"temperature": 0.2},
        }
        r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()
        content = (r.json().get("message") or {}).get("content", "") or ""
        obj = _parse_json_only(content)
    except Exception as e:
        if debug:
            print(f"[DIALOG] LLM call/parse failed: {e}")
        obj = {"action": "SOLVE", "reply": "", "new_intent": None, "slots": {}}
        content = None

    # new_intent 제안 처리(엔진이 최종 채택)
    new_intent = obj.get("new_intent", None)
    if isinstance(new_intent, str):
        new_intent = _norm_intent_name(new_intent)
    else:
        new_intent = None

    # 슬롯 병합
    slots_llm = obj.get("slots", {})
    if not isinstance(slots_llm, dict):
        slots_llm = {}
    slots = _merge_slots(slots_pre, slots_llm)

    # (안전) 혹시 LLM이 다시 ASK를 뱉어도, required 충족이면 SOLVE로 강제
    action = str(obj.get("action", "SOLVE") or "SOLVE").strip().upper()
    if action not in ("ASK", "SOLVE", "DONE", "ESCALATE_DONE"):
        action = "SOLVE"
    if action == "ASK":
        action = "SOLVE"

    if action == "DONE":
        return DialogResult(reply=FAREWELL_TEXT, action="DONE", slots=slots, new_intent=new_intent, raw=content)

    if action == "ESCALATE_DONE":
        return DialogResult(
            reply=(str(obj.get("reply", "") or "").strip() or ("확인이 어려워 관리자를 호출해 도움을 받아주세요. " + FAREWELL_TEXT)),
            action="ESCALATE_DONE",
            slots=slots,
            new_intent=new_intent,
            raw=content,
        )

    # SOLVE: 템플릿 우선
    best_tpl = _pick_best_template(templates, slots)
    where_txt = slots.get("where", "해당 위치")
    sym_txt = slots.get("symptom", "해당 증상")

    if best_tpl:
        reply = (
            f"확인 결과, **{where_txt}**에서 **{sym_txt}** 상황으로 보입니다.\n\n"
            f"{best_tpl}\n\n"
            "위 안내로 해결되셨나요? 다른 문제가 더 있으시면 말씀해 주세요."
        )
    else:
        # 템플릿이 없으면 LLM reply 사용(단, 헤더/태그 복사 방지)
        llm_reply = str(obj.get("reply", "") or "").strip()
        llm_reply = re.sub(r"(?m)^\s*#.*$", "", llm_reply).strip()
        llm_reply = re.sub(r"(?m)^\s*-\s*#.*$", "", llm_reply).strip()
        reply = llm_reply or (
            f"확인 결과, **{where_txt}**에서 **{sym_txt}** 상황으로 보입니다.\n\n"
            "관리자를 호출해 장비/회선을 점검해 주세요.\n\n"
            "위 안내로 해결되셨나요? 다른 문제가 더 있으시면 말씀해 주세요."
        )

    return DialogResult(reply=reply, action="SOLVE", slots=slots, new_intent=new_intent, raw=content)
