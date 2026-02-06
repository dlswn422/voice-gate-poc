from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Tuple

import requests

from src.nlu.llm_client import detect_intent_llm
from src.rag.manual_rag import ManualRAG


# ==================================================
# 환경/정책
# ==================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("DIALOG_MODEL", os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))

DEFAULT_HARD_TURN_LIMIT = int(os.getenv("SECOND_STAGE_HARD_TURN_LIMIT", "6") or 6)

FAREWELL_TEXT = "이용해 주셔서 감사합니다. 안녕히 가세요."


# ==================================================
# 슬롯 템플릿 (사용자 최종 정의)
# - 슬롯 값은 "문장 그대로(raw)" 저장
# ==================================================
SLOT_KEYS = ["symptom", "card_or_device", "error_message", "visit_place"]

REQUIRED_SLOTS_BY_INTENT: Dict[str, List[str]] = {
    "PAYMENT": ["symptom", "card_or_device", "error_message"],
    "EXIT": ["symptom", "error_message"],
    "ENTRY": ["symptom", "error_message"],
    "REGISTRATION": ["symptom", "error_message", "visit_place"],
    "TIME_PRICE": ["symptom"],
    "FACILITY": ["symptom", "error_message"],
    "COMPLAINT": ["symptom"],  # complaint는 symptom 받고 재분류
    "NONE": ["symptom"],
}

SLOT_DESC: Dict[str, str] = {
    "symptom": "현상/문제점. 사용자가 말한 문장을 그대로 받는다(예: '카드 승인 실패가 계속 떠요').",
    "card_or_device": "결제수단/매체. 사용자가 말한 문장을 그대로 받는다(예: '실물 카드로 했어요', '삼성페이').",
    "error_message": "오류 문구. 화면에 뜬 문구를 그대로 받는다. 없으면 '없음'이라고 말하도록 유도한다.",
    "visit_place": "방문업체/매장명. 사용자가 말한 문장을 그대로 받는다.",
}


# ==================================================
# 유틸
# ==================================================
def _norm_intent_name(x: Any) -> str:
    if not x:
        return "NONE"
    s = str(x).strip().upper()
    if s.startswith("INTENT."):
        s = s.split(".", 1)[-1]
    return s if s in REQUIRED_SLOTS_BY_INTENT else "NONE"


def _missing_slots(intent_name: str, slots: Dict[str, Any]) -> List[str]:
    req = REQUIRED_SLOTS_BY_INTENT.get(_norm_intent_name(intent_name), ["symptom"])
    miss: List[str] = []
    for k in req:
        v = (slots or {}).get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            miss.append(k)
    return miss


DONE_KEYWORDS = [
    "됐어요", "되었습니다", "해결", "괜찮아요",
    "그만", "종료", "끝", "마칠게",
    "고마워", "감사", "안녕",
]


def _normalize(text: str) -> str:
    return re.sub(r"[\s\.\,\!\?]+", "", (text or "").strip().lower())


def _is_done_utterance(text: str) -> bool:
    t = _normalize(text)
    return any(_normalize(k) in t for k in DONE_KEYWORDS)


def _manuals_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "manuals"))


def _read_manual(filename: str) -> str:
    path = os.path.join(_manuals_dir(), filename)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _ollama_chat(messages: List[Dict[str, str]], temperature: float = 0.6) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return ((data.get("message") or {}).get("content", "") or "").strip()


def _json_extract(text: str) -> Optional[dict]:
    """LLM 출력에서 JSON 1개를 최대한 뽑아낸다."""
    if not text:
        return None
    s = text.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and j > i:
        snippet = s[i:j + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def _slot_text(slots: Dict[str, Any], key: str) -> str:
    v = (slots or {}).get(key)
    return (str(v).strip() if v is not None else "")


# ==================================================
# PAYMENT 메뉴얼 CASE 파싱 (구조화)
# ==================================================
@dataclass
class ManualCase:
    case_id: str
    cond_kv: Dict[str, str]     # {"symptom": "...", "card_or_device": "...", "error_message": "..."}
    solve_text: str             # 출력용(메뉴얼 그대로)
    raw_conditions_text: str    # 디버그/LLM 후보 설명용


_CASE_RE = re.compile(r"^###\s*CASE\s*(P-\d+)\s*$", re.IGNORECASE | re.MULTILINE)


def _parse_case_conditions_to_kv(cond_block: str) -> Dict[str, str]:
    kv: Dict[str, str] = {}
    if not cond_block:
        return kv
    for ln in cond_block.splitlines():
        ln = ln.strip()
        m = re.match(r"^-+\s*([a-zA-Z0-9_]+)\s*:\s*(.+?)\s*$", ln)
        if not m:
            continue
        k = m.group(1).strip()
        v = m.group(2).strip()
        kv[k] = v
    return kv


def _parse_payment_cases(md: str) -> List[ManualCase]:
    if not md:
        return []

    headers = [(m.group(1).strip().upper(), m.start()) for m in _CASE_RE.finditer(md)]
    if not headers:
        return []

    cases: List[ManualCase] = []
    for idx, (cid, start) in enumerate(headers):
        end = headers[idx + 1][1] if idx + 1 < len(headers) else len(md)
        block = md[start:end].strip()

        cond_text = ""
        m_cond = re.search(r"\*\*조건\*\*\s*(.*?)(?:\*\*안내\s*문장\*\*|$)", block, re.IGNORECASE | re.DOTALL)
        if m_cond:
            cond_text = (m_cond.group(1) or "").strip()

        solve_text = ""
        m_solve = re.search(r"\*\*안내\s*문장\*\*\s*(.*)$", block, re.IGNORECASE | re.DOTALL)
        if m_solve:
            solve_text = (m_solve.group(1) or "").strip()

        cond_text_clean = "\n".join([ln.strip() for ln in cond_text.splitlines() if ln.strip()])
        solve_text_clean = "\n".join([ln.rstrip() for ln in solve_text.splitlines() if ln.strip()])

        if not solve_text_clean:
            continue

        kv = _parse_case_conditions_to_kv(cond_text_clean)

        cases.append(ManualCase(
            case_id=cid,
            cond_kv=kv,
            solve_text=solve_text_clean,
            raw_conditions_text=cond_text_clean,
        ))

    return cases


# ==================================================
# 1) 슬롯 질문(LLM 자율 생성) - 프롬프트/검증 강화
# ==================================================
SYSTEM_PROMPT_ASK = """
너는 주차장 고객지원 상담원이다.
목표: 부족한 슬롯 1개를 채우기 위한 질문 1개만 생성한다.

중요: '해결' 단계가 아니다. 해결책/추측/원인/권장/재시도/관리자 호출을 절대 말하지 마라.
질문만 한다.

절대 규칙:
- 질문은 한국어, 공손한 존댓말.
- 반드시 물음표(?)로 끝낼 것.
- 1~2문장 이내.
- 한 번에 슬롯 1개만 묻는다. (다른 슬롯을 같이 묻지 마라)
- 이미 채워진 내용은 다시 묻지 않는다.
- 사용자가 그대로 답하기 쉬운 형태로 묻는다.
- 필요한 경우 괄호 예시는 2~4개만. 긴 나열 금지.

슬롯 정의(값은 사용자가 말한 문장 그대로 저장됨):
- symptom: 현상/문제점
- card_or_device: 결제수단/매체
- error_message: 화면 오류 문구(없으면 '없음'이라고 답하도록 유도)
- visit_place: 방문업체/매장명

출력은 JSON 1개만. 다른 텍스트 금지.
형식:
{"reply":"질문문장?","pending_slot":"<missing_slot 그대로>"}

검증:
- pending_slot은 반드시 입력으로 받은 missing_slot과 정확히 일치해야 한다.
- reply는 반드시 ?로 끝나야 한다.
""".strip()


def _llm_make_slot_question(
    intent_name: str,
    missing_slot: str,
    slots: Dict[str, Any],
    history: Optional[List[Dict[str, str]]],
) -> str:
    slot_view = {k: (slots or {}).get(k) for k in SLOT_KEYS}

    user_context = (
        f"[intent]={intent_name}\n"
        f"[missing_slot]={missing_slot}\n"
        f"[missing_slot_desc]={SLOT_DESC.get(missing_slot,'')}\n"
        f"[filled_slots]={json.dumps(slot_view, ensure_ascii=False)}\n"
        "위 정보를 바탕으로 missing_slot 하나만 채우기 위한 질문을 1개 생성해라.\n"
        "해결책/조언은 절대 말하지 말고 질문만 해라.\n"
    )

    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT_ASK}]
    if history:
        for h in history[-6:]:
            if h.get("role") in ("user", "assistant"):
                msgs.append({"role": h["role"], "content": h.get("content", "")})
    msgs.append({"role": "user", "content": user_context})

    # ✅ 질문 품질 안정화를 위해 온도 낮춤
    content = _ollama_chat(msgs, temperature=0.35)
    obj = _json_extract(content) or {}
    reply = str(obj.get("reply", "")).strip()
    ps = str(obj.get("pending_slot", "")).strip()

    # ✅ 형식 강제 검증
    if reply and not reply.endswith("?"):
        reply = ""
    if reply and ps != missing_slot:
        reply = ""

    # ✅ 직전 assistant 질문 반복 방지(동일 문장 반복이면 폴백)
    if reply and history:
        last_a = next((h.get("content", "") for h in reversed(history) if h.get("role") == "assistant"), "")
        if last_a and _normalize(last_a) == _normalize(reply):
            reply = ""

    if reply:
        return reply

    # 최소 폴백(LLM 출력이 깨졌을 때만)
    if missing_slot == "card_or_device":
        return "어떤 결제수단/매체로 결제하셨나요? (실물 카드/삼성페이·휴대폰결제/QR·모바일 등)"
    if missing_slot == "error_message":
        return "화면에 표시된 오류 문구가 있나요? 있다면 그대로 말씀해 주세요. 없으면 '없음'이라고 말씀해 주세요."
    if missing_slot == "visit_place":
        return "방문하신 업체(매장)명이 있나요? 있다면 업체명을 말씀해 주세요."
    return "현재 어떤 문제가 발생했는지 한 문장으로 말씀해 주실 수 있을까요?"


# ==================================================
# 2) PAYMENT CASE 선택 안정화
# - error_message: '없음' vs '통신/서버 오류 문구 있음'은 혼동이 잦아서 하드필터로 안정화
# ==================================================
def _payment_error_bucket(error_message_raw: str) -> Optional[str]:
    t = (error_message_raw or "").strip()
    if not t:
        return None
    # ✅ 사용자가 명시적으로 '없음'이라고 말한 경우 최우선
    if t == "없음" or "없음" in t or "문구없" in t or "안떠" in t or "표시없" in t:
        return "없음"
    # ✅ 통신/서버/네트워크 단어가 있으면 해당 버킷
    if any(k in t for k in ["통신", "서버", "네트워크", "연결", "접속", "끊김"]):
        return "통신/서버 오류 문구 있음"
    return None


def _filter_payment_cases_by_error_message(slots: Dict[str, Any], cases: List[ManualCase]) -> List[ManualCase]:
    bucket = _payment_error_bucket(_slot_text(slots, "error_message"))
    if not bucket:
        return cases

    filtered: List[ManualCase] = []
    for c in cases:
        cond_val = (c.cond_kv.get("error_message") or "").strip()
        # 메뉴얼 조건 값에 bucket 문자열이 포함되는지 기준(포맷 변화에도 강함)
        if bucket in cond_val:
            filtered.append(c)
    return filtered or cases


SYSTEM_PROMPT_SELECT_CASE = """
너는 '케이스 선택기'다.
입력 슬롯(raw 문장)과 CASE 조건을 비교하여, 조건이 가장 정확히 일치하는 case_id를 고른다.

절대 규칙:
- solve 문장을 만들거나 바꾸거나 출력하지 마라. (case_id만 선택)
- error_message가 '없음'이면, '통신/서버 오류 문구 있음' CASE를 절대 선택하지 마라.
- error_message raw에 통신/서버/네트워크 단어가 없으면,
  '통신/서버 오류 문구 있음' CASE를 절대 선택하지 마라.
- 출력은 JSON 1개만. 다른 텍스트 금지.

출력(JSON만):
{"action":"SELECT","case_id":"P-01","confidence":0.0~1.0}
또는 (정보가 모자라면)
{"action":"ASK","need_slot":"error_message","confidence":0.0~1.0}
""".strip()


def _llm_select_case_id(
    intent_name: str,
    slots: Dict[str, Any],
    cases: List[ManualCase],
) -> Tuple[Optional[str], Optional[str], float]:
    """
    returns: (case_id, need_slot, confidence)
    """
    cases2 = _filter_payment_cases_by_error_message(slots, cases)

    # ✅ 후보가 1개면 LLM 없이 결정
    if len(cases2) == 1:
        return cases2[0].case_id, None, 0.99

    cands = []
    for c in cases2:
        cands.append({
            "case_id": c.case_id,
            "conditions": c.cond_kv,
        })

    slot_view = {k: (slots or {}).get(k) for k in SLOT_KEYS}
    prompt = (
        f"[intent]={intent_name}\n"
        f"[slots_raw]={json.dumps(slot_view, ensure_ascii=False)}\n"
        f"[candidates_json]={json.dumps(cands, ensure_ascii=False)}\n\n"
        "slots_raw의 의미를 보고 candidates_json 중 조건이 가장 일치하는 case_id를 선택해라.\n"
        "solve 문장은 절대 출력하지 마라.\n"
    )

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT_SELECT_CASE},
        {"role": "user", "content": prompt},
    ]
    # ✅ 결정 안정화(사실상 deterministic)
    content = _ollama_chat(msgs, temperature=0.0)
    obj = _json_extract(content) or {}

    action = str(obj.get("action", "")).strip().upper()
    conf = float(obj.get("confidence", 0.75) or 0.75)

    if action == "SELECT":
        cid = str(obj.get("case_id", "")).strip().upper()
        if cid and any(c.case_id == cid for c in cases2):
            return cid, None, conf
        return None, None, conf

    if action == "ASK":
        need = str(obj.get("need_slot", "")).strip()
        if need in SLOT_KEYS:
            return None, need, conf
        return None, None, conf

    return None, None, conf


# ==================================================
# 결과 타입
# ==================================================
DialogAction = Literal["ASK", "SOLVE", "DONE", "ESCALATE_DONE"]


@dataclass
class DialogResult:
    reply: str
    action: DialogAction = "ASK"
    slots: Optional[Dict[str, Any]] = None
    pending_slot: Optional[str] = None
    new_intent: Optional[str] = None
    confidence: float = 0.75


# ==================================================
# 메인 엔트리
# ==================================================
def dialog_llm_chat(
    user_text: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
    context: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> DialogResult:
    # 종료 발화
    if _is_done_utterance(user_text):
        return DialogResult(reply=FAREWELL_TEXT, action="DONE", confidence=1.0, slots=(context or {}).get("slots") or {})

    ctx = context or {}
    hard_limit = int(ctx.get("hard_turn_limit", DEFAULT_HARD_TURN_LIMIT) or DEFAULT_HARD_TURN_LIMIT)
    turn_count_user = int(ctx.get("turn_count_user", 0) or 0)

    current_intent = _norm_intent_name(ctx.get("intent"))
    slots = ctx.get("slots") if isinstance(ctx.get("slots"), dict) else {}
    pending_slot = ctx.get("pending_slot")

    # 턴 제한
    if turn_count_user >= hard_limit:
        return DialogResult(
            reply="여러 번 확인했지만 현재 정보로는 문제 상황을 정확히 특정하기 어렵습니다. 관리자를 호출해 도움을 받아주세요. " + FAREWELL_TEXT,
            action="ESCALATE_DONE",
            confidence=1.0,
            slots=slots,
            new_intent=current_intent,
        )

    # pending_slot이 있으면 이번 user_text를 해당 슬롯에 raw 그대로 저장
    if pending_slot:
        if not (isinstance(slots.get(pending_slot), str) and str(slots.get(pending_slot)).strip()):
            slots[pending_slot] = user_text.strip()
        pending_slot = None

    # COMPLAINT 처리: symptom 받고 1회 재분류
    if current_intent == "COMPLAINT":
        if not (slots.get("symptom") and str(slots.get("symptom")).strip()):
            q = "정확한 상황 판단을 위해, 지금 어떤 점이 문제인가요? (예: 결제/출차/입차/등록/요금/기기 오류 등)"
            return DialogResult(reply=q, action="ASK", slots=slots, pending_slot="symptom", confidence=0.95, new_intent="COMPLAINT")

        try:
            rr = detect_intent_llm(str(slots.get("symptom")))
            new_intent = _norm_intent_name(rr.intent.value)
        except Exception:
            new_intent = "NONE"

        if new_intent not in ("COMPLAINT", "NONE"):
            current_intent = new_intent
            # ✅ symptom 유지 + 새 intent에 필요한 슬롯만 유지
            new_slots = {"symptom": slots.get("symptom")}
            for k in REQUIRED_SLOTS_BY_INTENT.get(current_intent, []):
                if k != "symptom" and k in SLOT_KEYS:
                    new_slots[k] = slots.get(k)
            slots = new_slots
            if debug:
                print(f"[DIALOG] complaint -> reclassified intent={current_intent}")

    # 1) 부족 슬롯 있으면 질문
    missing = _missing_slots(current_intent, slots)
    if missing:
        next_slot = missing[0]
        q = _llm_make_slot_question(current_intent, next_slot, slots, history)
        if debug:
            print(f"[DIALOG-ASK] intent={current_intent} missing={missing} pending={next_slot}")
        return DialogResult(
            reply=q,
            action="ASK",
            slots=slots,
            pending_slot=next_slot,
            new_intent=current_intent,
            confidence=0.85,
        )

    # 2) 슬롯 다 찼으면 SOLVE (PAYMENT는 CASE 기반)
    if current_intent == "PAYMENT":
        md = _read_manual("payment_card_fail.md")
        cases = _parse_payment_cases(md)

        if debug:
            print(f"[PAYMENT] slots_raw={slots}")
            print(f"[PAYMENT] cases={len(cases)}")

        if not cases:
            return DialogResult(
                reply="결제 메뉴얼(CASE)을 찾지 못했습니다. 관리자를 호출해 도움을 받아주세요. " + FAREWELL_TEXT,
                action="ESCALATE_DONE",
                confidence=1.0,
                slots=slots,
                new_intent=current_intent,
            )

        case_id, need_slot, conf = _llm_select_case_id(current_intent, slots, cases)

        if need_slot:
            if need_slot in SLOT_KEYS and (not slots.get(need_slot) or not str(slots.get(need_slot)).strip()):
                q = _llm_make_slot_question(current_intent, need_slot, slots, history)
                if debug:
                    print(f"[PAYMENT] selector asked need_slot={need_slot}")
                return DialogResult(
                    reply=q,
                    action="ASK",
                    slots=slots,
                    pending_slot=need_slot,
                    new_intent=current_intent,
                    confidence=max(0.8, conf),
                )

        if case_id:
            chosen = next((c for c in cases if c.case_id == case_id), None)
            if chosen and chosen.solve_text.strip():
                if debug:
                    print(f"[PAYMENT] selected case_id={case_id} conf={conf:.2f}")
                # ✅ solve 문장은 메뉴얼 그대로
                return DialogResult(
                    reply=chosen.solve_text.strip(),
                    action="SOLVE",
                    slots=slots,
                    new_intent=current_intent,
                    confidence=min(0.99, max(0.8, conf)),
                )

        return DialogResult(
            reply="현재 입력 정보로는 결제 문제 유형을 정확히 특정하기 어렵습니다. 관리자를 호출해 도움을 받아주세요. " + FAREWELL_TEXT,
            action="ESCALATE_DONE",
            slots=slots,
            confidence=1.0,
            new_intent=current_intent,
        )

    # 3) PAYMENT 외 intent: RAG fallback (추후 CASE 방식으로 확장 권장)
    rag = ManualRAG()
    query = f"{current_intent} | symptom: {slots.get('symptom','')} | err: {slots.get('error_message','')} | visit: {slots.get('visit_place','')}"
    try:
        hits = rag.retrieve(query=query, preferred_docs=None, hard_filter=False, debug=debug)
    except Exception as e:
        if debug:
            print(f"[RAG] failed: {e}")
        hits = []

    if hits:
        top = getattr(hits[0], "text", "") or ""
        top = top.strip()
        if top:
            return DialogResult(reply=top, action="SOLVE", slots=slots, confidence=0.85, new_intent=current_intent)

    return DialogResult(
        reply="현재 메뉴얼에서 해당 상황의 해결 안내를 찾지 못했습니다. 관리자를 호출해 도움을 받아주세요. " + FAREWELL_TEXT,
        action="ESCALATE_DONE",
        slots=slots,
        confidence=1.0,
        new_intent=current_intent,
    )
