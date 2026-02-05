from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Tuple

import requests

from src.nlu.intent_schema import Intent
from src.rag.manual_rag import ManualRAG

# ==================================================
# 환경/정책
# ==================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "30"))

# 2차 강제 종료(관리자 호출) 턴 제한
DEFAULT_HARD_TURN_LIMIT = int(os.getenv("SECOND_STAGE_HARD_TURN_LIMIT", "6") or 6)

# ASK(질문 생성) 자유도 / SOLVE는 메뉴얼 문장 그대로 반환(LLM 사용 X)
ASK_TEMPERATURE = float(os.getenv("DIALOG_ASK_TEMPERATURE", "0.7") or 0.7)

# SOLVE를 LLM이 다듬지 못하게 강제(=메뉴얼 솔루션 그대로)
FORCE_MANUAL_TEMPLATE_SOLVE = True

# ==================================================
# 슬롯 정의 / 필수 슬롯
# ==================================================
SLOT_KEYS = [
    "symptom",
    "where",
    "when",
    "error_message",
    "attempted",
    "card_or_device",
    "plate_or_ticket",
]

# 의도별 필수 슬롯(최소)
REQUIRED_SLOTS_BY_INTENT: Dict[str, List[str]] = {
    "PAYMENT": ["where", "symptom"],
    "EXIT": ["where", "symptom"],
    "ENTRY": ["where", "symptom"],
    "REGISTRATION": ["where", "symptom"],
    "TIME_PRICE": ["symptom"],  # 무료/할인/요금 이슈는 where 없이도 해결 안내 가능
    "FACILITY": ["where", "symptom"],
    "COMPLAINT": ["symptom"],
    "NONE": ["symptom"],
}

# 질문 우선순위(LLM에게 힌트로 제공)
SLOT_PRIORITY_BY_INTENT: Dict[str, List[str]] = {
    "PAYMENT": ["where", "symptom", "card_or_device", "error_message", "attempted"],
    "EXIT": ["where", "symptom", "when", "error_message", "attempted"],
    "ENTRY": ["symptom", "where", "error_message", "attempted"],
    "REGISTRATION": ["where", "symptom", "error_message", "plate_or_ticket", "attempted"],
    "TIME_PRICE": ["symptom", "where", "plate_or_ticket", "error_message"],
    "FACILITY": ["where", "symptom", "error_message", "attempted"],
    "COMPLAINT": ["symptom", "where"],
    "NONE": ["symptom", "where"],
}

SLOT_GUIDE: Dict[str, str] = {
    "symptom": "사용자가 겪는 현상/문제점. 예) 승인 실패, 카드 인식 불가, 무반응/먹통, 통신 불가, 무료/할인 미적용, 차단기 안열림, 등록 실패 등",
    "where": "문제가 발생한 위치/기기. 예) 출구 정산기, 사전 정산기(키오스크), 입구, 출구, 차단기, 모바일/앱/QR 등",
    "when": "언제/어떤 시점인지. 예) 결제 직후, 정산 완료 후 출차 시도 시, 입차 시도 시 등",
    "error_message": "화면/기기에 표시된 오류 문구 또는 코드(그대로).",
    "attempted": "이미 시도해본 조치. 예) 재시도, 다른 카드, 재부팅, 다시 태그, 다른 결제수단 등",
    "card_or_device": "결제수단/방식 또는 매체. 예) 카드(IC/마그네틱), 모바일 QR, 앱, 현금 등",
    "plate_or_ticket": "차량번호/영수증/입차기록 관련 단서.",
}

# ==================================================
# 메뉴얼 매핑(의도별)
# ==================================================
# ManualRAG 쪽 preferred_docs에 전달 (파일명/키워드는 프로젝트에 맞춰 조정 가능)
INTENT_TO_DOCS: Dict[str, List[str]] = {
    "PAYMENT": [
        "payment_card_fail",
        "mobile_payment_qr_issue",
        "network_terminal_down",
    ],
    "EXIT": [
        "exit_gate_not_open",
        "network_terminal_down",
        "barrier_physical_fault",
    ],
    "ENTRY": [
        "entry_gate_not_open",
        "lpr_mismatch_or_no_entry_record",
        "barrier_physical_fault",
        "network_terminal_down",
    ],
    "REGISTRATION": [
        "visit_registration_fail",
        "lpr_mismatch_or_no_entry_record",
    ],
    "TIME_PRICE": [
        "discount_free_time_issue",
        "price_inquiry",
    ],
    "FACILITY": [
        "kiosk_ui_device_issue",
        "network_terminal_down",
        "barrier_physical_fault",
    ],
}

# ==================================================
# 종료 발화 감지(오인식 방지 포함)
# ==================================================
DONE_KEYWORDS = [
    "됐어요", "됐어", "됐습니다", "해결", "해결됐", "해결됨", "괜찮아요",
    "그만", "종료", "끝", "마칠게", "이만",
    "고마워", "감사", "안녕", "수고", "바이",
]


def _normalize(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[\s\.\,\!\?\u3002\uFF0E\uFF0C\uFF01\uFF1F]+", "", t)
    return t


def _is_done_utterance(text: str) -> bool:
    t = _normalize(text)
    if not t:
        return False

    # ✅ 부정형 포함은 종료로 오인식 금지
    if "안됐" in t or "안되" in t or "안돼" in t:
        return False

    return any(_normalize(k) in t for k in DONE_KEYWORDS)


FAREWELL_TEXT = "이용해 주셔서 감사합니다. 안전운전하세요."


# ==================================================
# 슬롯 유틸
# ==================================================
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
            if isinstance(v, str):
                v = v.strip()
                if not v:
                    v = None
            out[k] = v
    return out


def _missing_required_slots(intent_name: str, slots: Dict[str, Any]) -> List[str]:
    req = REQUIRED_SLOTS_BY_INTENT.get(_norm_intent_name(intent_name), ["symptom"])
    miss: List[str] = []
    for k in req:
        v = (slots or {}).get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            miss.append(k)
    return miss


def _pick_next_missing(intent_name: str, missing: List[str]) -> Optional[str]:
    if not missing:
        return None
    prio = SLOT_PRIORITY_BY_INTENT.get(_norm_intent_name(intent_name), [])
    for k in prio:
        if k in missing:
            return k
    return missing[0]


# ==================================================
# 휴리스틱 슬롯 추출(보조)
# - LLM이 slots 업데이트를 못하더라도 최소한의 안정성 확보
# ==================================================
_WHERE_PATTERNS: List[Tuple[str, str]] = [
    (r"(출구정산기|출구\s*정산기|출구)", "출구 정산기"),
    (r"(사전정산기|사전\s*정산기|키오스크|무인정산기)", "사전 정산기(키오스크)"),
    (r"(입구)", "입구"),
    (r"(차단기|차단봉|게이트)", "차단기"),
    (r"(모바일|앱|qr|큐알)", "모바일/앱/QR"),
]

_ERRMSG_RE = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{5,120})[\"'“”‘’]")


def _heuristic_extract_slots(text: str, intent_name: str) -> Dict[str, Any]:
    t = (text or "").strip()
    out: Dict[str, Any] = {}

    # where
    for pat, norm in _WHERE_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            out["where"] = norm
            break

    # error message quoted
    m = _ERRMSG_RE.search(t)
    if m:
        out["error_message"] = m.group(1).strip()

    # symptom 강화
    if re.search(r"(승인\s*실패|승인에\s*실패|승인이\s*실패|승인\s*거절|거절\s*되|승인\s*안\s*되|승인이\s*안\s*되|승인\s*안되)", t):
        out["symptom"] = "승인 실패"
    elif re.search(r"(통신할\s*수\s*없|서버와\s*통신|네트워크\s*오류|연결\s*실패|연결이\s*안\s*되|연결\s*안되)", t):
        out["symptom"] = "통신 오류"
        if "error_message" not in out and ("서버와 통신할 수 없습니다" in t or "서버와 통신할수 없습니다" in t):
            out["error_message"] = "서버와 통신할 수 없습니다"
    elif re.search(r"(카드\s*인식\s*불가|카드가\s*안\s*읽|ic\s*칩|칩이\s*안\s*읽|카드\s*오류)", t):
        out["symptom"] = "인식 불가"
    elif re.search(r"(무반응|먹통|멈췄|버튼이\s*안\s*눌|눌러도\s*반응|화면이\s*안\s*바뀌)", t):
        out["symptom"] = "무반응"
    elif (re.search(r"(무료|할인|감면)", t) and re.search(r"(적용\s*안|적용이\s*안|미적용|누락|안\s*됐|안됐)", t)):
        out["symptom"] = "무료/할인 미적용"
    elif re.search(r"(등록이\s*안|등록\s*안|등록이\s*안되|등록이\s*안\s*돼|등록\s*실패)", t):
        out["symptom"] = "등록 실패"
    elif re.search(r"(차단기|차단봉).*(안\s*열|안\s*올라|안올라|열리지|안\s*내려|내려가지)", t) or re.search(r"(안\s*열려|안\s*열림)", t):
        out["symptom"] = "차단기 미동작"

    return out


# ==================================================
# ManualRAG
# ==================================================
_rag = ManualRAG()

# 템플릿 섹션 파싱(문장을 "그대로" 뽑기)
_SOLVE_SECTION_RE = re.compile(
    r"##\s*해결 안내 문장 템플릿\s*\(SOLVE_TEMPLATE\)\s*(.*?)(?:\n##\s|\Z)",
    re.DOTALL,
)
_BULLET_RE = re.compile(r"^\s*-\s*(.+?)\s*$", re.MULTILINE)

# 템플릿 라인의 (태그) 파싱
_TAG_LINE_RE = re.compile(r"^\s*\(([^)]+)\)\s*(.+)\s*$")


def _parse_solve_templates_from_manual(manual_text: str) -> List[str]:
    """
    SOLVE_TEMPLATE 섹션의 bullet 라인을 원문 그대로(공백 정리는 최소) 추출.
    반환 예: ["(승인 실패) ...", "(통신 오류) ...", ...]
    """
    if not manual_text:
        return []
    m = _SOLVE_SECTION_RE.search(manual_text)
    if not m:
        return []
    block = m.group(1)

    out: List[str] = []
    for ln in _BULLET_RE.findall(block):
        s = ln.strip()
        if not s:
            continue
        # 너무 짧은 건 제외
        if len(s) < 10:
            continue
        out.append(s)

    # 중복 제거(순서 유지)
    uniq: List[str] = []
    seen = set()
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def _choose_template_line(templates: List[str], slots: Dict[str, Any]) -> Optional[str]:
    """
    slots.symptom에 맞는 (태그) 템플릿을 고른다.
    - 가장 정확: tag가 symptom에 포함되거나, symptom 키워드가 tag/문장에 포함
    - 없으면 첫 번째 템플릿
    """
    if not templates:
        return None

    symptom = (slots or {}).get("symptom") or ""
    symptom = str(symptom).strip()

    scored: List[Tuple[int, str]] = []
    for line in templates:
        m = _TAG_LINE_RE.match(line)
        tag = ""
        body = line
        if m:
            tag = m.group(1).strip()
            body = m.group(2).strip()

        score = 0
        if tag and symptom:
            if tag in symptom:
                score += 100
            if symptom in tag:
                score += 80

        if symptom:
            if "승인" in symptom and ("승인" in tag or "승인" in body):
                score += 60
            if ("통신" in symptom or "서버" in symptom or "네트워크" in symptom) and (("통신" in tag) or ("통신" in body) or ("서버" in body) or ("네트워크" in body)):
                score += 60
            if ("인식" in symptom or "카드" in symptom) and (("인식" in tag) or ("인식" in body) or ("카드" in body)):
                score += 60
            if ("무료" in symptom or "할인" in symptom) and (("무료" in tag) or ("할인" in tag) or ("무료" in body) or ("할인" in body)):
                score += 60
            if ("차단" in symptom) and (("차단" in tag) or ("차단" in body)):
                score += 40
            if ("등록" in symptom) and (("등록" in tag) or ("등록" in body)):
                score += 40

        scored.append((score, line))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_line = scored[0]
    if best_score <= 0:
        return templates[0]
    return best_line


def _build_solve_reply_from_manual(intent_name: str, slots: Dict[str, Any], manual_text: str) -> Optional[str]:
    """
    ✅ SOLVE는 메뉴얼 템플릿 문장 그대로 반환한다.
    """
    templates = _parse_solve_templates_from_manual(manual_text)
    chosen = _choose_template_line(templates, slots)
    if not chosen:
        return None

    return f"{chosen}\n\n위 안내로 해결되셨나요? 다른 문제가 더 있으시면 말씀해 주세요."


# ==================================================
# LLM(ASK) - 슬롯 업데이트 + 질문 1개 생성
# ==================================================
DialogAction = Literal["ASK", "SOLVE", "DONE", "FAILSAFE", "ESCALATE_DONE"]


@dataclass
class DialogResult:
    reply: str
    action: DialogAction = "ASK"
    confidence: float = 0.7
    slots: Dict[str, Any] = None
    new_intent: Optional[str] = None
    raw: Optional[str] = None

    def __post_init__(self):
        if self.slots is None:
            self.slots = {}


_SYSTEM_PROMPT_ASK = """
너는 주차장 무인정산/차단기/출입 시스템 상담원이다.
목표는 사용자 상황을 파악하기 위해 '슬롯'을 채우는 것이다.

규칙(반드시 지켜):
1) 사용자의 발화로부터 슬롯 값을 최대한 추출해 "slots"에 업데이트하라.
2) 아직 부족한 슬롯(missing_slots)을 채우기 위한 질문을 **딱 1개**만 하라.
3) 해결책/조치 안내는 절대 하지 말아라(ASK 단계).
4) 같은 의미의 질문을 반복하지 말아라. 사용자가 같은 답을 반복하면 그 답을 슬롯 값으로 확정하고 다음 슬롯으로 넘어가라.
5) 출력은 반드시 JSON 1개만.

출력 포맷:
{
  "action": "ASK",
  "reply": "<사용자에게 할 질문 1개>",
  "slots": { ... },
  "new_intent": null 또는 "EXIT/ENTRY/PAYMENT/REGISTRATION/TIME_PRICE/FACILITY/COMPLAINT/NONE",
  "confidence": 0.0~1.0
}
""".strip()


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        blob2 = blob.replace("'", '"')
        blob2 = re.sub(r",\s*}", "}", blob2)
        blob2 = re.sub(r",\s*]", "]", blob2)
        try:
            return json.loads(blob2)
        except Exception:
            return None


def _ollama_chat(messages: List[Dict[str, str]], *, temperature: float) -> str:
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
    return ((data.get("message") or {}).get("content") or "").strip()


def _llm_ask(
    user_text: str,
    *,
    history: Optional[List[Dict[str, str]]],
    intent_name: str,
    slots: Dict[str, Any],
    missing_slots: List[str],
    next_missing: Optional[str],
    debug: bool,
) -> Tuple[str, Dict[str, Any], Optional[str], float, str]:
    state = {
        "intent": intent_name,
        "slots": slots,
        "missing_slots": missing_slots,
        "next_missing_slot": next_missing,
        "slot_guide": SLOT_GUIDE,
        "slot_priority": SLOT_PRIORITY_BY_INTENT.get(intent_name, []),
        "rule": {
            "ask_one_question_only": True,
            "no_solution_in_ask": True,
            "avoid_repeating_question": True,
            "accept_repeated_user_answer_as_slot": True,
        },
    }

    msgs: List[Dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT_ASK}]
    msgs.append({"role": "user", "content": f"[STATE]\n{json.dumps(state, ensure_ascii=False)}"})

    if history:
        for m in history[-8:]:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})

    msgs.append({"role": "user", "content": user_text})

    raw = ""
    try:
        raw = _ollama_chat(msgs, temperature=ASK_TEMPERATURE)
    except Exception as e:
        if debug:
            print(f"[ASK_LLM] failed: {e}")
        return "상황을 조금만 더 자세히 말씀해 주실 수 있을까요?", {}, None, 0.4, raw

    obj = _extract_json(raw) or {}
    reply = str(obj.get("reply") or "").strip()
    upd = obj.get("slots") if isinstance(obj.get("slots"), dict) else {}
    new_intent = obj.get("new_intent")
    new_intent = _norm_intent_name(new_intent) if new_intent else None
    try:
        conf = float(obj.get("confidence", 0.7))
    except Exception:
        conf = 0.7

    if not reply:
        if next_missing == "where":
            reply = "문제가 발생한 위치/기기가 어디인가요? (예: 출구 정산기/키오스크/차단기/모바일·앱·QR)"
        elif next_missing == "symptom":
            reply = "어떤 현상이 문제인가요? (예: 승인 실패/카드 인식 불가/무반응/통신 오류/미적용 등)"
        elif next_missing == "error_message":
            reply = "화면에 뜨는 오류 문구나 코드가 있나요? 그대로 읽어 주세요."
        else:
            reply = "상황을 조금만 더 자세히 말씀해 주실 수 있을까요?"

    return reply, upd, new_intent, max(0.0, min(1.0, conf)), raw


# ==================================================
# main entry
# ==================================================
def dialog_llm_chat(
    user_text: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
    context: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> DialogResult:
    user_text = (user_text or "").strip()
    if not user_text:
        return DialogResult(reply="말씀을 다시 한 번 부탁드릴게요.", action="ASK", confidence=0.4, slots=(context or {}).get("slots") or {})

    if _is_done_utterance(user_text):
        return DialogResult(reply=FAREWELL_TEXT, action="DONE", confidence=1.0, slots=(context or {}).get("slots") or {})

    ctx = context or {}
    hard_limit = int(ctx.get("hard_turn_limit", DEFAULT_HARD_TURN_LIMIT) or DEFAULT_HARD_TURN_LIMIT)
    turn_count_user = int(ctx.get("turn_count_user", 0) or 0)

    first_intent = _norm_intent_name(ctx.get("first_intent"))
    current_intent = _norm_intent_name(ctx.get("current_intent") or first_intent)

    ctx_slots = ctx.get("slots") or {}
    if not isinstance(ctx_slots, dict):
        ctx_slots = {}

    if turn_count_user >= hard_limit:
        return DialogResult(
            reply="여러 번 확인했지만 현재 정보로는 문제 상황을 정확히 특정하기 어렵습니다. 관리자를 호출해 도움을 받아주세요. " + FAREWELL_TEXT,
            action="ESCALATE_DONE",
            confidence=1.0,
            slots=ctx_slots,
        )

    heuristic_slots = _heuristic_extract_slots(user_text, current_intent)
    merged = _merge_slots(ctx_slots, heuristic_slots)

    missing = _missing_required_slots(current_intent, merged)
    next_missing = _pick_next_missing(current_intent, missing)

    if not missing:
        preferred_docs = ctx.get("preferred_docs")
        if not isinstance(preferred_docs, list) or not preferred_docs:
            preferred_docs = INTENT_TO_DOCS.get(current_intent, [])

        manual_text = ""
        try:
            hits = _rag.retrieve(
                query=f"{current_intent} {merged.get('where') or ''} {merged.get('symptom') or ''} {merged.get('error_message') or ''} {user_text}",
                preferred_docs=preferred_docs,
                hard_filter=True if preferred_docs else False,
                debug=debug,
            )
            for h in (hits or [])[:5]:
                raw = getattr(h, "text", "") or ""
                if not raw:
                    continue
                if "SOLVE_TEMPLATE" in raw or "해결 안내 문장 템플릿" in raw:
                    manual_text = raw
                    break
            if not manual_text and hits:
                manual_text = getattr(hits[0], "text", "") or ""
        except Exception as e:
            if debug:
                print(f"[RAG] failed: {e}")

        solved = _build_solve_reply_from_manual(current_intent, merged, manual_text)
        if solved:
            return DialogResult(reply=solved, action="SOLVE", confidence=1.0, slots=merged)

        return DialogResult(
            reply="현재 메뉴얼에서 해당 상황의 해결 안내를 찾지 못했습니다. 관리자를 호출해 도움을 받아주세요. " + FAREWELL_TEXT,
            action="ESCALATE_DONE",
            confidence=0.9,
            slots=merged,
        )

    reply, upd_slots, new_intent, conf, raw = _llm_ask(
        user_text,
        history=history,
        intent_name=current_intent,
        slots=merged,
        missing_slots=missing,
        next_missing=next_missing,
        debug=debug,
    )
    merged2 = _merge_slots(merged, upd_slots)

    if new_intent and new_intent != current_intent:
        current_intent = new_intent

    if debug:
        try:
            print("[DIALOG-ASK-RAW]", raw)
            print("[DIALOG-ASK-SLOTS]", json.dumps(merged2, ensure_ascii=False))
            print("[DIALOG-ASK-MISSING]", _missing_required_slots(current_intent, merged2))
        except Exception:
            pass

    return DialogResult(
        reply=reply,
        action="ASK",
        confidence=conf,
        slots=merged2,
        new_intent=new_intent,
        raw=raw if debug else None,
    )
