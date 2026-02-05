from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import requests

from src.nlu.intent_schema import Intent
from src.rag.manual_rag import ManualRAG


# ==================================================
# Config
# ==================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "30"))

DEFAULT_HARD_TURN_LIMIT = int(os.getenv("SECOND_STAGE_HARD_TURN_LIMIT", "6") or 6)

# ASK(자유도↑) / SOLVE(자유도↓)
ASK_TEMPERATURE = float(os.getenv("DIALOG_ASK_TEMPERATURE", "0.7") or 0.7)
SOLVE_TEMPERATURE = float(os.getenv("DIALOG_SOLVE_TEMPERATURE", "0.15") or 0.15)

# 모델이 JSON이 아니라 잡다한 텍스트를 섞어도 첫 JSON만 뽑아 쓰기
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


DialogAction = Literal["ASK", "SOLVE", "DONE", "FAILSAFE", "ESCALATE_DONE"]

SLOT_KEYS = [
    "symptom",        # 문제 현상 (예: 승인 실패, 무반응, 통신 불가, 무료시간 미적용)
    "where",          # 위치/기기 (예: 출구 정산기, 사전 정산기(키오스크), 입구, 차단기)
    "when",           # 언제/어떤 시점
    "error_message",  # 화면 문구/오류 코드
    "attempted",      # 이미 해본 조치
    "card_or_device", # 카드/단말/폰/QR 등 관련 정보
]

ALLOWED_INTENTS = {
    "EXIT", "ENTRY", "PAYMENT", "REGISTRATION", "TIME_PRICE", "FACILITY", "COMPLAINT", "NONE"
}


# ==================================================
# Result
# ==================================================
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


# ==================================================
# Done utterance (false-positive 방지)
#   - "안됐어요" 같은 부정문에 '됐'이 포함되어도 종료로 처리하지 않게
# ==================================================
_DONE_HARD = {
    "종료", "끝", "그만", "마칠게", "이만", "끊을게",
    "됐어요", "됐어", "됐습니다", "해결", "해결됨", "해결됐", "정상", "문제없",
}
_DONE_SOFT = {"고마워", "감사", "안녕", "수고", "잘가", "바이"}


def _normalize(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[\s\.\,\!\?\u3002\uFF0E\uFF0C\uFF01\uFF1F]+", "", t)
    return t


def _is_done_utterance(text: str) -> bool:
    t = _normalize(text)
    if not t:
        return False

    # 명확한 감사/인사 → 종료로 허용 (부정문이어도)
    if any(k in t for k in map(_normalize, _DONE_SOFT)):
        return True

    # 부정형: 안됐/안돼/안되 등은 종료로 오인하면 안 됨
    if "안됐" in t or "안되" in t or "안돼" in t:
        return False

    # 완전 일치 / 말끝 일치만 허용 (부분 포함으로는 종료 처리 X)
    for kw in _DONE_HARD:
        k = _normalize(kw)
        if t == k or t.endswith(k):
            return True

    return False


# ==================================================
# Slot utils
# ==================================================
def _merge_slots(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    for k in SLOT_KEYS:
        if k in (update or {}):
            v = update.get(k)
            if isinstance(v, str):
                v = v.strip()
                if v == "":
                    v = None
            out[k] = v
    return out


def _missing_required(required_slots: List[str], slots: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    for k in required_slots or []:
        if k not in SLOT_KEYS:
            continue
        v = (slots or {}).get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            missing.append(k)
    return missing


# ==================================================
# Heuristic prefill (가벼운 보조)
#   - 질문/해답을 하드코딩하지 않고, 슬롯 힌트만 채운다.
# ==================================================
_WHERE_PATTERNS = [
    (r"(출구정산기|출구\s*정산기|출구)", "출구 정산기"),
    (r"(사전정산기|사전\s*정산기|키오스크|무인정산기)", "사전 정산기(키오스크)"),
    (r"(입구)", "입구"),
    (r"(차단기|차단봉|게이트)", "차단기"),
]
_ERRMSG_RE = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{5,60})[\"'“”‘’]")


def _heuristic_extract_slots(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    out: Dict[str, Any] = {}

    # where
    for pat, norm in _WHERE_PATTERNS:
        if re.search(pat, t):
            out["where"] = norm
            break

    # error message quoted
    m = _ERRMSG_RE.search(t)
    if m:
        out["error_message"] = m.group(1).strip()

    # common symptoms
    if any(k in t for k in ["승인 실패", "승인실패"]):
        out["symptom"] = "승인 실패"
    elif any(k in t for k in ["통신할 수 없", "서버와 통신", "네트워크", "연결"]):
        out["symptom"] = "통신 불가"
        if "error_message" not in out and "서버와 통신할 수 없습니다" in t:
            out["error_message"] = "서버와 통신할 수 없습니다"
    elif any(k in t for k in ["무반응", "먹통", "버튼이 안", "눌러도"]):
        out["symptom"] = "무반응/먹통"
    elif any(k in t for k in ["무료", "할인", "감면"]) and any(k in t for k in ["적용 안", "적용이 안", "미적용", "안됐"]):
        out["symptom"] = "무료/할인 시간 미적용"
    elif any(k in t for k in ["안 열", "안올라", "안 올라", "열리지"]):
        out["symptom"] = "차단기 미동작"
    elif any(k in t for k in ["등록이 안", "등록 안", "등록이 안되", "등록이 안 돼"]):
        out["symptom"] = "등록 실패"

    return out


# ==================================================
# ManualRAG
# ==================================================
_rag = ManualRAG()


def _build_manual_context(hits: List[Any], max_chars: int = 1800) -> str:
    """
    모델이 메뉴얼을 '참고'해서 답하도록, 관련 청크를 짧게 제공.
    """
    if not hits:
        return ""
    chunks: List[str] = []
    for c in hits[:3]:
        txt = getattr(c, "text", "") or ""
        if not txt:
            continue
        txt = txt.strip()
        txt = re.sub(r"\r\n", "\n", txt)
        txt = re.sub(r"[ \t]+", " ", txt)
        chunks.append(txt)
    joined = "\n\n---\n\n".join(chunks).strip()
    if len(joined) > max_chars:
        joined = joined[:max_chars].rstrip() + "…"
    return joined


# ==================================================
# LLM call + parse
# ==================================================
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


def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = JSON_OBJ_RE.search(text)
    if not m:
        return None
    blob = m.group(0).strip()
    try:
        return json.loads(blob)
    except Exception:
        # trailing commas / single quotes 등 간단 보정
        blob2 = blob.replace("'", '"')
        blob2 = re.sub(r",\s*}", "}", blob2)
        blob2 = re.sub(r",\s*]", "]", blob2)
        try:
            return json.loads(blob2)
        except Exception:
            return None


def _norm_intent_name(x: Optional[str]) -> str:
    if not x:
        return "NONE"
    s = str(x).strip().upper()
    if s.startswith("INTENT."):
        s = s.split(".", 1)[-1]
    return s if s in ALLOWED_INTENTS else "NONE"


def _fallback_question(missing: List[str], intent_name: str) -> str:
    # 정말 실패했을 때만 안전 질문
    if not missing:
        return "상황을 조금만 더 자세히 말씀해 주실 수 있을까요?"
    k = missing[0]
    if k == "where":
        return "문제가 발생한 위치/기기가 어디인가요? (예: 출구/키오스크/차단기)"
    if k == "symptom":
        return "어떤 현상이 문제인가요? (예: 승인 실패/무반응/오류 문구/미적용 등)"
    if k == "error_message":
        return "화면에 뜨는 오류 문구나 코드가 있나요? 그대로 읽어 주세요."
    if k == "attempted":
        return "이미 시도해 보신 조치가 있나요? (예: 재시도/재부팅/다른 카드/다시 태그)"
    return "조금만 더 자세히 말씀해 주세요."


# ==================================================
# Prompts
# ==================================================
SYSTEM_PROMPT_ASK = """
너는 주차장 무인정산/차단기/출입 시스템 상담원이다.
목표는 '슬롯'을 채우기 위해 사용자에게 **한 번에 하나의 질문**만 하는 것이다.

규칙:
- 사용자의 발화를 바탕으로 슬롯을 업데이트한다.
- 아직 부족한 슬롯을 채우기 위한 질문을 **딱 1개**만 한다.
- 질문은 자연스럽게, 상황에 맞게 바꿔 말해도 된다(하드코딩된 문구 반복 금지).
- 해결책/조치 안내는 절대 먼저 하지 않는다(ASK 단계).
- 출력은 반드시 JSON 1개로만 한다.

출력 JSON 스키마:
{
  "action": "ASK",
  "reply": "<사용자에게 할 질문 1개>",
  "slots": { "symptom": ..., "where": ..., "when": ..., "error_message": ..., "attempted": ..., "card_or_device": ... },
  "new_intent": "<필요하면 다른 의도로 전환: EXIT/ENTRY/PAYMENT/REGISTRATION/TIME_PRICE/FACILITY/COMPLAINT/NONE 또는 null>",
  "confidence": 0.0~1.0
}
""".strip()


SYSTEM_PROMPT_SOLVE = """
너는 주차장 무인정산/차단기/출입 시스템 상담원이다.
목표는 사용자 정보(슬롯)와 [MANUAL_CONTEXT]를 참고해서 **해결 안내**를 제공하는 것이다.

규칙:
- 답변은 반드시 [MANUAL_CONTEXT]의 해결 안내(템플릿/조치 순서)를 우선 반영한다.
- 임의로 새로운 조치를 만들어내지 말고, 메뉴얼에 없는 내용은 '가능성' 정도로만 말한다.
- 한국어로, 간결하고 실행 가능한 단계로 안내한다.
- 마지막에 "위 안내로 해결되셨나요? 다른 문제가 더 있으시면 말씀해 주세요." 를 포함한다.
- 출력은 반드시 JSON 1개로만 한다.

출력 JSON 스키마:
{
  "action": "SOLVE",
  "reply": "<해결 안내 답변>",
  "slots": {...},
  "new_intent": "<필요하면 다른 의도로 전환 ... 또는 null>",
  "confidence": 0.0~1.0
}
""".strip()


# ==================================================
# Main entry
# ==================================================
def dialog_llm_chat(
    user_text: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
    context: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> DialogResult:
    """
    context keys (권장):
      - first_intent: str
      - current_intent: str
      - required_slots: list[str]
      - slots: dict
      - hard_turn_limit: int
      - turn_count_user: int   # SECOND_STAGE에서 사용자 발화 누적(현재 입력 전 기준)
    """
    user_text = (user_text or "").strip()
    if not user_text:
        return DialogResult(reply="말씀을 다시 한 번 부탁드릴게요.", action="ASK", confidence=0.4, slots=context.get("slots") if context else {})

    if _is_done_utterance(user_text):
        return DialogResult(reply="이용해 주셔서 감사합니다. 안전운전하세요.", action="DONE", confidence=1.0, slots=context.get("slots") if context else {})

    ctx = context or {}
    hard_limit = int(ctx.get("hard_turn_limit", DEFAULT_HARD_TURN_LIMIT) or DEFAULT_HARD_TURN_LIMIT)
    turn_count_user = int(ctx.get("turn_count_user", 0) or 0)

    first_intent = _norm_intent_name(ctx.get("first_intent"))
    current_intent = _norm_intent_name(ctx.get("current_intent") or first_intent)

    ctx_slots = ctx.get("slots") if isinstance(ctx.get("slots"), dict) else {}
    required_slots = ctx.get("required_slots") if isinstance(ctx.get("required_slots"), list) else ["where", "symptom"]

    # 1) 턴 제한(관리자 호출 + 종료)
    if turn_count_user >= hard_limit:
        return DialogResult(
            reply="여러 번 확인했지만 현재 정보로는 문제 상황을 정확히 특정하기 어렵습니다. 관리자를 호출해 도움을 받아주세요. 이용해 주셔서 감사합니다. 안전운전하세요.",
            action="ESCALATE_DONE",
            confidence=1.0,
            slots=ctx_slots,
            new_intent=current_intent,
        )

    # 2) 휴리스틱 선반영
    merged_slots = _merge_slots(ctx_slots, _heuristic_extract_slots(user_text))
    missing = _missing_required(required_slots, merged_slots)

    # 3) RAG
    manual_context = ""
    try:
        query = f"{current_intent} {merged_slots.get('where') or ''} {merged_slots.get('symptom') or ''} {merged_slots.get('error_message') or ''} {user_text}"
        hits = _rag.retrieve(query=query, preferred_docs=ctx.get("preferred_docs"), hard_filter=False, debug=debug)
        manual_context = _build_manual_context(hits or [])
    except Exception as e:
        if debug:
            print(f"[RAG] failed: {e}")

    # 4) ASK vs SOLVE 결정
    mode: DialogAction = "ASK" if missing else "SOLVE"

    # 5) LLM 호출
    system_prompt = SYSTEM_PROMPT_ASK if mode == "ASK" else SYSTEM_PROMPT_SOLVE
    temperature = ASK_TEMPERATURE if mode == "ASK" else SOLVE_TEMPERATURE

    msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    state_summary = {
        "first_intent": first_intent,
        "current_intent": current_intent,
        "required_slots": required_slots,
        "missing_slots": missing,
        "slots": merged_slots,
        "turn_count_user": turn_count_user,
    }
    msgs.append({"role": "user", "content": f"[STATE]\n{json.dumps(state_summary, ensure_ascii=False)}"})
    if manual_context:
        msgs.append({"role": "user", "content": f"[MANUAL_CONTEXT]\n{manual_context}"})

    if history:
        for m in history[-8:]:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})

    msgs.append({"role": "user", "content": user_text})

    raw = ""
    try:
        raw = _ollama_chat(msgs, temperature=temperature)
        obj = _extract_json_obj(raw) or {}
    except Exception as e:
        if debug:
            print(f"[DIALOG] LLM call failed: {e}")
        obj = {}

    # 6) 파싱/정규화
    action = str(obj.get("action") or "").strip().upper()
    reply = str(obj.get("reply") or "").strip()
    new_intent = _norm_intent_name(obj.get("new_intent")) if obj.get("new_intent") else None
    conf = obj.get("confidence")
    try:
        conf = float(conf)
    except Exception:
        conf = 0.7

    upd_slots = obj.get("slots") if isinstance(obj.get("slots"), dict) else {}
    merged_slots = _merge_slots(merged_slots, upd_slots)

    # 7) 정책 강제: missing이면 무조건 ASK
    missing2 = _missing_required(required_slots, merged_slots)
    if missing2:
        action = "ASK"
        if not reply:
            reply = _fallback_question(missing2, current_intent)
    else:
        action = "SOLVE"
        if not reply:
            reply = "확인한 정보 기준으로 안내드리겠습니다.\n위 안내로 해결되셨나요? 다른 문제가 더 있으시면 말씀해 주세요."

    reply = reply.replace("\r\n", "\n").strip()

    return DialogResult(
        reply=reply,
        action=action,  # type: ignore[arg-type]
        confidence=max(0.0, min(1.0, conf)),
        slots=merged_slots,
        new_intent=new_intent,
        raw=raw if debug else None,
    )
