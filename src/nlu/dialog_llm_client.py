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

FAREWELL_TEXT = "이용해 주셔서 감사합니다. 안전운전하세요."


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


# intent → manuals 파일 후보 매핑 (일부 intent는 시스템 내부에서만 쓰일 수 있음)
INTENT_TO_DOCS: Dict[str, List[str]] = {
    "PAYMENT_ISSUE": ["payment_card_fail.md", "discount_free_time_issue.md"],
    "PRICE_INQUIRY": ["price_inquiry.md"],
    "TIME_ISSUE": ["discount_free_time_issue.md"],
    "REGISTRATION_ISSUE": ["visit_registration_fail.md"],
    "NETWORK_ISSUE": ["network_terminal_down.md", "network_down.md"],
    "ENTRY_FLOW_ISSUE": ["entry_gate_not_open.md", "lpr_mismatch_or_no_entry_record.md"],
    "EXIT_FLOW_ISSUE": ["exit_gate_not_open.md", "exit_barrier_issue.md", "lpr_mismatch_or_no_entry_record.md"],
    "BARRIER_PHYSICAL_FAULT": ["barrier_physical_fault.md"],
    # HELP_REQUEST는 하드필터 금지 + 키워드 기반 후보 추정으로 처리
}

_rag = ManualRAG()


def _infer_docs_for_help_request(user_text: str) -> List[str]:
    """
    HELP_REQUEST는 범위가 너무 넓어서 특정 문서로 하드필터하면 빗나가기 쉬움.
    -> 키워드 기반으로 "우선순위 있는 후보 리스트"를 만들고,
       retrieve()에서 순서 기반 boost를 줘서 top1이 더 잘 맞게 한다.
    """
    t = _normalize(user_text)
    docs: List[str] = []

    def add(*names: str) -> None:
        for n in names:
            if n not in docs:
                docs.append(n)

    # 결제/정산/카드
    if any(k in t for k in ["결제", "카드", "정산", "영수증", "승인", "오류", "실패", "환불"]):
        add("payment_card_fail.md")

    # 무료시간/할인/요금
    if any(k in t for k in ["무료", "할인", "시간", "추가", "연장", "요금", "금액", "가격"]):
        add("discount_free_time_issue.md", "price_inquiry.md")

    # 방문등록/차량등록
    if any(k in t for k in ["방문", "등록", "사전", "권한", "차량등록"]):
        add("visit_registration_fail.md")

    # 네트워크/통신/연결
    if any(k in t for k in ["네트워크", "통신", "연결", "인터넷", "서버", "끊", "다운", "오프라인"]):
        add("network_down.md", "network_terminal_down.md")

    # 입차/출차/게이트
    if any(k in t for k in ["입차", "들어", "진입"]):
        # 입차 전용을 앞쪽에 둬야 retrieve에서 top1이 잘 잡힘
        add("entry_gate_not_open.md", "gate_not_open.md")

    if any(k in t for k in ["출차", "나가", "퇴차", "진출"]):
        # 출차 전용을 앞쪽에 둬야 retrieve에서 top1이 잘 잡힘
        add("exit_gate_not_open.md", "exit_barrier_issue.md", "gate_not_open.md")

    # 입차 기록 없음 / 번호판 불일치 / 인식
    if any(k in t for k in ["입차기록", "기록", "없대", "없다", "번호판", "인식", "lpr", "불일치", "미인식"]):
        # 이 케이스는 매우 중요하니 앞쪽에 끌어올림
        # (이미 entry/exit 후보가 들어가 있을 수 있으니 중복 제거는 add가 처리)
        # 우선순위: lpr 문서를 entry/exit보다 앞에 두는 게 더 자연스러울 때가 많음
        # -> 이미 docs에 entry/exit가 들어갔다면, lpr을 앞으로 당겨준다.
        lpr_doc = "lpr_mismatch_or_no_entry_record.md"
        if lpr_doc not in docs:
            docs.insert(0, lpr_doc)
        else:
            # 이미 있으면 앞으로 이동
            docs.remove(lpr_doc)
            docs.insert(0, lpr_doc)

    return docs


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
        prefer_boost=0.45,  # 순서 기반 boost와 합쳐져서 HELP_REQUEST에서도 top1이 더 잘 맞음
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

    # HELP_REQUEST는 키워드 기반 후보 추정
    first_intent = (context or {}).get("first_intent") or ""
    help_candidates: List[str] = []
    if first_intent.strip() == "HELP_REQUEST":
        help_candidates = _infer_docs_for_help_request(user_text)

    # 최종 preferred_docs 결정 (HELP_REQUEST 후보가 있으면 그걸 우선)
    final_preferred_docs = help_candidates or preferred_docs

    # RAG 실패해도 상담 생성은 계속되게 안전장치
    try:
        manual_context = _build_manual_context(
            user_text,
            preferred_docs=final_preferred_docs if final_preferred_docs else None,
            # - 특정 intent 매핑은 하드필터(정확도↑)
            # - HELP_REQUEST는 하드필터 금지(범위 넓어서 빗나감)
            hard_filter=(True if (preferred_docs and not help_candidates) else False),
            debug=debug,
        )
    except Exception as e:
        manual_context = ""
        if debug:
            print(f"[RAG] manual build failed: {e}")

    if debug:
        print(
            f"[RAG] first_intent={(context or {}).get('first_intent')} "
            f"preferred_docs={preferred_docs} help_candidates={help_candidates}"
        )
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
