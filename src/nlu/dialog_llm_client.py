from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Tuple

import requests

# ==================================================
# 설정 및 환경 변수
# ==================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "30"))
# ==================================================
# 프롬프트 
# ==================================================
_SYSTEM_PROMPT = """
너는 주차장 시스템의 '지능형 상담 전문가'입니다. 모든 답변은 반드시 친절한 존댓말로 작성하세요.
사용자의 문제를 끝까지 추적하여 매뉴얼에 기반한 최적의 해결책을 제시하고 대화를 마무리하는 것이 목표입니다.
사용자가 이미 말한 내용을 질문으로 다시 묻지 말고 필요한 정보를 위한 질문을 해라.

### [핵심 대화 원칙]
1. 공감적 확인: 사용자가 말한 증상(symptom)을 문장 서두에 언급하여 잘 이해했음을 보여라.
   (예: "결제 중에 기계가 갑자기 멈춰서 당황스러우시겠어요.")
2. 중복 질문 금지: 사용자가 이미 답변한 내용(slots에 저장된 정보)을 절대 다시 묻지 마라.
3. 타겟 질문: 매뉴얼의 '필수 슬롯(required_slots)' 중 비어 있는 정보만 구체적으로 질문하고,
   "무슨 문제인가요?" 같은 포괄적인 질문은 피하고 "현재 계신 곳이 사전 정산기인가요, 아니면 출구인가요?"처럼 구체적으로 물어라.
4. 매뉴얼 엄수: 매뉴얼에 없는 해결책을 임의로 제안하지 마세요.

### [데이터 구조]
1. [USER_TEXT]: 사용자의 현재 발화
2. [CONTEXT]: 현재 파악된 정보(slots), 의도, 대화 이력
3. [MANUAL_CONTEXT]: RAG로 검색된 매뉴얼 원문 (가장 중요)

### [수행 단계 및 빠른 판단 기준]
1단계: 정보 파악 및 매뉴얼 대조 - 질문과 유사한 매뉴얼을 즉시 대조하세요.
2단계: 해결 불가능성 판단 (Fast-Fail) - 아래 경우 즉시 'ESCALATE'로 결정하세요.
  - 매뉴얼과 질문의 상관관계가 없을 때
  - 매뉴얼의 해결책을 이미 시도했으나 실패했을 때
  - 현재 정보로 매뉴얼의 어떤 조항도 적용할 수 없을 때
3단계: 액션 결정
  - DONE: 해결 완료 및 종료 인사
  - SOLVE: 매뉴얼 기반 해결책 제시
  - ASK: 정보가 부족하여 추가 질문이 필요할 때
  - ESCALATE: 해결 불가하여 관리자 호출 안내

### [출력 포맷 (JSON)]
{
  "thought": "판단 근거 및 추론 과정",
  "action": "DONE | SOLVE | ASK | ESCALATE",
  "reply": "사용자에게 전달할 친절한 존댓말 응답",
  "slots": { "업데이트된 슬롯 정보" },
  "new_intent": "문맥상 의도가 바뀌었다면 해당 Intent 명칭, 아니면 null",
  "is_sufficient": true | false
}
""".strip()

# ==================================================
# 데이터 구조
# ==================================================
DialogAction = Literal["ASK", "SOLVE", "DONE", "ESCALATE"]

@dataclass
class DialogResult:
    reply: str
    action: DialogAction = "ASK"
    slots: Dict[str, Any] = None
    new_intent: Optional[str] = None
    thought: Optional[str] = None
    raw: Optional[str] = None

    def __post_init__(self):
        if self.slots is None:
            self.slots = {}

# ==================================================
# JSON 추출 및 Ollama 통신
# ==================================================
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    m = _JSON_OBJ_RE.search(text)
    if not m: return None
    try:
        return json.loads(m.group(0))
    except:
        return None

def _ollama_chat(messages: List[Dict[str, str]]) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1}
    }
    r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    return r.json()["message"]["content"]

# ==================================================
#  메인 대화 함수 / 하드 코딩 제거
# ==================================================
def dialog_llm_chat(
    user_text: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
    context: Optional[Dict[str, Any]] = None,
    manual_context: str = "", 
    debug: bool = False,
) -> DialogResult:
    
    ctx = context or {}
    current_slots = ctx.get("slots", {})
    current_intent = ctx.get("current_intent", "NONE")
    
    # LLM에게 넘길 상태 정보 구
    state_info = {
        "current_intent": current_intent,
        "slots": current_slots,
        "history_count": len(history) if history else 0
    }

    # 메시지 조립 / 프롬프트 + 매뉴얼 + 이력 + 사용자 입력
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"[CONTEXT]\n{json.dumps(state_info, ensure_ascii=False)}"},
        {"role": "user", "content": f"[MANUAL_CONTEXT]\n{manual_context}"}
    ]

    if history:
        messages.extend(history[-6:]) # 최근 대화 6개만 참고

    messages.append({"role": "user", "content": f"[USER_TEXT]\n{user_text}"})

    # LLM 호출 및 결과 파싱
    try:
        raw_output = _ollama_chat(messages)
        obj = _extract_json(raw_output) or {}
    except Exception as e:
        return DialogResult(reply="죄송합니다. 상담 시스템에 일시적인 오류가 발생했습니다.", action="ESCALATE")

    # 라마의 판단 결과를 그대로 반환
    return DialogResult(
        reply=obj.get("reply", "상황을 확인 중입니다. 잠시만 기다려 주세요."),
        action=obj.get("action", "ASK"),
        slots=obj.get("slots", current_slots),
        new_intent=obj.get("new_intent"),
        thought=obj.get("thought"),
        raw=raw_output if debug else None
    )