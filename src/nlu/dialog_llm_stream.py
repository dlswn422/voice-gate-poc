from __future__ import annotations

from typing import AsyncIterator, Dict, Any, Optional, List

from src.nlu.dialog_llm_client import dialog_llm_chat
from src.nlu.dialog_llm_client import DialogResult


async def dialog_llm_stream(
    user_text: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
    context: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> AsyncIterator[Dict[str, Any]]:
    """
    WebSocket 전용 대화 스트림 어댑터

    - 기존 dialog_llm_chat()을 그대로 재사용
    - 결과를 WS 이벤트 단위로 분해
    """

    # --------------------------------------------------
    # 1️⃣ THINKING 상태 먼저 알림
    # --------------------------------------------------
    yield {
        "type": "assistant_state",
        "state": "THINKING",
    }

    # --------------------------------------------------
    # 2️⃣ 기존 대화 로직 실행 (블로킹)
    # --------------------------------------------------
    result: DialogResult = dialog_llm_chat(
        user_text,
        history=history,
        context=context,
        debug=debug,
    )

    # --------------------------------------------------
    # 3️⃣ 답변 메시지
    # --------------------------------------------------
    if result.reply:
        yield {
            "type": "assistant_message",
            "text": result.reply,
        }

    # --------------------------------------------------
    # 4️⃣ 종료 / 상태 이벤트
    # --------------------------------------------------
    yield {
        "type": "assistant_done",
        "action": result.action,
        "slots": result.slots,
        "pending_slot": result.pending_slot,
        "new_intent": result.new_intent,
        "confidence": result.confidence,
    }
