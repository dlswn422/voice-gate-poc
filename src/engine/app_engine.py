from src.nlu.intent_embedding import detect_intent_embedding
# from src.nlu.llm_client import detect_intent_llm  # 필요 시 사용

from src.nlu.intent_schema import Intent
from src.engine.intent_logger import log_intent, log_dialog
from src.nlu.dialog_llm_client import dialog_llm_chat

import uuid
import time
import re
from typing import Dict, Any


# ==================================================
# 정책 설정
# ==================================================
SITE_ID = "parkassist_local"

DONE_COOLDOWN_SEC = 1.2
SECOND_STAGE_HARD_TURN_LIMIT = 6


# ==================================================
# 1차 확정 시 원턴 응답
# ==================================================
DEFAULT_ONE_TURN_REPLY = "현재 어떤 문의가 있으신가요?"

ONE_TURN_RESPONSES = {
    Intent.EXIT:
        "입/출차 과정에서 문제가 있는 것 같아요.\n어떤 상황인지 조금 더 말씀해 주세요.",
    Intent.ENTRY:
        "입차 중 차량 인식이나 차단기 쪽에 문제가 있어 보입니다.\n현재 상황을 조금 더 알려주세요.",
    Intent.PAYMENT:
        "주차 요금 결제와 관련된 문제로 보입니다.\n어떤 점이 불편하신지 말씀해 주세요.",
    Intent.REGISTRATION:
        "차량이나 방문자 등록 과정에서 문제가 발생한 것 같아요.\n어디에서 막혔는지 알려주세요.",
    Intent.TIME_PRICE:
        "주차 시간이나 요금에 대해 확인이 필요해 보입니다.\n궁금하신 부분을 말씀해 주세요.",
    Intent.FACILITY:
        "주차장 기기나 차단기에 이상이 있는 것 같아요.\n현재 상태를 조금 더 설명해 주세요.",
}


# ==================================================
# NONE 시 안내 메시지
# ==================================================
NONE_RETRY_TEXT = (
    "말씀을 정확히 이해하지 못했어요. "
    "어떤 도움을 원하시는지 말씀해 주세요."
)

# ==================================================
# 종료 감지
# ==================================================
DONE_KEYWORDS = [
    "됐어요", "되었습니다", "해결", "괜찮아요",
    "그만", "종료", "끝", "마칠게",
    "고마워", "감사", "안녕",
]

FAREWELL_TEXT = "네, 이용해 주셔서 감사합니다. 안녕히 가세요."


def _normalize(text: str) -> str:
    return re.sub(r"[\s\.\,\!\?]+", "", text.strip().lower())


def _is_done_utterance(text: str) -> bool:
    t = _normalize(text)
    return any(_normalize(k) in t for k in DONE_KEYWORDS)


# ==================================================
# AppEngine
# ==================================================
class AppEngine:
    """
    AppEngine (FINAL)

    ✔ 1차 Intent → 원턴 응답
    ✔ 원턴 시 키워드 UI 노출(one_turn)
    ✔ 키워드 클릭 = 일반 발화와 동일 처리
    ✔ 이후 무조건 SECOND_STAGE
    """

    def __init__(self):
        self._reset_all()

    # --------------------------------------------------
    # 내부 상태 초기화
    # --------------------------------------------------
    def _reset_all(self):
        self.session_id = None
        self.state = "FIRST_STAGE"   # FIRST_STAGE | SECOND_STAGE
        self.first_intent = None
        self.intent_log_id = None

        self.dialog_turn_index = 0
        self.dialog_history = []

        self._ignore_until_ts = 0.0
        self._just_one_turn = False

        self.second_turn_count_user = 0
        self.second_slots = {}
        self.second_pending_slot = None

    # --------------------------------------------------
    # 세션 관리
    # --------------------------------------------------
    def _start_new_session(self):
        self.session_id = str(uuid.uuid4())
        self.state = "FIRST_STAGE"
        self.dialog_turn_index = 0
        self.dialog_history = []

        self._just_one_turn = False
        self.second_turn_count_user = 0
        self.second_slots = {}
        self.second_pending_slot = None

        print(f"[ENGINE] 🆕 New session started: {self.session_id}")

    def _end_session(self, reason: str):
        print(f"[ENGINE] 🛑 Session ended ({reason}): {self.session_id}")
        self._reset_all()
        self._ignore_until_ts = time.time() + DONE_COOLDOWN_SEC

    # --------------------------------------------------
    # 로그
    # --------------------------------------------------
    def _log_dialog(self, role, content, model="system"):
        self.dialog_turn_index += 1
        log_dialog(
            intent_log_id=self.intent_log_id,
            session_id=self.session_id,
            role=role,
            content=content,
            model=model,
            turn_index=self.dialog_turn_index,
        )

        if role in ("user", "assistant"):
            self.dialog_history.append({
                "role": role,
                "content": content,
            })

    # --------------------------------------------------
    # 응답 포맷 (UI 연동 포함)
    # --------------------------------------------------
    def _make_response(
        self,
        text: str,
        *,
        conversation_state: str,
        end_session: bool = False,
        one_turn: bool = False,
        intent: str | None = None,
    ) -> Dict[str, Any]:
        return {
            "type": "assistant_message",
            "text": text,
            "conversation_state": conversation_state,
            "end_session": end_session,
            "session_id": self.session_id,
            "one_turn": one_turn,
            "intent": intent,
        }

    # --------------------------------------------------
    # SECOND_STAGE 처리
    # --------------------------------------------------
    def _handle_second_stage(self, text: str) -> Dict[str, Any]:
        if _is_done_utterance(text):
            self._log_dialog("user", text)
            self._log_dialog("assistant", FAREWELL_TEXT)
            self._end_session("user_done")

            return self._make_response(
                FAREWELL_TEXT,
                conversation_state="ENDED",
                end_session=True,
            )

        self._log_dialog("user", text)

        res = dialog_llm_chat(
            text,
            history=self.dialog_history,
            context={
                "session_id": self.session_id,
                "intent": self.first_intent,
                "turn_count_user": self.second_turn_count_user,
                "hard_turn_limit": SECOND_STAGE_HARD_TURN_LIMIT,
                "slots": self.second_slots,
                "pending_slot": self.second_pending_slot,
            },
            debug=True,
        )

        reply = getattr(res, "reply", "") or "조금 더 자세히 말씀해 주실 수 있을까요?"
        self._log_dialog("assistant", reply, model="llama-3.1-8b")

        self.second_turn_count_user += 1
        self.second_slots = getattr(res, "slots", self.second_slots)
        self.second_pending_slot = getattr(res, "pending_slot", None)

        if getattr(res, "action", "") in ("DONE", "ESCALATE_DONE"):
            self._end_session("llm_done")
            return self._make_response(
                reply,
                conversation_state="ENDED",
                end_session=True,
            )

        return self._make_response(
            reply,
            conversation_state="WAITING_USER",
        )

    # --------------------------------------------------
    # 메인 엔트리
    # --------------------------------------------------
    def handle_text(self, text: Any) -> Dict[str, Any]:
        now = time.time()

        # ==================================================
        # UI 키워드 입력 처리
        # ==================================================
        if isinstance(text, dict) and text.get("type") == "ui_keyword":
            text = text.get("text", "")

        if not isinstance(text, str) or not text.strip():
            return self._make_response(
                "다시 한 번 말씀해 주세요.",
                conversation_state="WAITING_USER",
            )

        if now < self._ignore_until_ts:
            return self._make_response(
                "",
                conversation_state="WAITING_USER",
            )

        if not self.session_id:
            self._start_new_session()

        # ==================================================
        # 원턴 이후 → SECOND_STAGE
        # ==================================================
        if self._just_one_turn:
            self.state = "SECOND_STAGE"
            self._just_one_turn = False
            return self._handle_second_stage(text)

        # ==================================================
        # FIRST_STAGE
        # ==================================================
        if self.state == "FIRST_STAGE":
            result = detect_intent_embedding(text)

            self.intent_log_id = log_intent(
                utterance=text,
                predicted_intent=result.intent.value,
                predicted_confidence=result.confidence,
                source="kiosk",
                site_id=SITE_ID,
            )

            self.first_intent = result.intent.value
            self._log_dialog("user", text)

            # --------------------------------------------------
            # NONE → 원턴 + 키워드
            # --------------------------------------------------
            if result.intent == Intent.NONE:
                self._log_dialog("assistant", NONE_RETRY_TEXT)
                self._just_one_turn = True

                return self._make_response(
                    NONE_RETRY_TEXT,
                    conversation_state="WAITING_USER",
                    one_turn=True,
                    intent=Intent.NONE.value,
                )

            # --------------------------------------------------
            # 확정 Intent → 원턴 응답
            # --------------------------------------------------
            # reply = DEFAULT_ONE_TURN_REPLY
            
            reply = ONE_TURN_RESPONSES.get(
                result.intent,
                "현재 어떤 문제가 발생했나요?"
            )

            self._log_dialog("assistant", reply)
            self._just_one_turn = True

            return self._make_response(
                reply,
                conversation_state="WAITING_USER",
                one_turn=True,
                intent=self.first_intent,
            )

        # ==================================================
        # SECOND_STAGE
        # ==================================================
        return self._handle_second_stage(text)
