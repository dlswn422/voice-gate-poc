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
ONE_TURN_RESPONSES = {
    Intent.EXIT: "출차하려면 요금 정산이 완료되어야 차단기가 열립니다. 현재 어떤 문제가 발생했나요?",
    Intent.ENTRY: "입차 시 차량이 인식되면 차단기가 자동으로 열립니다. 현재 어떤 문제가 발생했나요?",
    Intent.PAYMENT: "주차 요금은 정산기나 출구에서 결제하실 수 있습니다. 현재 어떤 문제가 발생했나요?",
    Intent.REGISTRATION: "차량이나 방문자 등록은 키오스크에서 진행하실 수 있습니다. 현재 어떤 문제가 발생했나요?",
    Intent.TIME_PRICE: "주차 시간과 요금은 키오스크 화면에서 확인하실 수 있습니다. 현재 어떤 문제가 발생했나요?",
    Intent.FACILITY: "기기나 차단기에 이상이 있는 경우 관리실 도움을 받으실 수 있습니다. 현재 어떤 문제가 발생했나요?",
}

# ==================================================
# NONE 시 안내 메시지 (원래 기능 유지)
# ==================================================
NONE_RETRY_TEXT = (
    "말씀을 정확히 이해하지 못했어요. "
    "출차, 결제, 등록 중 어떤 도움을 원하시는지 말씀해 주세요."
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

    ✔ Intent 확정 → 원턴
    ✔ Intent NONE → 안내 메시지 (원래 UX)
    ✔ NONE 다음 발화 → 2차 대화 승격
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
        self._just_one_turn = False  # ⭐ 핵심 플래그

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
    # 응답 포맷
    # --------------------------------------------------
    def _make_response(
        self,
        text: str,
        *,
        conversation_state: str,
        end_session: bool = False,
    ) -> Dict[str, Any]:
        return {
            "type": "assistant_message",
            "text": text,
            "conversation_state": conversation_state,
            "end_session": end_session,
            "session_id": self.session_id,
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
    def handle_text(self, text: str) -> Dict[str, Any]:
        now = time.time()

        if not text or not text.strip():
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
        # ⭐ 원턴 이후 (NONE 포함) → SECOND_STAGE
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
            # 🔹 NONE → 안내 메시지 (원래 UX 유지)
            # --------------------------------------------------
            if result.intent == Intent.NONE:
                self._log_dialog("assistant", NONE_RETRY_TEXT)

                # ⭐ 다음 턴에서 2차로 올리기 위한 플래그
                self._just_one_turn = True

                return self._make_response(
                    NONE_RETRY_TEXT,
                    conversation_state="WAITING_USER",
                )

            # --------------------------------------------------
            # 🔹 확정 Intent → 원턴 응답
            # --------------------------------------------------
            reply = ONE_TURN_RESPONSES.get(
                result.intent,
                "현재 어떤 문제가 발생했나요?"
            )

            self._log_dialog("assistant", reply)
            self._just_one_turn = True

            return self._make_response(
                reply,
                conversation_state="WAITING_USER",
            )

        # ==================================================
        # SECOND_STAGE
        # ==================================================
        return self._handle_second_stage(text)
