from src.nlu.intent_embedding import detect_intent_embedding
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
# NONE 시 안내 메시지
# (※ dialog_llm_client에서 NONE 메뉴얼로도 대응 가능하지만,
#  "정확히 이해 못함" UX를 유지하고 싶으면 이 문구 유지)
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


# ==================================================
# 관리실 호출 인터럽트 (전역)
# ==================================================
CALL_ADMIN_KEYWORDS = [
    "관리실", "관리인",
    "직원", "사람",
    "불러", "불러줘", "와줘",
    "호출", "연결",
    "도와", "도움",
]


def _normalize(text: str) -> str:
    return re.sub(r"[\s\.\,\!\?]+", "", text.strip().lower())


def _is_done_utterance(text: str) -> bool:
    t = _normalize(text)
    return any(_normalize(k) in t for k in DONE_KEYWORDS)


def _is_call_admin_utterance(text: str) -> bool:
    """
    STT 특성상 문장이 깨져도
    '관리실', '직원', '사람', '불러' 류가 섞이면 바로 인터럽트
    """
    t = _normalize(text)
    return any(_normalize(k) in t for k in CALL_ADMIN_KEYWORDS)


class AppEngine:
    """
    AppEngine (REFINED)

    ✔ 전역 인터럽트 (관리실 호출)
    ✔ 1차 Intent 분류
    ✔ 질문 생성 없음(LLM이 질문하지 않음)
    ✔ 1차부터 바로 dialog_llm_client로 메뉴얼 기반 응답 생성
    ✔ 이후 무조건 SECOND_STAGE에서 동일 로직 반복
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

        self.second_turn_count_user = 0
        self.second_slots = {}              # dialog_llm_client가 slots["symptom"] 사용
        self.second_pending_slot = None     # 정책상 사용 안 하지만 호환 위해 유지

    # --------------------------------------------------
    # 세션 관리
    # --------------------------------------------------
    def _start_new_session(self):
        self.session_id = str(uuid.uuid4())
        self.state = "FIRST_STAGE"
        self.dialog_turn_index = 0
        self.dialog_history = []

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
        intent: str | None = None,
        system_action: str | None = None,
    ):
        return {
            "type": "assistant_message",
            "text": text,
            "conversation_state": conversation_state,
            "end_session": end_session,
            "session_id": self.session_id,
            "intent": intent,
            "system_action": system_action,
        }

    # --------------------------------------------------
    # 관리실 호출 처리
    # --------------------------------------------------
    def _handle_call_admin(self, text: str):
        self._log_dialog("user", text)
        reply = "관리실에 연락했습니다.\n잠시만 기다려 주세요."
        self._log_dialog("assistant", reply)

        self._end_session("call_admin")

        return self._make_response(
            reply,
            conversation_state="ENDED",
            end_session=True,
            system_action="CALL_ADMIN",
        )

    # --------------------------------------------------
    # dialog_llm_client 호출 공통
    # --------------------------------------------------
    def _run_dialog(self, text: str) -> Dict[str, Any]:
        """
        - dialog_llm_client가 symptom 슬롯을 자동 세팅함 (비어있으면 user_text로)
        - 질문(ASK) 없음 정책이므로 reply는 항상 SOLVE/ESCALATE/DONE 중 하나로 나옴
        """
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

        # 상태 업데이트
        self.second_turn_count_user += 1
        self.second_slots = getattr(res, "slots", self.second_slots) or self.second_slots
        self.second_pending_slot = getattr(res, "pending_slot", None)

        action = getattr(res, "action", "SOLVE")

        # 종료 액션 처리
        if action in ("DONE", "ESCALATE_DONE"):
            self._end_session("llm_done")
            return self._make_response(
                reply,
                conversation_state="ENDED",
                end_session=True,
                intent=self.first_intent,
            )

        return self._make_response(
            reply,
            conversation_state="WAITING_USER",
            end_session=False,
            intent=self.first_intent,
        )

    # --------------------------------------------------
    # 메인 엔트리
    # --------------------------------------------------
    def handle_text(self, text: Any) -> Dict[str, Any]:
        now = time.time()

        # UI 키워드 입력 처리
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

        # 전역 인터럽트: 관리실 호출
        if _is_call_admin_utterance(text):
            return self._handle_call_admin(text)

        # 종료 발화 (엔진 레벨에서도 처리)
        if _is_done_utterance(text):
            self._log_dialog("user", text)
            self._log_dialog("assistant", FAREWELL_TEXT)
            self._end_session("user_done")
            return self._make_response(
                FAREWELL_TEXT,
                conversation_state="ENDED",
                end_session=True,
            )

        # ==================================================
        # FIRST_STAGE: 의도 분류 후 즉시 dialog_llm_client 실행
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

            # NONE 처리: UX 유지하고 싶으면 아래처럼 리트라이만 띄움
            # (원하면 이 블록을 제거하고, 그냥 NONE intent로 _run_dialog(text) 돌려도 됨)
            if result.intent == Intent.NONE:
                self._log_dialog("assistant", NONE_RETRY_TEXT)
                # FIRST_STAGE 유지 (재질문 받고 다시 분류)
                return self._make_response(
                    NONE_RETRY_TEXT,
                    conversation_state="WAITING_USER",
                    end_session=False,
                    intent=Intent.NONE.value,
                )

            # 이제부터는 바로 SECOND_STAGE로 전환
            self.state = "SECOND_STAGE"

            # 정책: 질문 생성 없이, 1턴부터 메뉴얼 답변 생성
            return self._run_dialog(text)

        # ==================================================
        # SECOND_STAGE: 계속 dialog_llm_client 실행
        # ==================================================
        return self._run_dialog(text)
