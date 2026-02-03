from src.nlu.llm_client import detect_intent_llm
from src.nlu.intent_schema import Intent
from src.engine.intent_logger import log_intent, log_dialog
from src.nlu.dialog_llm_client import dialog_llm_chat

import uuid
import time
import re


# --------------------------------------------------
# 정책 설정
# --------------------------------------------------
CONFIDENCE_THRESHOLD = 0.75
SITE_ID = "parkassist_local"


# ==================================================
# 원턴 응답 템플릿
# ==================================================
ONE_TURN_RESPONSES = {
    Intent.EXIT: "출차하려면 요금 정산이 완료되어야 차단기가 열립니다. 혹시 정산은 이미 하셨나요?",
    Intent.ENTRY: "입차 시 차량이 인식되면 차단기가 자동으로 열립니다.",
    Intent.PAYMENT: "주차 요금은 정산기나 출구에서 결제하실 수 있습니다.",
    Intent.REGISTRATION: "차량이나 방문자 등록은 키오스크에서 진행하실 수 있습니다.",
    Intent.TIME_PRICE: "주차 시간과 요금은 키오스크 화면에서 확인하실 수 있습니다.",
    Intent.FACILITY: "기기 이상 시 관리실로 문의해 주세요.",
}


DONE_KEYWORDS = [
    "됐어요", "되었습니다", "해결", "괜찮아요",
    "그만", "종료", "끝", "마칠게",
    "고마워", "감사", "안녕",
]
FAREWELL_TEXT = "네, 이용해 주셔서 감사합니다. 안녕히 가세요."
DONE_COOLDOWN_SEC = 1.2


def _normalize(text: str) -> str:
    t = text.strip().lower()
    return re.sub(r"[\s\.\,\!\?]+", "", t)


def _is_done_utterance(text: str) -> bool:
    t = _normalize(text)
    return any(k.replace(" ", "") in t for k in DONE_KEYWORDS)


class AppEngine:
    """
    상태 기반 CX AppEngine (FINAL)

    상태:
    - FIRST_STAGE  : 첫 발화
    - ONE_TURN     : 원턴 응답 직후
    - SECOND_STAGE : 멀티턴 상담
    """

    def __init__(self):
        self.state = "FIRST_STAGE"

        self.session_id = None
        self.intent_log_id = None
        self.first_intent = None

        self.dialog_turn_index = 0
        self.dialog_history = []

        self._ignore_until_ts = 0.0

    # --------------------------------------------------
    # 세션 보장
    # --------------------------------------------------
    def _ensure_session(self):
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
            self.dialog_turn_index = 0
            self.dialog_history = []

    # --------------------------------------------------
    # dialog 로그
    # --------------------------------------------------
    def _log_dialog(self, role, content, model="stt"):
        self._ensure_session()
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
            self.dialog_history.append({"role": role, "content": content})

    # --------------------------------------------------
    # 멀티턴 처리
    # --------------------------------------------------
    def _handle_second_stage(self, text: str):
        if time.time() < self._ignore_until_ts:
            return

        if _is_done_utterance(text):
            self._log_dialog("user", text)
            self._log_dialog("assistant", FAREWELL_TEXT, model="system")
            print(f"[DIALOG] {FAREWELL_TEXT}")
            self.end_session()
            self._ignore_until_ts = time.time() + DONE_COOLDOWN_SEC
            return

        self._log_dialog("user", text)

        res = dialog_llm_chat(
            text,
            history=self.dialog_history,
            context={
                "session_id": self.session_id,
                "intent_log_id": self.intent_log_id,
                "first_intent": self.first_intent,
            },
            debug=True,
        )

        reply = getattr(res, "reply", "") or ""
        self._log_dialog("assistant", reply, model="llama-3.1-8b")
        print(f"[DIALOG] {reply}")

    # --------------------------------------------------
    # STT 텍스트 엔트리포인트
    # --------------------------------------------------
    def handle_text(self, text: str):
        if not text or not text.strip():
            return
        if time.time() < self._ignore_until_ts:
            return

        print("=" * 50)
        print(f"[ENGINE] State={self.state}")
        print(f"[ENGINE] Text={text}")

        # --------------------------------------------------
        # SECOND_STAGE
        # --------------------------------------------------
        if self.state == "SECOND_STAGE":
            self._handle_second_stage(text)
            print("=" * 50)
            return

        # --------------------------------------------------
        # ONE_TURN → 무조건 멀티턴 승격
        # --------------------------------------------------
        if self.state == "ONE_TURN":
            print("[ENGINE] ONE_TURN follow-up → SECOND_STAGE")
            self.state = "SECOND_STAGE"
            self._handle_second_stage(text)
            print("=" * 50)
            return

        # --------------------------------------------------
        # FIRST_STAGE
        # --------------------------------------------------
        self._ensure_session()

        result = detect_intent_llm(text)
        print(f"[ENGINE] Intent={result.intent.name}")

        self.intent_log_id = log_intent(
            utterance=text,
            predicted_intent=result.intent.value,
            predicted_confidence=1.0,
            source="kiosk",
            site_id=SITE_ID,
        )

        self.first_intent = result.intent.value
        self._log_dialog("user", text)

        # 멀티턴 필요 조건
        if result.intent == Intent.COMPLAINT:
            print("[ENGINE] Decision: multiturn")
            self.state = "SECOND_STAGE"
            self._handle_second_stage(text)
            print("=" * 50)
            return

        # --------------------------------------------------
        # 원턴 응답
        # --------------------------------------------------
        reply = ONE_TURN_RESPONSES.get(
            result.intent,
            "안내를 도와드릴 수 있는 상담으로 연결하겠습니다."
        )

        print("[ENGINE] Decision: one-turn")
        print(f"[ONE-TURN] {reply}")

        self._log_dialog("assistant", reply, model="system")
        self.state = "ONE_TURN"

        print("=" * 50)

    # --------------------------------------------------
    # 세션 종료
    # --------------------------------------------------
    def end_session(self):
        print(f"[ENGINE] Session ended: {self.session_id}")

        self.state = "FIRST_STAGE"
        self.session_id = None
        self.intent_log_id = None
        self.first_intent = None
        self.dialog_turn_index = 0
        self.dialog_history = []
