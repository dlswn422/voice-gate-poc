from src.nlu.llm_client import detect_intent_llm
from src.nlu.intent_schema import Intent
from src.engine.intent_logger import log_intent, log_dialog
from src.nlu.dialog_llm_client import dialog_llm_chat

import uuid
import time
import re


# ==================================================
# 정책 설정
# ==================================================
CONFIDENCE_THRESHOLD = 0.75
SITE_ID = "parkassist_local"

IDLE_TIMEOUT_SEC = 15.0
DONE_COOLDOWN_SEC = 1.2


# ==================================================
# 원턴 응답 (⚠️ 질문형)
# ==================================================
ONE_TURN_RESPONSES = {
    Intent.EXIT: "출차하려면 요금 정산이 완료되어야 차단기가 열립니다. 혹시 정산은 이미 하셨나요?",
    Intent.ENTRY: "입차 시 차량이 인식되면 차단기가 자동으로 열립니다. 차량이 인식되지 않았다면 잠시 정차해 주세요.",
    Intent.PAYMENT: "주차 요금은 정산기나 출구에서 결제하실 수 있습니다. 이미 결제를 진행하셨나요?",
    Intent.REGISTRATION: "차량이나 방문자 등록은 키오스크에서 진행하실 수 있습니다. 아직 등록 전이신가요?",
    Intent.TIME_PRICE: "주차 시간과 요금은 키오스크 화면에서 확인하실 수 있습니다. 어느 부분이 궁금하신가요?",
    Intent.FACILITY: "기기나 차단기에 이상이 있는 경우 관리실 도움을 받으실 수 있습니다. 현재 어떤 문제가 발생했나요?",
}

NONE_RETRY_TEXT = (
    "말씀을 정확히 이해하지 못했어요. "
    "출차, 결제, 등록 중 어떤 도움을 원하시는지 말씀해 주세요."
)

DONE_KEYWORDS = [
    "됐어요", "되었습니다", "해결", "괜찮아요",
    "그만", "종료", "끝", "마칠게",
    "고마워", "감사", "안녕",
]

FAREWELL_TEXT = "네, 해결되셨다니 다행입니다. 이용해 주셔서 감사합니다. 안녕히 가세요."


# ==================================================
# 유틸
# ==================================================
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
    ✔ 1차 의도 확정 후 세션 동안 의도 고정
    ✔ 원턴(질문) → 다음 발화는 무조건 멀티턴
    ✔ FIRST_STAGE로 되돌아가지 않음
    ✔ idle-timeout 입력 중 종료 버그 해결
    """

    def __init__(self):
        self.state = "FIRST_STAGE"

        self.session_id = None
        self.first_intent = None
        self.intent_log_id = None

        self.dialog_turn_index = 0
        self.dialog_history = []

        self._none_retry_count = 0
        self._ignore_until_ts = 0.0
        self._last_activity_ts = 0.0

        self._last_handled_utterance_id = None
        self._just_one_turn = False

    # --------------------------------------------------
    # 세션 시작
    # --------------------------------------------------
    def _start_new_session(self):
        self.session_id = str(uuid.uuid4())
        self.state = "FIRST_STAGE"

        self.first_intent = None
        self.intent_log_id = None
        self.dialog_turn_index = 0
        self.dialog_history = []

        self._none_retry_count = 0
        self._just_one_turn = False
        self._last_activity_ts = time.time()

        print(f"[ENGINE] 🆕 New session started: {self.session_id}")

    # --------------------------------------------------
    # 세션 종료
    # --------------------------------------------------
    def end_session(self, reason: str = ""):
        print(f"[ENGINE] 🛑 Session ended ({reason}): {self.session_id}")

        self.session_id = None
        self.state = "FIRST_STAGE"
        self.first_intent = None
        self.intent_log_id = None
        self.dialog_turn_index = 0
        self.dialog_history = []

        self._none_retry_count = 0
        self._just_one_turn = False
        self._last_handled_utterance_id = None

    # --------------------------------------------------
    # idle timeout (외부 watchdog용)
    # --------------------------------------------------
    def check_idle_timeout(self):
        if self.session_id and time.time() - self._last_activity_ts >= IDLE_TIMEOUT_SEC:
            self.end_session(reason="idle-timeout")

    # --------------------------------------------------
    # confidence
    # --------------------------------------------------
    def calculate_confidence(self, text: str, intent: Intent) -> float:
        score = 0.4
        KEYWORDS = {
            Intent.EXIT: ["출차", "나가", "차단기"],
            Intent.ENTRY: ["입차", "들어가"],
            Intent.PAYMENT: ["결제", "요금", "정산"],
            Intent.REGISTRATION: ["등록", "번호판"],
            Intent.TIME_PRICE: ["시간", "요금"],
            Intent.FACILITY: ["기계", "고장", "이상"],
            Intent.COMPLAINT: ["왜", "안돼", "짜증"],
        }
        hits = sum(1 for k in KEYWORDS.get(intent, []) if k in text)
        score += 0.35 if hits else 0.15
        score += 0.05 if len(text) <= 4 else 0.2
        return round(min(score, 1.0), 2)

    # --------------------------------------------------
    # dialog log
    # --------------------------------------------------
    def _log_dialog(self, role, content, model="stt"):
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
    # SECOND_STAGE
    # --------------------------------------------------
    def _handle_second_stage(self, text):
        if _is_done_utterance(text):
            self._log_dialog("user", text)
            self._log_dialog("assistant", FAREWELL_TEXT, model="system")
            print(f"[DIALOG] {FAREWELL_TEXT}")
            self.end_session(reason="done")
            self._ignore_until_ts = time.time() + DONE_COOLDOWN_SEC
            return

        self._log_dialog("user", text)

        res = dialog_llm_chat(
            text,
            history=self.dialog_history,
            context={
                "session_id": self.session_id,
                "intent": self.first_intent,
            },
            debug=True,
        )

        reply = getattr(res, "reply", "") or "조금 더 자세히 말씀해 주실 수 있을까요?"
        self._log_dialog("assistant", reply, model="llama-3.1-8b")
        print(f"[DIALOG] {reply}")

    # --------------------------------------------------
    # STT 엔트리포인트
    # --------------------------------------------------
    def handle_text(self, text: str, *, utterance_id: str | None = None):
        now = time.time()

        if not text or not text.strip():
            return
        if now < self._ignore_until_ts:
            return

        # ✅ 입력이 들어왔으므로 활동 시간 갱신
        self._last_activity_ts = now

        # STT 중복 방지
        if utterance_id and utterance_id == self._last_handled_utterance_id:
            print("[ENGINE] ⚠️ duplicated utterance ignored")
            return
        self._last_handled_utterance_id = utterance_id

        if not self.session_id:
            self._start_new_session()

        print("=" * 50)
        print(f"[ENGINE] State={self.state}")
        print(f"[ENGINE] Text={text}")

        # ==================================================
        # 🔥 원턴 직후 후속 발화 → 무조건 멀티턴
        # ==================================================
        if self._just_one_turn:
            print("[ENGINE] 🔁 one-turn follow-up → SECOND_STAGE")
            self.state = "SECOND_STAGE"
            self._just_one_turn = False
            self._handle_second_stage(text)
            return

        # --------------------------------------------------
        # FIRST_STAGE
        # --------------------------------------------------
        if self.state == "FIRST_STAGE":
            result = detect_intent_llm(text)
            result.confidence = self.calculate_confidence(text, result.intent)

            print(f"[ENGINE] Intent={result.intent.name}, confidence={result.confidence:.2f}")

            self.intent_log_id = log_intent(
                utterance=text,
                predicted_intent=result.intent.value,
                predicted_confidence=result.confidence,
                source="kiosk",
                site_id=SITE_ID,
            )

            self.first_intent = result.intent.value
            self._log_dialog("user", text)

            if result.intent == Intent.NONE:
                self._none_retry_count += 1
                self._log_dialog("assistant", NONE_RETRY_TEXT, model="system")
                print(f"[ONE-TURN] {NONE_RETRY_TEXT}")
                return

            if result.intent == Intent.COMPLAINT or result.confidence < CONFIDENCE_THRESHOLD:
                self.state = "SECOND_STAGE"
                self._handle_second_stage(text)
                return

            reply = ONE_TURN_RESPONSES.get(result.intent)
            self._log_dialog("assistant", reply, model="system")
            print(f"[ONE-TURN] {reply}")
            self._just_one_turn = True
            return

        # --------------------------------------------------
        # SECOND_STAGE
        # --------------------------------------------------
        if self.state == "SECOND_STAGE":
            self._handle_second_stage(text)
            return
