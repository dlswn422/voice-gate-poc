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
# 원턴(즉시 응답) 템플릿
# ==================================================
ONE_TURN_RESPONSES = {
    Intent.EXIT: (
        "출차하려면 요금 정산이 완료되어야 차단기가 열립니다. "
        "혹시 정산은 이미 하셨나요?"
    ),
    Intent.ENTRY: (
        "입차 시 차량이 인식되면 차단기가 자동으로 열립니다. "
        "차량이 인식되지 않았다면 잠시 정차해 주세요."
    ),
    Intent.PAYMENT: (
        "주차 요금은 정산기나 출구에서 결제하실 수 있습니다. "
        "이미 결제를 진행하셨나요?"
    ),
    Intent.REGISTRATION: (
        "차량이나 방문자 등록은 키오스크에서 진행하실 수 있습니다. "
        "아직 등록 전이신가요?"
    ),
    Intent.TIME_PRICE: (
        "주차 시간과 요금은 키오스크 화면에서 확인하실 수 있습니다. "
        "어느 부분이 궁금하신가요?"
    ),
    Intent.FACILITY: (
        "기기나 차단기에 이상이 있는 경우 관리실 도움을 받으실 수 있습니다. "
        "현재 어떤 문제가 발생했나요?"
    ),
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
DONE_COOLDOWN_SEC = 1.2


def _normalize(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"[\s\.\,\!\?\u3002\uFF0E\uFF0C\uFF01\uFF1F]+", "", t)
    return t


def _is_done_utterance(text: str) -> bool:
    t = _normalize(text)
    return any(k.replace(" ", "") in t for k in DONE_KEYWORDS)


class AppEngine:
    """
    STT -> Intent-1(주제) 분류(단 1회) -> (정책) one-turn or multiturn -> Dialog LLM

    ✅ 목표 반영:
    - 2차(Dialog)에서는 의도 재분류를 하지 않는다.
    - 첫 턴 Intent-1 결과를 세션이 끝날 때까지 유지한다.
    - 원턴/Intent.NONE 재질문도 dialog_logs에 남기되, session_id NOT NULL 보장.
    """

    def __init__(self):
        self.state = "FIRST_STAGE"

        # 세션 식별자 (dialog_logs NOT NULL)
        self.session_id: str | None = None

        # 첫 턴에서만 결정되는 세션 intent (예: "PAYMENT")
        self.first_intent: str | None = None
        self.intent_log_id: int | None = None

        self.dialog_turn_index = 0
        self.dialog_history: list[dict[str, str]] = []

        self._ignore_until_ts = 0.0

        # one-turn 후 follow-up 오면 자동으로 2nd stage 승격
        self._just_one_turn = False
        self._last_one_turn_intent: Intent | None = None

        # Intent.NONE 재질문 카운트
        self._none_retry_count = 0

    # --------------------------------------------------
    # 세션 보장 (dialog_logs.session_id NOT NULL)
    # --------------------------------------------------
    def _ensure_session(self) -> None:
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
            self.dialog_turn_index = 0
            self.dialog_history = []

    # --------------------------------------------------
    # 안전한 dialog 로깅 래퍼
    # --------------------------------------------------
    def _log_dialog(self, role: str, content: str, model: str = "stt") -> None:
        self._ensure_session()
        self.dialog_turn_index += 1
        try:
            log_dialog(
                intent_log_id=self.intent_log_id,
                session_id=self.session_id,
                role=role,
                content=content,
                model=model,
                turn_index=self.dialog_turn_index,
            )
        except Exception as e:
            # 로깅 실패는 흐름을 멈추지 않게
            print(f"❌ [INTENT_LOGGER] Failed to log dialog: {e}")

        # dialog_llm_client에 넘길 history는 role/user/assistant만 사용
        if role in ("user", "assistant"):
            self.dialog_history.append({"role": role, "content": content})

    # --------------------------------------------------
    # confidence 계산 (기존 유지)
    # --------------------------------------------------
    def calculate_confidence(self, text: str, intent: Intent) -> float:
        score = 0.4
        text = text.strip()

        KEYWORDS_BY_INTENT = {
            Intent.EXIT: ["출차", "나가", "차단기"],
            Intent.ENTRY: ["입차", "들어가", "차단기"],
            Intent.PAYMENT: ["결제", "요금", "카드", "정산"],
            Intent.REGISTRATION: ["등록", "번호판", "방문"],
            Intent.TIME_PRICE: ["시간", "무료", "초과", "요금"],
            Intent.FACILITY: ["기계", "화면", "고장", "이상"],
            Intent.COMPLAINT: ["왜", "안돼", "이상", "짜증"],
        }

        hits = sum(1 for k in KEYWORDS_BY_INTENT.get(intent, []) if k in text)
        score += 0.35 if hits >= 1 else 0.15
        score += 0.05 if len(text) <= 4 else 0.2

        return round(min(score, 1.0), 2)

    # --------------------------------------------------
    # 멀티턴 판단 (기존 유지)
    # --------------------------------------------------
    def should_use_multiturn(self, intent: Intent, confidence: float, text: str) -> bool:
        if intent == Intent.COMPLAINT:
            return True
        if any(k in text for k in ["안돼", "이상", "왜", "멈췄", "실패"]):
            return True
        if confidence < CONFIDENCE_THRESHOLD:
            return True
        return False

    # --------------------------------------------------
    # 2차 멀티턴 처리
    # --------------------------------------------------
    def _handle_second_stage(self, text: str):
        if time.time() < self._ignore_until_ts:
            return

        try:
            if _is_done_utterance(text):
                self._log_dialog("user", text, model="stt")
                self._log_dialog("assistant", FAREWELL_TEXT, model="system")
                print(f"[DIALOG] {FAREWELL_TEXT}")
                self.end_session()
                self._ignore_until_ts = time.time() + DONE_COOLDOWN_SEC
                return

            # ✅ SECOND_STAGE에서는 Intent-1 재분류 금지
            self._log_dialog("user", text, model="stt")

            res = dialog_llm_chat(
                text,
                history=self.dialog_history,
                context={
                    "session_id": self.session_id,
                    "intent_log_id": self.intent_log_id,
                    "first_intent": self.first_intent,  # ✅ 세션 고정 intent
                },
                debug=True,
            )

            reply = getattr(res, "reply", "") or ""
            action = getattr(res, "action", None)

            if action == "DONE":
                reply = FAREWELL_TEXT

            self._log_dialog("assistant", reply, model="llama-3.1-8b")
            print(f"[DIALOG] {reply}")

            if action == "DONE":
                self.end_session()
                self._ignore_until_ts = time.time() + DONE_COOLDOWN_SEC

        except Exception as e:
            print(f"[ENGINE] 2nd-stage failed: {repr(e)}")

    # --------------------------------------------------
    # STT 텍스트 엔트리포인트
    # --------------------------------------------------
    def handle_text(self, text: str):
        if not text or not text.strip():
            return
        if time.time() < self._ignore_until_ts:
            return

        # ✅ 이미 멀티턴 세션이면: 바로 2차로만 처리(의도 재분류 X)
        if self.state == "SECOND_STAGE":
            print("=" * 50)
            print(f"[ENGINE] State={self.state}")
            print(f"[ENGINE] Text={text}")
            self._handle_second_stage(text)
            print("=" * 50)
            return

        # FIRST_STAGE 처리
        print("=" * 50)
        print(f"[ENGINE] State={self.state}")
        print(f"[ENGINE] Text={text}")

        # 세션은 FIRST_STAGE에서도 미리 만들어둔다 (원턴/재질문 로그도 DB에 남기기 위함)
        self._ensure_session()

        # DONE이면 종료
        if _is_done_utterance(text):
            self._log_dialog("user", text, model="stt")
            self._log_dialog("assistant", FAREWELL_TEXT, model="system")
            print(f"[DIALOG] {FAREWELL_TEXT}")
            self.end_session()
            self._ignore_until_ts = time.time() + DONE_COOLDOWN_SEC
            print("=" * 50)
            return

        # 1차 Intent-1 분류 (세션 시작 1회)
        result = detect_intent_llm(text)
        result.confidence = self.calculate_confidence(text, result.intent)

        print(f"[ENGINE] Intent={result.intent.name}, confidence={result.confidence:.2f}")

        # intent 로그 (대화 세션의 anchor)
        self.intent_log_id = log_intent(
            utterance=text,
            predicted_intent=result.intent.value,
            predicted_confidence=result.confidence,
            source="kiosk",
            site_id=SITE_ID,
        )

        # 세션 고정 intent 저장
        self.first_intent = result.intent.value

        # user 발화도 dialog_logs에 남김
        self._log_dialog("user", text, model="stt")

        # ------------------------------
        # Intent.NONE 처리 (재질문 1회)
        # ------------------------------
        if result.intent == Intent.NONE:
            self._none_retry_count += 1

            if self._none_retry_count == 1:
                print("[ENGINE] Intent.NONE → retry question")
                print(f"[ONE-TURN] {NONE_RETRY_TEXT}")

                self._log_dialog("assistant", NONE_RETRY_TEXT, model="system")
                self._just_one_turn = False
                self._last_one_turn_intent = None

                print("=" * 50)
                return

            # 두 번 NONE이면 멀티턴으로 승격
            print("[ENGINE] Intent.NONE twice → escalate to 2nd stage")
            self.state = "SECOND_STAGE"
            self.dialog_turn_index = 0
            self.dialog_history = []
            self._handle_second_stage(text)
            print("=" * 50)
            return

        # NONE가 아니면 리셋
        self._none_retry_count = 0

        # 멀티턴 여부 판단
        if self.should_use_multiturn(result.intent, result.confidence, text):
            print("[ENGINE] Decision: multiturn → 2nd stage")
            self.state = "SECOND_STAGE"
            self.dialog_turn_index = 0
            self.dialog_history = []
            self._handle_second_stage(text)
            print("=" * 50)
            return

        # ------------------------------
        # 원턴 응답
        # ------------------------------
        reply = ONE_TURN_RESPONSES.get(
            result.intent,
            "안내를 도와드릴 수 있는 상담으로 연결하겠습니다."
        )

        print("[ENGINE] Decision: one-turn")
        print(f"[ONE-TURN] {reply}")

        self._log_dialog("assistant", reply, model="system")

        # follow-up 오면 2nd stage로 승격하기 위한 플래그
        self._just_one_turn = True
        self._last_one_turn_intent = result.intent

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

        self._just_one_turn = False
        self._last_one_turn_intent = None
        self._none_retry_count = 0
