from nlu.llm_client import detect_intent_llm
from nlu.intent_schema import Intent
from engine.intent_logger import log_intent, log_dialog
from nlu.dialog_llm_client import dialog_llm_chat

import uuid
import time
import re


# --------------------------------------------------
# 정책 설정
# --------------------------------------------------
CONFIDENCE_THRESHOLD = 0.75
SITE_ID = "parkassist_local"

# ==================================================
# 멀티턴 정책 (핵심)
# ==================================================
# 구조적으로 "도움 요청" → 무조건 멀티턴
MULTI_TURN_INTENTS = {
    Intent.ENTRY_FLOW_ISSUE,
    Intent.EXIT_FLOW_ISSUE,
    Intent.PAYMENT_ISSUE,
    Intent.REGISTRATION_ISSUE,
    Intent.COMPLAINT,
}

# 구조적으로 "정보 요청" → 1턴 가능
ONE_TURN_INTENTS = {
    Intent.PRICE_INQUIRY,
    Intent.HOW_TO_EXIT,
    Intent.HOW_TO_REGISTER,
    Intent.TIME_ISSUE,
}

# --------------------------------------------------
# DONE 강제 종료 정책
# --------------------------------------------------
DONE_KEYWORDS = [
    "됐어요", "되었습니다", "해결", "괜찮아요",
    "그만", "종료", "끝", "마칠게",
    "고마워", "감사", "안녕"
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
    주차장 키오스크 CX용 App Engine

    상태:
    - FIRST_STAGE  : 1차 의도 분류
    - SECOND_STAGE : 2차 멀티턴 상담
    """

    def __init__(self):
        self.state = "FIRST_STAGE"
        self.session_id = None

        # 2차 컨텍스트용
        self.first_intent = None
        self.intent_log_id = None

        # 멀티턴 상태
        self.dialog_turn_index = 0
        self.dialog_history = []

        # DONE 쿨다운
        self._ignore_until_ts = 0.0

    # ==================================================
    # confidence 계산 (AppEngine 책임)
    # ==================================================
    def calculate_confidence(self, text: str, intent: Intent) -> float:
        score = 0.4
        text = text.strip()

        intent_name = getattr(intent, "name", str(intent))

        KEYWORDS_BY_INTENT_NAME = {
            "EXIT_FLOW_ISSUE": ["출차", "나가", "차단기", "안열려", "안 열려"],
            "ENTRY_FLOW_ISSUE": ["입차", "들어가", "차단기", "안열려", "안 열려"],
            "PAYMENT_ISSUE": ["결제", "요금", "카드", "정산", "승인"],
            "TIME_ISSUE": ["시간", "무료", "초과"],
            "PRICE_INQUIRY": ["얼마", "요금", "가격"],
            "HOW_TO_EXIT": ["어떻게", "출차", "나가"],
            "HOW_TO_REGISTER": ["등록", "어디", "방법"],
        }

        hits = sum(1 for k in KEYWORDS_BY_INTENT_NAME.get(intent_name, []) if k in text)

        score += 0.35 if hits >= 1 else 0.15

        if len(text) < 3:
            score += 0.05
        elif any(f in text for f in ["어", "음", "..."]):
            score += 0.10
        else:
            score += 0.25

        INTENT_RISK_WEIGHT_BY_NAME = {
            "HOW_TO_EXIT": 1.0,
            "PRICE_INQUIRY": 1.0,
            "TIME_ISSUE": 0.95,
            "EXIT_FLOW_ISSUE": 0.85,
            "ENTRY_FLOW_ISSUE": 0.85,
            "PAYMENT_ISSUE": 0.85,
            "REGISTRATION_ISSUE": 0.80,
            "COMPLAINT": 0.70,
        }

        score *= INTENT_RISK_WEIGHT_BY_NAME.get(intent_name, 0.6)
        return round(min(score, 1.0), 2)

    # ==================================================
    # 멀티턴 여부 판단 (정책 핵심)
    # ==================================================
    def should_use_multiturn(self, intent: Intent, confidence: float) -> bool:
        # 구조적으로 도움 요청 → 무조건 멀티턴
        if intent in MULTI_TURN_INTENTS:
            return True

        # 정보성이라도 confidence 낮으면 확인 필요
        if intent in ONE_TURN_INTENTS and confidence < CONFIDENCE_THRESHOLD:
            return True

        return False

    # ==================================================
    # 2차 멀티턴 처리
    # ==================================================
    def _handle_second_stage(self, text: str):
        if time.time() < self._ignore_until_ts:
            return

        try:
            # DONE 강제 종료
            if _is_done_utterance(text):
                self.dialog_turn_index += 1
                log_dialog(
                    intent_log_id=self.intent_log_id,
                    session_id=self.session_id,
                    role="user",
                    content=text,
                    model="stt",
                    turn_index=self.dialog_turn_index,
                )

                self.dialog_turn_index += 1
                log_dialog(
                    intent_log_id=self.intent_log_id,
                    session_id=self.session_id,
                    role="assistant",
                    content=FAREWELL_TEXT,
                    model="system",
                    turn_index=self.dialog_turn_index,
                )

                print(f"[DIALOG] {FAREWELL_TEXT}")
                self.end_second_stage()
                self._ignore_until_ts = time.time() + DONE_COOLDOWN_SEC
                return

            # user 로그
            self.dialog_turn_index += 1
            log_dialog(
                intent_log_id=self.intent_log_id,
                session_id=self.session_id,
                role="user",
                content=text,
                model="stt",
                turn_index=self.dialog_turn_index,
            )

            self.dialog_history.append({"role": "user", "content": text})

            # 2차 LLM 호출
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
            action = getattr(res, "action", None)

            if action == "DONE":
                reply = FAREWELL_TEXT

            # assistant 로그
            self.dialog_turn_index += 1
            log_dialog(
                intent_log_id=self.intent_log_id,
                session_id=self.session_id,
                role="assistant",
                content=reply,
                model="llama-3.1-8b",
                turn_index=self.dialog_turn_index,
            )

            self.dialog_history.append({"role": "assistant", "content": reply})
            print(f"[DIALOG] {reply}")

            if action == "DONE":
                self.end_second_stage()
                self._ignore_until_ts = time.time() + DONE_COOLDOWN_SEC

        except Exception as e:
            print(f"[ENGINE] 2nd-stage failed: {repr(e)}")

    # ==================================================
    # STT 텍스트 엔트리포인트
    # ==================================================
    def handle_text(self, text: str):
        if not text or not text.strip():
            return

        if time.time() < self._ignore_until_ts:
            return

        print("=" * 50)
        print(f"[ENGINE] State={self.state}")
        print(f"[ENGINE] Text={text}")

        # ------------------------------
        # 2차 멀티턴
        # ------------------------------
        if self.state == "SECOND_STAGE":
            self._handle_second_stage(text)
            print("=" * 50)
            return

        # ------------------------------
        # 1차 의도 분류
        # ------------------------------
        result = detect_intent_llm(text)
        self.first_intent = result.intent.value

        result.confidence = self.calculate_confidence(text, result.intent)

        print(f"[ENGINE] Intent={result.intent.name}, confidence={result.confidence:.2f}")

        self.intent_log_id = log_intent(
            utterance=text,
            predicted_intent=result.intent.value,
            predicted_confidence=result.confidence,
            source="kiosk",
            site_id=SITE_ID,
        )

        if self.intent_log_id is None or result.intent == Intent.NONE:
            print("=" * 50)
            return

        # ------------------------------
        # 멀티턴 여부 판단
        # ------------------------------
        if self.should_use_multiturn(result.intent, result.confidence):
            print("[ENGINE] Decision: multiturn → 2nd stage")

            self.state = "SECOND_STAGE"
            self.session_id = str(uuid.uuid4())
            self.dialog_turn_index = 0
            self.dialog_history = []

            self._handle_second_stage(text)
            print("=" * 50)
            return

        # ------------------------------
        # 1턴 종료 (정보성)
        # ------------------------------
        print("[ENGINE] Decision: one-turn handled by 1st stage")
        print("=" * 50)

    # ==================================================
    # 상담 종료
    # ==================================================
    def end_second_stage(self):
        print(f"[ENGINE] Session ended: {self.session_id}")
        self.state = "FIRST_STAGE"
        self.session_id = None
        self.intent_log_id = None
        self.dialog_turn_index = 0
        self.dialog_history = []
        self.first_intent = None
