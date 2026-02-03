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
DONE_COOLDOWN_SEC = 1.2


def _normalize(text: str) -> str:
    t = text.strip().lower()
    return re.sub(r"[\s\.\,\!\?]+", "", t)


def _is_done_utterance(text: str) -> bool:
    t = _normalize(text)
    return any(k.replace(" ", "") in t for k in DONE_KEYWORDS)


class AppEngine:
    """
    STT → Intent-1(1회) → 정책 판단
    → one-turn or SECOND_STAGE → Dialog LLM

    ✅ 해결 사항
    - one-turn 이후 follow-up 발화 시 SECOND_STAGE 자동 승격
    - SECOND_STAGE 진입 후 FIRST_STAGE로 되돌아가지 않음
    - Intent-1은 세션 시작 시 1회만 수행
    """

    def __init__(self):
        self.state = "FIRST_STAGE"

        self.session_id = None
        self.first_intent = None
        self.intent_log_id = None

        self.dialog_turn_index = 0
        self.dialog_history = []

        self._ignore_until_ts = 0.0

        # 🔑 핵심 상태
        self._just_one_turn = False
        self._none_retry_count = 0

    # --------------------------------------------------
    # 세션 보장
    # --------------------------------------------------
    def _ensure_session(self):
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
            self.dialog_turn_index = 0
            self.dialog_history = []

    # --------------------------------------------------
    # confidence 계산
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
    # 멀티턴 판단
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
    # dialog 로깅
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
    # SECOND_STAGE 처리
    # --------------------------------------------------
    def _handle_second_stage(self, text):
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
    # STT 엔트리포인트 (🔥 핵심 수정 지점)
    # --------------------------------------------------
    def handle_text(self, text):
        if not text or not text.strip():
            return
        if time.time() < self._ignore_until_ts:
            return

        print("=" * 50)
        print(f"[ENGINE] State={self.state}")
        print(f"[ENGINE] Text={text}")

        # ✅ 1️⃣ one-turn 직후 follow-up → 무조건 SECOND_STAGE
        if self._just_one_turn:
            print("[ENGINE] One-turn follow-up → escalate to SECOND_STAGE")
            self._just_one_turn = False
            self.state = "SECOND_STAGE"
            self._handle_second_stage(text)
            print("=" * 50)
            return

        # ✅ 2️⃣ 이미 멀티턴이면 계속 유지
        if self.state == "SECOND_STAGE":
            self._handle_second_stage(text)
            print("=" * 50)
            return

        # --------------------------------------------------
        # FIRST_STAGE
        # --------------------------------------------------
        self._ensure_session()

        if _is_done_utterance(text):
            self._log_dialog("user", text)
            self._log_dialog("assistant", FAREWELL_TEXT, model="system")
            print(f"[DIALOG] {FAREWELL_TEXT}")
            self.end_session()
            print("=" * 50)
            return

        # Intent-1 (세션 1회)
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

        # Intent.NONE 재질문
        if result.intent == Intent.NONE:
            self._none_retry_count += 1
            if self._none_retry_count == 1:
                self._log_dialog("assistant", NONE_RETRY_TEXT, model="system")
                print(f"[ONE-TURN] {NONE_RETRY_TEXT}")
                print("=" * 50)
                return

            print("[ENGINE] Intent.NONE twice → SECOND_STAGE")
            self.state = "SECOND_STAGE"
            self._handle_second_stage(text)
            print("=" * 50)
            return

        self._none_retry_count = 0

        # 멀티턴 판단
        if self.should_use_multiturn(result.intent, result.confidence, text):
            print("[ENGINE] Decision: multiturn → SECOND_STAGE")
            self.state = "SECOND_STAGE"
            self._handle_second_stage(text)
            print("=" * 50)
            return

        # one-turn
        reply = ONE_TURN_RESPONSES.get(result.intent)
        print("[ENGINE] Decision: one-turn")
        print(f"[ONE-TURN] {reply}")
        self._log_dialog("assistant", reply, model="system")

        self._just_one_turn = True
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
        self._none_retry_count = 0
