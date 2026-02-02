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
    Intent.EXIT: "정산이 완료되면 차단기가 자동으로 열립니다.",
    Intent.ENTRY: "입차 시 차량 인식 후 차단기가 자동으로 열립니다.",
    Intent.PAYMENT: "주차 요금은 정산기 또는 출구에서 확인하실 수 있습니다.",
    Intent.REGISTRATION: "차량 또는 방문자 등록은 키오스크에서 진행하실 수 있습니다.",
    Intent.TIME_PRICE: "주차 시간 및 요금은 키오스크 화면에서 확인하실 수 있습니다.",
    Intent.FACILITY: "기기 이상 시 관리실로 문의해 주세요.",
}

# ==================================================
# DONE 강제 종료 정책
# ==================================================
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
    주차장 키오스크 CX App Engine (FINAL)

    - 1차 모델: 의도 분류 전용
    - 원턴 가능 → 즉시 시스템 응답
    - 원턴 후 추가 발화 → 자동 멀티턴 승격
    - 멀티턴 → 2차 LLM 상담
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

        # ✅ 원턴 후 승격용 상태
        self._just_one_turn = False
        self._last_one_turn_intent = None

    # ==================================================
    # confidence 계산
    # ==================================================
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

    # ==================================================
    # 멀티턴 여부 판단
    # ==================================================
    def should_use_multiturn(self, intent: Intent, confidence: float, text: str) -> bool:
        if intent == Intent.COMPLAINT:
            return True
        if any(k in text for k in ["안돼", "이상", "왜", "멈췄"]):
            return True
        if confidence < CONFIDENCE_THRESHOLD:
            return True
        return False

    # ==================================================
    # 2차 멀티턴 처리
    # ==================================================
    def _handle_second_stage(self, text: str):
        if time.time() < self._ignore_until_ts:
            return

        try:
            if _is_done_utterance(text):
                self._log_dialog("user", text)
                self._log_dialog("assistant", FAREWELL_TEXT, model="system")
                print(f"[DIALOG] {FAREWELL_TEXT}")
                self.end_second_stage()
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
            action = getattr(res, "action", None)

            if action == "DONE":
                reply = FAREWELL_TEXT

            self._log_dialog("assistant", reply, model="llama-3.1-8b")
            print(f"[DIALOG] {reply}")

            if action == "DONE":
                self.end_second_stage()
                self._ignore_until_ts = time.time() + DONE_COOLDOWN_SEC

        except Exception as e:
            print(f"[ENGINE] 2nd-stage failed: {repr(e)}")

    def _log_dialog(self, role: str, content: str, model: str = "stt"):
        self.dialog_turn_index += 1
        log_dialog(
            intent_log_id=self.intent_log_id,
            session_id=self.session_id,
            role=role,
            content=content,
            model=model,
            turn_index=self.dialog_turn_index,
        )
        self.dialog_history.append({"role": role, "content": content})

    # ==================================================
    # STT 텍스트 엔트리포인트
    # ==================================================
    def handle_text(self, text: str):
        if not text or not text.strip():
            return
        if time.time() < self._ignore_until_ts:
            return

        # --------------------------------------------------
        # ✅ 원턴 직후 추가 발화 → 자동 멀티턴 승격
        # --------------------------------------------------
        if self._just_one_turn:
            self._just_one_turn = False
            print("[ENGINE] One-turn follow-up detected → escalate to 2nd stage")

            self.state = "SECOND_STAGE"
            self.session_id = str(uuid.uuid4())
            self.dialog_turn_index = 0
            self.dialog_history = []
            self.first_intent = (
                self._last_one_turn_intent.value
                if self._last_one_turn_intent else None
            )

            self._handle_second_stage(text)
            print("=" * 50)
            return

        print("=" * 50)
        print(f"[ENGINE] State={self.state}")
        print(f"[ENGINE] Text={text}")

        # ------------------------------
        # 2차 멀티턴 중
        # ------------------------------
        if self.state == "SECOND_STAGE":
            self._handle_second_stage(text)
            print("=" * 50)
            return

        # ------------------------------
        # 1차 의도 분류
        # ------------------------------
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

        if result.intent == Intent.NONE:
            print("=" * 50)
            return

        self.first_intent = result.intent.value

        # ------------------------------
        # 멀티턴 여부 판단
        # ------------------------------
        if self.should_use_multiturn(result.intent, result.confidence, text):
            print("[ENGINE] Decision: multiturn → 2nd stage")

            self.state = "SECOND_STAGE"
            self.session_id = str(uuid.uuid4())
            self.dialog_turn_index = 0
            self.dialog_history = []

            self._handle_second_stage(text)
            print("=" * 50)
            return

        # ------------------------------
        # ✅ 원턴 즉시 응답
        # ------------------------------
        reply = ONE_TURN_RESPONSES.get(
            result.intent,
            "안내를 도와드릴 수 있는 상담으로 연결하겠습니다."
        )

        print("[ENGINE] Decision: one-turn → immediate response")
        print(f"[ONE-TURN] {reply}")

        log_dialog(
            intent_log_id=self.intent_log_id,
            session_id=None,
            role="assistant",
            content=reply,
            model="system",
            turn_index=0,
        )

        # ✅ 원턴 후 상태 기록
        self._just_one_turn = True
        self._last_one_turn_intent = result.intent

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
        self._just_one_turn = False
        self._last_one_turn_intent = None
