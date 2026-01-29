from nlu.llm_client import detect_intent_llm
from nlu.intent_schema import Intent
from engine.intent_logger import log_intent

import uuid
import time


# --------------------------------------------------
# 정책 설정
# --------------------------------------------------

CONFIDENCE_THRESHOLD = 0.75
SITE_ID = "parkassist_local"


class AppEngine:
    """
    주차장 키오스크 CX용 App Engine

    상태:
    - FIRST_STAGE  : 1차 의도 분류 단계
    - SECOND_STAGE : 2차 상담(라마) 단계
    """

    def __init__(self):
        # 🔑 상태 관리
        self.state = "FIRST_STAGE"
        self.session_id = None

    # ==================================================
    # 🔧 confidence 계산 로직
    # ==================================================

    def calculate_confidence(self, text: str, intent: Intent) -> float:
        score = 0.0
        text = text.strip()

        KEYWORDS = {
            Intent.EXIT_FLOW_ISSUE: ["출차", "나가", "차단기", "안 열려"],
            Intent.ENTRY_FLOW_ISSUE: ["입차", "들어가", "차단기", "안 열려"],
            Intent.PAYMENT_ISSUE: ["결제", "요금", "카드", "정산"],
            Intent.TIME_ISSUE: ["시간", "무료", "초과"],
            Intent.PRICE_INQUIRY: ["얼마", "요금", "가격"],
            Intent.HOW_TO_EXIT: ["어떻게", "출차", "나가"],
            Intent.HOW_TO_REGISTER: ["등록", "어디", "방법"],
        }

        hits = sum(
            1 for k in KEYWORDS.get(intent, [])
            if k in text
        )

        if hits >= 2:
            score += 0.45
        elif hits == 1:
            score += 0.30
        else:
            score += 0.10

        if len(text) < 3:
            score += 0.05
        elif any(f in text for f in ["어", "음", "..."]):
            score += 0.10
        else:
            score += 0.25

        INTENT_RISK_WEIGHT = {
            Intent.HOW_TO_EXIT: 1.0,
            Intent.PRICE_INQUIRY: 1.0,
            Intent.TIME_ISSUE: 0.9,
            Intent.EXIT_FLOW_ISSUE: 0.7,
            Intent.ENTRY_FLOW_ISSUE: 0.7,
            Intent.PAYMENT_ISSUE: 0.7,
            Intent.REGISTRATION_ISSUE: 0.6,
            Intent.COMPLAINT: 0.5,
        }

        score *= INTENT_RISK_WEIGHT.get(intent, 0.6)
        return round(min(score, 1.0), 2)

    # ==================================================
    # 🎙️ STT 텍스트 처리 엔트리포인트
    # ==================================================

    def handle_text(self, text: str):
        if not text or not text.strip():
            return

        print("=" * 50)
        print(f"[ENGINE] State={self.state}")
        print(f"[ENGINE] Text={text}")

        # ==================================================
        # 🟢 2차 상담 단계
        # ==================================================
        if self.state == "SECOND_STAGE":
            print("[ENGINE] 2nd-stage dialog input")
            print("[ENGINE] → dialog_logs로만 저장 (intent ❌)")
            print("[ENGINE] (여기서 라마 호출)")
            print("=" * 50)
            return

        # ==================================================
        # 🔵 1차 의도 분류 단계
        # ==================================================
        try:
            result = detect_intent_llm(text)
        except Exception as e:
            print("[ENGINE] LLM inference failed:", e)
            print("=" * 50)
            return

        result.confidence = self.calculate_confidence(
            text=text,
            intent=result.intent,
        )

        print(
            f"[ENGINE] Intent={result.intent.name}, "
            f"confidence={result.confidence:.2f}"
        )

        # 1차 판단 로그는 무조건 적재 (딱 1번)
        log_intent(
            utterance=text,
            predicted_intent=result.intent.value,
            predicted_confidence=result.confidence,
            source="kiosk",
            site_id=SITE_ID,
        )

        if result.intent == Intent.NONE:
            print("[ENGINE] Decision: irrelevant utterance")
            print("=" * 50)
            return

        # ==================================================
        # confidence 기준 분기
        # ==================================================
        if result.confidence < CONFIDENCE_THRESHOLD:
            print("[ENGINE] Decision: low confidence → 2nd stage")

            # 🔑 상태 전환
            self.state = "SECOND_STAGE"
            self.session_id = str(uuid.uuid4())

            print(f"[ENGINE] Session started: {self.session_id}")
            print("[ENGINE] Next input goes to dialog_logs")
            print("=" * 50)
            return

        print("[ENGINE] Decision: passed 1st-stage classification")
        print("[ENGINE] Action: defer execution to next stage")
        print("=" * 50)

    # ==================================================
    # 🔚 상담 종료 시 호출
    # ==================================================

    def end_second_stage(self):
        print(f"[ENGINE] Session ended: {self.session_id}")
        self.state = "FIRST_STAGE"
        self.session_id = None
