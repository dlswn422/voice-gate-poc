from nlu.llm_client import detect_intent_llm
from nlu.intent_schema import Intent
from engine.intent_logger import log_intent

import uuid
import time


# --------------------------------------------------
# 정책 설정
# --------------------------------------------------

CONFIDENCE_THRESHOLD = 0.75   # 1차 의도 분류 신뢰도 기준
SITE_ID = "parkassist_local"  # 주차장 / 현장 식별자


class AppEngine:
    """
    주차장 키오스크 CX용 App Engine (1차 단계)

    역할:
    1. STT로부터 텍스트 수신
    2. LLM 기반 1차 의도 분류 (intent ONLY)
    3. confidence 시스템 계산
    4. 분류 결과를 DB에 적재
    5. 신뢰도 기준으로 다음 단계 위임 여부 판단

    주의:
    - 이 단계에서는 실제 제어를 수행하지 않는다
    - confidence는 '1차 판단을 바로 써도 되는지'에 대한 점수다
    """

    # ==================================================
    # 🔧 confidence 계산 로직 (핵심)
    # ==================================================

    def calculate_confidence(self, text: str, intent: Intent) -> float:
        """
        DB 구조를 변경하지 않고
        confidence를 연속값으로 계산한다 (0.0 ~ 1.0)

        의미:
        - 이 1차 판단을 지금 써도 될지에 대한 시스템 점수
        """

        score = 0.0
        text = text.strip()

        # ------------------------------
        # 1️⃣ 의도별 키워드 명확성
        # ------------------------------
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

        # ------------------------------
        # 2️⃣ 텍스트 품질 (STT 신뢰도 보정)
        # ------------------------------
        if len(text) < 3:
            score += 0.05
        elif any(f in text for f in ["어", "음", "..."]):
            score += 0.10
        else:
            score += 0.25

        # ------------------------------
        # 3️⃣ intent 위험도 보정
        # ------------------------------
        # 바로 실행하기 위험한 intent일수록 보수적으로
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
        # ==================================================
        # 0️⃣ 기본 방어 로직
        # ==================================================
        if not text or not text.strip():
            return

        request_id = str(uuid.uuid4())
        received_at = time.time()

        print("=" * 50)
        print("[ENGINE] Speech received")
        print(f"[ENGINE] Text: {text}")
        print(f"[ENGINE] Request ID: {request_id}")

        # ==================================================
        # 1️⃣ 1차 의도 분류 (LLM: intent ONLY)
        # ==================================================
        try:
            result = detect_intent_llm(text)
        except Exception as e:
            print("[ENGINE] LLM inference failed:", e)
            print("=" * 50)
            return

        # ==================================================
        # 2️⃣ confidence 시스템 계산 (덮어쓰기)
        # ==================================================
        result.confidence = self.calculate_confidence(
            text=text,
            intent=result.intent,
        )

        print(
            f"[ENGINE] Intent={result.intent.name}, "
            f"confidence={result.confidence:.2f}"
        )

        # ==================================================
        # 3️⃣ DB 적재 (기존 구조 그대로)
        # ==================================================
        log_intent(
            utterance=text,
            predicted_intent=result.intent.value,
            predicted_confidence=result.confidence,
            source="kiosk",
            site_id=SITE_ID,
        )

        # ==================================================
        # 4️⃣ 주차장 CX와 무관한 발화
        # ==================================================
        if result.intent == Intent.NONE:
            print("[ENGINE] Decision: irrelevant utterance")
            print("=" * 50)
            return

        # ==================================================
        # 5️⃣ confidence 기준 분기
        # ==================================================
        if result.confidence < CONFIDENCE_THRESHOLD:
            print(
                "[ENGINE] Decision: low confidence "
                f"({result.confidence:.2f} < {CONFIDENCE_THRESHOLD:.2f})"
            )
            print("[ENGINE] Action: hand off to next stage (2nd LLM)")
            print("=" * 50)
            return

        # ==================================================
        # 6️⃣ 1차 분류 통과 (실행은 다음 단계)
        # ==================================================
        print("[ENGINE] Decision: passed 1st-stage classification")
        print("[ENGINE] Action: defer execution to next stage")
        print("=" * 50)