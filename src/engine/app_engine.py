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
    2. LLM 기반 1차 의도 분류
    3. 분류 결과를 학습 데이터로 DB 적재
    4. 신뢰도 기준으로 다음 단계 위임 여부 판단

    주의:
    - 이 단계에서는 실제 제어를 수행하지 않는다
    - 실행 여부는 반드시 다음 단계에서 결정한다
    """

    def handle_text(self, text: str):
        # ==================================================
        # 0️⃣ 기본 방어 로직
        # ==================================================
        if not text or not text.strip():
            return

        request_id = str(uuid.uuid4())
        received_at = time.time()  # 추후 확장용 (현재는 로그만)

        print("=" * 50)
        print("[ENGINE] Speech received")
        print(f"[ENGINE] Text: {text}")
        print(f"[ENGINE] Request ID: {request_id}")

        # ==================================================
        # 1️⃣ 1차 의도 분류 (LLM)
        # ==================================================
        try:
            result = detect_intent_llm(text)
        except Exception as e:
            print("[ENGINE] LLM inference failed:", e)
            print("=" * 50)
            return

        print(
            f"[ENGINE] Intent={result.intent.name}, "
            f"confidence={result.confidence:.2f}"
        )

        # ==================================================
        # 2️⃣ 학습 데이터 DB 적재
        # ==================================================
        # 이 데이터는 '원천 학습 데이터'로 사용되며
        # 추후 사람 검수를 통해 최종 라벨로 확정된다
        log_intent(
            utterance=text,
            predicted_intent=result.intent.value,
            predicted_confidence=result.confidence,
            source="kiosk",
            site_id=SITE_ID,
        )

        # ==================================================
        # 3️⃣ 주차장 CX와 무관한 발화
        # ==================================================
        if result.intent == Intent.NONE:
            print("[ENGINE] Decision: irrelevant utterance")
            print("=" * 50)
            return

        # ==================================================
        # 4️⃣ 신뢰도 기준 판단
        # ==================================================
        if result.confidence < CONFIDENCE_THRESHOLD:
            print(
                "[ENGINE] Decision: low confidence "
                f"({result.confidence:.2f} < {CONFIDENCE_THRESHOLD:.2f})"
            )
            print("[ENGINE] Action: hand off to next stage")
            print("=" * 50)
            return

        # ==================================================
        # 5️⃣ 1차 분류 통과 (실행 아님)
        # ==================================================
        print("[ENGINE] Decision: passed 1st-stage classification")
        print("[ENGINE] Action: defer execution to next stage")
        print("=" * 50)