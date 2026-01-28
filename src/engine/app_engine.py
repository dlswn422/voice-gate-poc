from nlu.llm_client import detect_intent_llm
from nlu.intent_schema import Intent
from engine.intent_logger import log_intent   # 학습 데이터 적재용 모듈
import uuid
import time

CONFIDENCE_THRESHOLD = 0.75   # 1차 의도 분류 신뢰도 기준
SITE_ID = "parkassist_local"  # 현장/주차장 식별자 (환경별로 변경 가능)


class AppEngine:
    """
    주차장 키오스크 CX용 App Engine (1차 단계)

    역할 요약:
    1. STT 텍스트 수신
    2. 1차 의도 분류 (LLM 기반)
    3. 의도 분류 결과를 학습 데이터로 DB 적재
    4. 신뢰도 기준으로 '보류 / 다음 단계 위임' 판단

    ⚠️ 주의:
    - 이 단계에서는 차단기 제어를 직접 수행하지 않는다
    - OPEN_GATE / CLOSE_GATE 같은 즉시 제어 개념은 존재하지 않는다
    """

    def handle_text(self, text: str):
        # ==============================================
        # 0️⃣ 기본 방어 로직
        # ==============================================
        if not text or not text.strip():
            return

        request_id = str(uuid.uuid4())
        received_at = time.time()

        print("\n" + "=" * 50)
        print("📥 [ENGINE] 음성 발화 수신")
        print(f"🗣  STT TEXT        : \"{text}\"")
        print(f"🆔 request_id      : {request_id}")

        # ==============================================
        # 1️⃣ 1차 의도 분류 (LLM)
        # ==============================================
        try:
            result = detect_intent_llm(text)
        except Exception as e:
            print("❌ [ENGINE] LLM 추론 실패:", e)
            print("=" * 50)
            return

        print(
            f"🧠 [INTENT] {result.intent.name} "
            f"(confidence={result.confidence:.2f})"
        )

        # ==============================================
        # 2️⃣ 학습 데이터 DB 적재
        # ==============================================
        # 이 시점의 데이터가 '원천 학습 데이터'
        # (LLM 예측값 + confidence → 추후 사람 검수)
        log_intent(
            utterance=text,
            predicted_intent=result.intent.value,
            predicted_confidence=result.confidence,
            source="kiosk",
            site_id=SITE_ID,
        )

        # ==============================================
        # 3️⃣ 주차장 CX와 무관한 발화
        # ==============================================
        if result.intent == Intent.NONE:
            print("🚫 [DECISION] 주차장 CX 무관 발화 → 종료")
            print("=" * 50)
            return

        # ==============================================
        # 4️⃣ 신뢰도 기준 판단
        # ==============================================
        if result.confidence < CONFIDENCE_THRESHOLD:
            print(
                "🟡 [DECISION] 의도는 있으나 신뢰도 낮음\n"
                f"    └ confidence {result.confidence:.2f} "
                f"< threshold {CONFIDENCE_THRESHOLD:.2f}\n"
                "    └ 2차 모델 또는 추가 UX 단계로 위임 대상"
            )
            print("=" * 50)
            return

        # ==============================================
        # 5️⃣ 1차 분류 통과 (실행 아님)
        # ==============================================
        print(
            "🟢 [DECISION] 1차 의도 분류 통과\n"
            "    └ 실제 제어 / 안내 / 응답은 다음 단계에서 결정"
        )
        print("=" * 50)
