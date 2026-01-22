from nlu.llm_client import detect_intent_llm
from nlu.intent_schema import Intent

CONFIDENCE_THRESHOLD = 0.75   # 🔑 유일한 정책 값


class AppEngine:
    def handle_text(self, text: str):
        print("\n" + "=" * 50)
        print("📥 [ENGINE] 음성 명령 수신")
        print(f"🗣  STT TEXT        : \"{text}\"")

        # LLM 추론
        result = detect_intent_llm(text)
        print(
            f"🧠 [LLM] 의도 추론     : {result.intent.name}"
            f" (confidence={result.confidence:.2f})"
        )

        # 1️⃣ 명령 여부 판단
        if result.intent == Intent.NONE:
            print("🚫 [DECISION] 차단기 제어와 무관 → 실행 안 함")
            print("=" * 50)
            return

        # 2️⃣ 신뢰도 기준 적용
        if result.confidence < CONFIDENCE_THRESHOLD:
            print(
                "🚫 [DECISION] 신뢰도 기준 미달\n"
                f"    └ confidence {result.confidence:.2f} "
                f"< threshold {CONFIDENCE_THRESHOLD:.2f}"
            )
            print("=" * 50)
            return

        # 3️⃣ 최종 실행 판단
        print("✅ [DECISION] 제어 조건 충족 → 실행")

        if result.intent == Intent.OPEN_GATE:
            self.open_gate()
        elif result.intent == Intent.CLOSE_GATE:
            self.close_gate()

        print("=" * 50)

    def open_gate(self):
        print("🟢 [CONTROL] 차단기 열기 실행")

    def close_gate(self):
        print("🔴 [CONTROL] 차단기 닫기 실행")
