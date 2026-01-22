from nlu.llm_client import detect_intent_llm
from nlu.intent_schema import Intent

CONFIDENCE_THRESHOLD = 0.75   # 🔑 유일한 정책 값


class AppEngine:
    def handle_text(self, text: str):
        print(f"📥 [ENGINE] STT TEXT: {text}")

        result = detect_intent_llm(text)
        print(f"🧠 [LLM] INTENT: {result.intent} ({result.confidence:.2f})")

        # 1️⃣ NONE은 무시
        if result.intent == Intent.NONE:
            print("🚫 [ENGINE] 명령 아님 → 무시")
            return

        # 2️⃣ confidence 기준만 적용
        if result.confidence < CONFIDENCE_THRESHOLD:
            print(
                f"🚫 [ENGINE] confidence 부족 "
                f"({result.confidence:.2f} < {CONFIDENCE_THRESHOLD:.2f}) → 무시"
            )
            return

        # 3️⃣ 실행
        if result.intent == Intent.OPEN_GATE:
            self.open_gate()
        elif result.intent == Intent.CLOSE_GATE:
            self.close_gate()

    def open_gate(self):
        print("🟢 [CONTROL] 차단기 열기 실행")

    def close_gate(self):
        print("🔴 [CONTROL] 차단기 닫기 실행")