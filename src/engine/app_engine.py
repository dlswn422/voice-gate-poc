from nlu.llm_client import detect_intent_llm
from nlu.intent_schema import Intent
from nlu.dialog_llm_client import (
    dialog_llm_chat,
    make_dialog_system_messages,
    DialogLLMResponse,
)

CONFIDENCE_THRESHOLD = 0.75   # 🔑 유일한 정책 값


def _is_yes(text: str) -> bool:
    t = text.strip().lower()
    return any(k in t for k in ["예", "네", "응", "맞", "그래", "y", "yes"])


def _is_no(text: str) -> bool:
    t = text.strip().lower()
    return any(k in t for k in ["아니", "아뇨", "no", "n", "ㄴㄴ", "싫"])


class AppEngine:
    def __init__(self):
        # IDLE: 1차 의도 분류
        # DIALOG: 2차 대화형 LLM (Llama)로 추가 질문/해결
        # CONFIRM: 실제 제어 실행 전 사용자 확인 단계
        self.state = "IDLE"
        self.dialog_messages = []
        self.pending_intent: Intent | None = None

    def handle_text(self, text: str):
        print("\n" + "=" * 50)
        print("📥 [ENGINE] 음성 명령 수신")
        print(f"🗣  STT TEXT        : \"{text}\"")

        # =========================
        # CONFIRM 단계
        # =========================
        if self.state == "CONFIRM":
            if _is_yes(text) and self.pending_intent is not None:
                print("✅ [CONFIRM] 사용자 확인: 예 → 실행")
                self._execute_intent(self.pending_intent)
                self._reset_dialog()
            elif _is_no(text):
                print("🚫 [CONFIRM] 사용자 확인: 아니오 → 취소")
                self._reset_dialog()
            else:
                print("❓ [CONFIRM] 예/아니오로만 답해주세요.")
            print("=" * 50)
            return

        # =========================
        # DIALOG 단계 (2차 LLM)
        # =========================
        if self.state == "DIALOG":
            self._dialog_step(user_text=text)
            print("=" * 50)
            return

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
            # ➜ 2차 대화형 모델로 에스컬레이션
            self._start_dialog(
                original_text=text,
                intent_hint=result.intent.name,
                confidence=result.confidence,
                reason="LOW_CONFIDENCE",
            )
            print("=" * 50)
            return

        # 2.5️⃣ HELP/INFO는 "명령"이 아니라 상담 영역 → 2차로 전달
        if result.intent in (Intent.HELP_REQUEST, Intent.INFO_REQUEST):
            print("ℹ️ [DECISION] 도움/안내 요청 → 2차 대화형 모델로 전환")
            self._start_dialog(
                original_text=text,
                intent_hint=result.intent.name,
                confidence=result.confidence,
                reason="HELP_OR_INFO",
            )
            print("=" * 50)
            return

        # 3️⃣ 최종 실행 판단
        print("✅ [DECISION] 제어 조건 충족 → 실행")

        self._execute_intent(result.intent)

        print("=" * 50)

    def _execute_intent(self, intent: Intent):
        if intent == Intent.OPEN_GATE:
            self.open_gate()
        elif intent == Intent.CLOSE_GATE:
            self.close_gate()

    def _reset_dialog(self):
        self.state = "IDLE"
        self.dialog_messages = []
        self.pending_intent = None

    def _start_dialog(
        self,
        original_text: str,
        intent_hint: str,
        confidence: float,
        reason: str,
    ):
        """2차 LLM 대화 시작."""
        self.state = "DIALOG"
        self.dialog_messages = make_dialog_system_messages()

        # 1차 결과를 힌트로 전달(모델이 상황을 빨리 잡게)
        self.dialog_messages.append(
            {
                "role": "user",
                "content": (
                    f"[1차 의도 힌트] intent={intent_hint}, confidence={confidence:.2f}, reason={reason}\n"
                    f"[사용자 원문] {original_text}"
                ),
            }
        )

        # 시작하자마자 2차 응답 1회 생성
        self._dialog_step(user_text=None)

    def _dialog_step(self, user_text: str | None):
        """2차 대화형 LLM 1 step."""
        if user_text:
            self.dialog_messages.append({"role": "user", "content": user_text})

        try:
            resp: DialogLLMResponse = dialog_llm_chat(self.dialog_messages)
        except Exception as e:
            print("❌ [DIALOG] Llama 호출 실패:", e)
            print("➡️ Ollama가 실행 중인지 확인: `ollama serve` / `ollama run llama3.1:8b`")
            self._reset_dialog()
            return

        # 사용자에게 보여줄 문장
        print("🤖 [Llama]", resp.assistant)
        self.dialog_messages.append({"role": "assistant", "content": resp.assistant})

        # 모델이 제어를 제안하면 확인 단계로
        if resp.suggested_intent in ("OPEN_GATE", "CLOSE_GATE") and resp.confirm:
            self.pending_intent = Intent.OPEN_GATE if resp.suggested_intent == "OPEN_GATE" else Intent.CLOSE_GATE
            self.state = "CONFIRM"
            prompt = resp.confirm_prompt or "실행할까요? 예/아니오로 답해주세요."
            print("🧩 [CONFIRM PROMPT]", prompt)
            return

        # 종료 조건
        if resp.state == "END":
            self._reset_dialog()

    def open_gate(self):
        print("🟢 [CONTROL] 차단기 열기 실행")

    def close_gate(self):
        print("🔴 [CONTROL] 차단기 닫기 실행")
