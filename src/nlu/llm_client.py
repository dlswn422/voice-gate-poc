from nlu.intent_schema import Intent, IntentResult


class LLMClient:
    """
    LLM 기반 NLU 자리
    지금은 Mock, 나중에 실제 LLM으로 교체
    """

    def analyze(self, text: str) -> IntentResult:
        text = text.lower()

        if "차단기" in text and ("열어" in text or "열어주세요" in text):
            return {
                "intent": Intent.OPEN_GATE,
                "confidence": 0.9,
                "requires_human": False,
            }

        if "관리자" in text or "사람" in text:
            return {
                "intent": Intent.CALL_ADMIN,
                "confidence": 0.8,
                "requires_human": True,
            }

        return {
            "intent": Intent.UNKNOWN,
            "confidence": 0.4,
            "requires_human": True,
        }
