from nlu.intent_schema import Intent, IntentResult


class RuleEngine:
    def decide(self, result: IntentResult) -> str:
        if result["intent"] == Intent.OPEN_GATE and result["confidence"] >= 0.8:
            return "OPEN_GATE"

        if result["requires_human"]:
            return "CALL_ADMIN"

        return "REJECT"
