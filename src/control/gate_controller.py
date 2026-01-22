from nlu.intent_schema import Intent


def execute_gate_action(intent: Intent):
    if intent == Intent.OPEN_GATE:
        print("🟢 [CONTROL] 차단기 열기 실행")

    elif intent == Intent.CLOSE_GATE:
        print("🔴 [CONTROL] 차단기 닫기 실행")
