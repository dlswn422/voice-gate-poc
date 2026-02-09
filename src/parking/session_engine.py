from enum import Enum

class SessionState(str, Enum):
    IDLE = "IDLE"
    ENTERED = "ENTERED"
    EXIT_DETECTED = "EXIT_DETECTED"
    PAYMENT_REQUIRED = "PAYMENT_REQUIRED"
    PAYMENT_IN_PROGRESS = "PAYMENT_IN_PROGRESS"
    PAID = "PAID"
    ASSISTING = "ASSISTING"
    EXITED = "EXITED"


class ParkingSessionEngine:
    def __init__(self):
        self.state = SessionState.IDLE

    def handle_event(self, event: dict) -> dict:
        """
        event = { type: "...", ... }
        return = { state, action?, message?, lock_voice? }
        """

        etype = event["type"]

        # ▶ 번호판 인식
        if etype == "VEHICLE_DETECTED":
            if event.get("direction") == "ENTRY":
                self.state = SessionState.ENTERED
                return self._speak("입차가 확인되었습니다.")
            else:
                self.state = SessionState.EXIT_DETECTED
                if event.get("payment_status") != "PAID":
                    self.state = SessionState.PAYMENT_REQUIRED
                    return {
                        **self._speak("결제가 필요합니다."),
                        "lock_voice": True,
                    }
                self.state = SessionState.EXITED
                return self._speak("출차가 확인되었습니다.")

        # ▶ 사용자 음성
        if etype == "USER_SPOKE":
            self.state = SessionState.ASSISTING
            return {
                "state": self.state,
                "action": "HANDLE_INTENT",
                "text": event["text"],
            }

        # ▶ 결제 결과
        if etype == "PAYMENT_RESULT":
            if event["result"] == "SUCCESS":
                self.state = SessionState.PAID
                return self._speak("결제가 완료되었습니다.")
            else:
                self.state = SessionState.ASSISTING
                return self._speak("결제에 실패했습니다. 어떤 문제가 있으신가요?")

        return {"state": self.state}

    def _speak(self, text: str):
        return {
            "state": self.state,
            "action": "SPEAK",
            "text": text,
        }
