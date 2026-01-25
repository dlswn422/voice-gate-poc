from enum import Enum
from pydantic import BaseModel


class Intent(str, Enum):
    # 즉시 제어
    OPEN_GATE = "OPEN_GATE"
    CLOSE_GATE = "CLOSE_GATE"

    # 안내 / 도움
    HELP_REQUEST = "HELP_REQUEST"   # 문제 상황, 안됨, 오류
    INFO_REQUEST = "INFO_REQUEST"   # 방법 문의, 절차 질문

    # 무관
    NONE = "NONE"


class IntentResult(BaseModel):
    intent: Intent
    confidence: float
