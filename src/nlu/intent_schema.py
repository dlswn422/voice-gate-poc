from enum import Enum
from pydantic import BaseModel


class Intent(str, Enum):
    """
    주차장 키오스크 CX 기준 '1차 발화 의도 분류' Enum (축소판)

    - 1차는 오직 라우팅용 태깅
    - 실행/조치/질문 생성은 하지 않음
    """

    PAYMENT = "PAYMENT"
    REGISTRATION = "REGISTRATION"
    FACILITY = "FACILITY"     
    NONE = "NONE"

class IntentResult(BaseModel):
    intent: Intent
    confidence: float
