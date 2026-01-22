from enum import Enum
from typing import TypedDict


class Intent(str, Enum):
    OPEN_GATE = "OPEN_GATE"
    CALL_ADMIN = "CALL_ADMIN"
    INFO = "INFO"
    UNKNOWN = "UNKNOWN"


class IntentResult(TypedDict):
    intent: Intent
    confidence: float
    requires_human: bool