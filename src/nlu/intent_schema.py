from enum import Enum
from pydantic import BaseModel


class Intent(str, Enum):
    OPEN_GATE = "OPEN_GATE"
    CLOSE_GATE = "CLOSE_GATE"
    NONE = "NONE"


class IntentResult(BaseModel):
    intent: Intent
    confidence: float
