from enum import Enum
from pydantic import BaseModel
from typing import Dict, List

class Intent(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    PAYMENT = "PAYMENT"
    REGISTRATION = "REGISTRATION"
    TIME_PRICE = "TIME_PRICE"
    FACILITY = "FACILITY"
    COMPLAINT = "COMPLAINT"
    NONE = "NONE"

# 메뉴얼 파일명 매핑
# RAG가 검색할 때 이 리스트에 포함된 문서들에 가산점
INTENT_TO_DOCS: Dict[str, List[str]] = {
    "ENTRY": ["entry_gate_not_open", "lpr_mismatch_or_no_entry_record", "barrier_physical_fault", "network_terminal_down"],
    "EXIT": ["exit_gate_not_open", "lpr_mismatch_or_no_entry_record", "barrier_physical_fault", "network_terminal_down"],
    "PAYMENT": ["payment_card_fail", "mobile_payment_qr_issue", "price_inquiry", "discount_free_time_issue", "network_terminal_down"],
    "REGISTRATION": ["visit_registration_fail", "lpr_mismatch_or_no_entry_record"],
    "TIME_PRICE": ["discount_free_time_issue", "price_inquiry"],
    "FACILITY": ["kiosk_ui_device_issue", "barrier_physical_fault", "network_terminal_down"],
    "COMPLAINT": [], 
    "NONE": []

    # "ENTRY": [],
    # "EXIT": [],
    # "PAYMENT": [],
    # "REGISTRATION": [],
    # "TIME_PRICE": [],
    # "FACILITY": [],
    # "COMPLAINT": [], 
    # "NONE": []
}

class IntentResult(BaseModel):
    intent: Intent
    confidence: float