"""
dispatcher.py
─────────────
Intent 분류 결과 → DB 로직 라우팅 → Raw DB 데이터 반환.

★ 이전 버전과의 핵심 차이:
  tts_message 를 이 계층에서 하드코딩하지 않습니다.
  대신 DB 에서 조회한 원시 데이터(raw_data)만 반환하고,
  자연어 멘트 생성은 intent.py 의 generate_reply_stream() 이 담당합니다.

반환 스키마:
  {
      "status"    : "success" | "escalate" | "error" | "info",
      "intent"    : str,
      "raw_data"  : dict,   # LLaMA Step2 에 그대로 전달될 원시 DB 데이터
      "escalate"  : bool,   # True 면 main.py 가 관리자 알림 추가 실행
  }
"""

import logging
from supabase import Client

from intents import fee_logic, payment_logic, facility_logic

logger = logging.getLogger(__name__)



# ══════════════════════════════════════════════════════════════
# 에스컬레이션 (분노 감지 or 기기 고장)
# ══════════════════════════════════════════════════════════════
def _make_escalate(intent: str, reason: str) -> dict:
    logger.warning("⚠️  에스컬레이션 | intent=%s | reason=%s", intent, reason)
    return {
        "status": "escalate",
        "intent": intent,
        "escalate": True,
        "raw_data": {
            "situation": "escalate",
            "reason": reason,
            "action": "관리자 즉시 호출",
        },
    }


# ══════════════════════════════════════════════════════════════
# Intent 별 DB 라우터
# ══════════════════════════════════════════════════════════════
def _route(supabase: Client, plate_number: str, intent: str) -> dict:
    """
    intent 에 따라 DB 로직을 호출하고 raw_data 를 반환합니다.
    tts_message 를 직접 생성하지 않습니다.
    """

    # ── admin ─────────────────────────────────────────────────
    if intent == "admin":
        return _make_escalate(intent=intent, reason="사용자 관리자 호출 요청")

    # ── fee ──────────────────────────────────────────────────
    if intent == "fee":
        result = fee_logic.get_fee_info(supabase, plate_number)
        if result.get("status") != "success":
            return {
                "status": "error", "intent": intent, "escalate": False,
                "raw_data": {"error": result.get("message", "요금 조회 실패")},
            }
        return {
            "status": "success", "intent": intent, "escalate": False,
            "raw_data": {
                "plate_number":   result["plate_number"],
                "is_monthly":     result["is_monthly"],
                "total_minutes":  result["total_minutes"],
                "calculated_fee": result["calculated_fee"],
            },
        }

    # ── payment ───────────────────────────────────────────────
    elif intent == "payment":
        result = payment_logic.get_payment_status(supabase, plate_number)
        if result.get("payment_status") == "NONE":
            return {
                "status": "info", "intent": intent, "escalate": False,
                "raw_data": {"payment_status": "NONE", "message": "결제 이력 없음"},
            }
        if result.get("status") not in ("success", "info"):
            return {
                "status": "error", "intent": intent, "escalate": False,
                "raw_data": {"error": result.get("message", "결제 조회 실패")},
            }
        return {
            "status": "success", "intent": intent, "escalate": False,
            "raw_data": {
                "plate_number":   plate_number,
                "payment_status": result.get("payment_status"),
                "fail_reason":    result.get("fail_reason", ""),
                "tried_at":       result.get("tried_at", ""),
            },
        }

    # ── facility ──────────────────────────────────────────────
    elif intent == "facility":
        result = facility_logic.get_device_status(supabase, "GATE_OUT_01")
        if result.get("status") != "success":
            return {
                "status": "error", "intent": intent, "escalate": False,
                "raw_data": {"error": result.get("message", "기기 상태 조회 실패")},
            }
        if result.get("need_admin"):
            return _make_escalate(
                intent=intent,
                reason=f"기기({result['device_id']}) {result['status_description']}",
            )
        return {
            "status": "success", "intent": intent, "escalate": False,
            "raw_data": {
                "device_id":          result["device_id"],
                "device_status":      result["device_status"],
                "status_description": result["status_description"],
            },
        }

    # ── none / fallback ───────────────────────────────────────
    else:
        return {
            "status": "info", "intent": "none", "escalate": False,
            "raw_data": {"message": "분류 불가"},
        }


# ══════════════════════════════════════════════════════════════
# 공개 인터페이스
# ══════════════════════════════════════════════════════════════
def dispatch(
    supabase: Client,
    plate_number: str,
    intent: str,
) -> dict:
    """
    LLaMA Step1 분류 결과를 받아 DB 로직을 실행합니다.

    Returns:
        {
            "status"   : "success" | "escalate" | "error" | "info"
            "intent"   : str
            "escalate" : bool
            "raw_data" : dict   ← intent.generate_reply_stream() 에 전달
        }
    """
    logger.info(
        "[dispatch] plate=%-10s intent=%s",
        plate_number, intent,
    )
    return _route(supabase, plate_number, intent)
