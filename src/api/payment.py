from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal

from src.db.postgres import get_conn
from src.speech.tts import synthesize

router = APIRouter()

# ==================================================
# Request Schema
# ==================================================
class DemoPaymentRequest(BaseModel):
    parking_session_id: str
    result: Literal["SUCCESS", "FAIL"]
    reason: Optional[
        Literal[
            "LIMIT_EXCEEDED",
            "NETWORK_ERROR",
            "INSUFFICIENT_FUNDS",
            "USER_CANCEL",
            "ETC",
        ]
    ] = None


# ==================================================
# TTS Message Map
# ==================================================
PAYMENT_TTS = {
    "SUCCESS": "결제가 완료되었습니다.\n출차를 진행해주세요.",
    "LIMIT_EXCEEDED": "결제 한도를 초과했습니다.\n다른 결제 수단을 이용해주세요.",
    "NETWORK_ERROR": "결제 중 네트워크 오류가 발생했습니다.\n잠시 후 다시 시도해주세요.",
    "INSUFFICIENT_FUNDS": "잔액이 부족합니다.\n다른 결제 수단을 이용해주세요.",
    "USER_CANCEL": "결제가 취소되었습니다.",
    "ETC": "결제에 실패했습니다.\n관리자에게 문의해주세요.",
}


# ==================================================
# Demo Payment API
# ==================================================
@router.post("/api/payment/demo")
def demo_payment(req: DemoPaymentRequest):
    conn = get_conn()
    cur = conn.cursor()

    try:
        # --------------------------------------------------
        # 1️⃣ parking_session 조회
        # --------------------------------------------------
        cur.execute(
            """
            SELECT id, exit_time
            FROM parking_session
            WHERE id = %s
            LIMIT 1
            """,
            (req.parking_session_id,),
        )
        session = cur.fetchone()

        if not session:
            raise HTTPException(
                status_code=404,
                detail="PARKING_SESSION_NOT_FOUND",
            )

        # 출차 이후 결제 차단
        if session["exit_time"] is not None:
            raise HTTPException(
                status_code=409,
                detail="PAYMENT_NOT_ALLOWED_AFTER_EXIT",
            )

        # --------------------------------------------------
        # 2️⃣ payment 조회
        # --------------------------------------------------
        cur.execute(
            """
            SELECT id, payment_status
            FROM payment
            WHERE parking_session_id = %s
            LIMIT 1
            """,
            (req.parking_session_id,),
        )
        payment = cur.fetchone()

        if not payment:
            raise HTTPException(
                status_code=404,
                detail="PAYMENT_NOT_FOUND",
            )

        payment_id = payment["id"]
        payment_status = payment["payment_status"]

        # 이미 결제 완료
        if payment_status in ("PAID", "FREE"):
            raise HTTPException(
                status_code=409,
                detail="PAYMENT_ALREADY_COMPLETED",
            )

        # --------------------------------------------------
        # ✅ SUCCESS
        # --------------------------------------------------
        if req.result == "SUCCESS":
            cur.execute(
                """
                UPDATE payment
                SET payment_status = 'PAID'
                WHERE id = %s
                """,
                (payment_id,),
            )

            cur.execute(
                """
                INSERT INTO payment_log (
                    parking_session_id,
                    payment_id,
                    result,
                    reason
                )
                VALUES (%s, %s, 'SUCCESS', NULL)
                """,
                (req.parking_session_id, payment_id),
            )

            conn.commit()

            message = PAYMENT_TTS["SUCCESS"]

            return {
                "success": True,
                "result": "SUCCESS",
                "message": message,
                "tts_url": synthesize(message),
            }

        # --------------------------------------------------
        # ❌ FAIL
        # --------------------------------------------------
        if req.result == "FAIL":
            if not req.reason:
                raise HTTPException(
                    status_code=400,
                    detail="FAIL_REASON_REQUIRED",
                )

            cur.execute(
                """
                INSERT INTO payment_log (
                    parking_session_id,
                    payment_id,
                    result,
                    reason
                )
                VALUES (%s, %s, 'FAIL', %s)
                """,
                (req.parking_session_id, payment_id, req.reason),
            )

            conn.commit()

            message = PAYMENT_TTS.get(
                req.reason,
                "결제에 실패했습니다.\n다시 시도해주세요."
            )

            return {
                "success": True,
                "result": "FAIL",
                "reason": req.reason,
                "message": message,
                "tts_url": synthesize(message),
            }

    except HTTPException:
        conn.rollback()
        raise

    except Exception as e:
        conn.rollback()
        print("[PAYMENT DEMO ERROR]", e)
        raise HTTPException(
            status_code=500,
            detail="INTERNAL_ERROR",
        )

    finally:
        conn.close()
