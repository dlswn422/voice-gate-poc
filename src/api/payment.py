from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal

from src.db.postgres import get_conn
import src.app_state as app_state

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
# Demo Payment API (FINAL)
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

            # 🔔 중앙 세션 엔진에 이벤트 전달
            engine_result = app_state.session_engine.handle_event({
                "type": "PAYMENT_RESULT",
                "result": "SUCCESS",
            })

            return {
                "success": True,
                "result": "SUCCESS",
                **engine_result,
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

            engine_result = app_state.session_engine.handle_event({
                "type": "PAYMENT_RESULT",
                "result": "FAIL",
                "reason": req.reason,
            })

            return {
                "success": True,
                "result": "FAIL",
                "reason": req.reason,
                **engine_result,
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
