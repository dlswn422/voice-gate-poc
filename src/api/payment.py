from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal

from src.db.postgres import get_conn

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
                detail="PARKING_SESSION_NOT_FOUND"
            )

        # --------------------------------------------------
        # 2️⃣ 출차 상태가 아닌 경우 → 결제 차단
        # exit_time IS NULL = 아직 출차 전
        # --------------------------------------------------
        if session["exit_time"] is not None:
            raise HTTPException(
                status_code=409,
                detail="PAYMENT_NOT_ALLOWED_AFTER_EXIT"
            )

        # --------------------------------------------------
        # 3️⃣ payment 조회
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
                detail="PAYMENT_NOT_FOUND"
            )

        payment_id = payment["id"]
        payment_status = payment["payment_status"]

        # --------------------------------------------------
        # 4️⃣ 이미 결제 완료된 경우 → 차단
        # --------------------------------------------------
        if payment_status in ("PAID", "FREE"):
            raise HTTPException(
                status_code=409,
                detail="PAYMENT_ALREADY_COMPLETED"
            )

        # --------------------------------------------------
        # 5️⃣ SUCCESS 처리
        # --------------------------------------------------
        if req.result == "SUCCESS":
            # payment 상태 업데이트
            cur.execute(
                """
                UPDATE payment
                SET payment_status = 'PAID'
                WHERE id = %s
                """,
                (payment_id,),
            )

            # payment_log 기록
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

            return {
                "success": True,
                "result": "SUCCESS",
            }

        # --------------------------------------------------
        # 6️⃣ FAIL 처리
        # --------------------------------------------------
        if req.result == "FAIL":
            if not req.reason:
                raise HTTPException(
                    status_code=400,
                    detail="FAIL_REASON_REQUIRED"
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

            return {
                "success": True,
                "result": "FAIL",
                "reason": req.reason,
            }

    except HTTPException:
        conn.rollback()
        raise

    except Exception as e:
        conn.rollback()
        print("[PAYMENT DEMO ERROR]", e)
        raise HTTPException(
            status_code=500,
            detail="INTERNAL_ERROR"
        )

    finally:
        conn.close()
