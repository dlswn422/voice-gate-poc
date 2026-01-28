# engine/intent_logger.py

import os
import psycopg2
from contextlib import contextmanager

DATABASE_URL = os.getenv("DATABASE_URL")


# ==================================================
# DB 연결 헬퍼
# ==================================================

@contextmanager
def get_conn():
    """
    PostgreSQL 커넥션 컨텍스트 매니저

    - 커넥션/트랜잭션 안전 처리
    - 예외 발생 시 rollback
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ==================================================
# Intent 로그 적재 (학습 데이터용)
# ==================================================

def log_intent(
    utterance: str,
    predicted_intent: str,
    predicted_confidence: float,
    source: str | None = None,
    site_id: str | None = None,
):
    """
    1차 의도 분류 결과를 학습 데이터로 DB에 적재한다.

    정책:
    - 실패해도 AppEngine 흐름을 절대 막지 않는다
    - 모든 예외는 내부에서 처리하고 swallow 한다
    """
    if not DATABASE_URL:
        print("⚠️ [INTENT_LOGGER] DATABASE_URL not set")
        return

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO intent_logs (
                        utterance,
                        predicted_intent,
                        predicted_confidence,
                        source,
                        site_id
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        utterance,
                        predicted_intent,
                        predicted_confidence,
                        source,
                        site_id,
                    ),
                )

    except Exception as e:
        # ❗ 절대 raise 하지 않음 (엔진 안정성 최우선)
        print("❌ [INTENT_LOGGER] Failed to log intent:", e)