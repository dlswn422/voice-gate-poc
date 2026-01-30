# engine/intent_logger.py

from __future__ import annotations

import os
from contextlib import contextmanager

import psycopg2


def _db_url() -> str | None:
    # ✅ import 시점 고정 금지: 매 호출 시점에 읽어야 .env 로드 이후에도 동작
    return os.getenv("DATABASE_URL")


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
    database_url = _db_url()
    if not database_url:
        raise RuntimeError("DATABASE_URL not set")

    conn = psycopg2.connect(database_url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ==================================================
# 1차 Intent 로그 적재 (학습 데이터용)
# ==================================================
def log_intent(
    utterance: str,
    predicted_intent: str,
    predicted_confidence: float,
    source: str | None = None,
    site_id: str | None = None,
) -> int | None:
    """
    1차 의도 분류 결과를 DB(intent_logs)에 적재한다.

    정책:
    - 실패해도 AppEngine 흐름을 절대 막지 않는다(best-effort)
    반환:
    - 성공 시: intent_logs.id(int)
    - 실패 시: None
    """
    if not _db_url():
        print("⚠️ [INTENT_LOGGER] DATABASE_URL not set")
        return None

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
                    RETURNING id
                    """,
                    (utterance, predicted_intent, predicted_confidence, source, site_id),
                )
                return cur.fetchone()[0]
    except Exception as e:
<<<<<<< HEAD
        print("❌ [INTENT_LOGGER] Failed to log intent:", e)
        return None


# ==================================================
# 2차 상담 대화 로그 저장 (dialog_logs)
# ==================================================
=======
        # ❗ 절대 raise 하지 않음 (엔진 안정성 최우선)
        print("❌ [INTENT_LOGGER] Failed to log intent:", e)
        
        
# ==================================================
# 2️⃣ 2차 상담 대화 로그 저장
# ==================================================

>>>>>>> origin/hanse/stt
def log_dialog(
    intent_log_id: int,
    session_id: str,
    role: str,
    content: str,
    model: str,
    turn_index: int,
<<<<<<< HEAD
) -> None:
    """
    2차 상담(라마) 대화를 dialog_logs 테이블에 저장한다.

    role: 'user' | 'assistant'
    정책: 실패해도 엔진 중단 X(best-effort)
    """
    if not _db_url():
        print("⚠️ [INTENT_LOGGER] DATABASE_URL not set")
        return
=======
):
    """
    2차 상담(라마) 대화를 dialog_logs 테이블에 저장한다.

    role:
        - 'user'
        - 'assistant'
    """
>>>>>>> origin/hanse/stt

    if role not in ("user", "assistant"):
        raise ValueError("role must be 'user' or 'assistant'")

<<<<<<< HEAD
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO dialog_logs (
                        intent_log_id,
                        session_id,
                        role,
                        content,
                        model,
                        turn_index
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (intent_log_id, session_id, role, content, model, turn_index),
                )
    except Exception as e:
        print("❌ [INTENT_LOGGER] Failed to log dialog:", e)
=======
    sql = """
        INSERT INTO dialog_logs (
            intent_log_id,
            session_id,
            role,
            content,
            model,
            turn_index
        )
        VALUES (%s, %s, %s, %s, %s, %s);
    """

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    intent_log_id,
                    session_id,
                    role,
                    content,
                    model,
                    turn_index,
                ),
            )
            conn.commit()
    finally:
        conn.close()
>>>>>>> origin/hanse/stt
