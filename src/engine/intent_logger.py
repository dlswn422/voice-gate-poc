import os
import psycopg2
from contextlib import contextmanager

def _db_url() -> str | None:
    # dotenv가 main에서 load된 이후 값이 들어오므로, 호출 시점에 읽는 게 안전
    return os.getenv("DATABASE_URL")

@contextmanager
def get_conn():
    """
    PostgreSQL 커넥션 컨텍스트 매니저
    - 성공: commit
    - 예외: rollback
    - 마지막: close
    """
    conn = psycopg2.connect(_db_url())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def log_intent(
    utterance: str,
    predicted_intent: str,
    predicted_confidence: float,
    source: str,
    site_id: str,
) -> int | None:
    """
    1차 의도 분류 로그 저장 (intent_logs)
    - 성공 시 intent_logs.id 반환 (2차 dialog_logs FK로 사용)
    - 실패해도 raise 하지 않고 None 반환
    """
    if not _db_url():
        print("⚠ [INTENT_LOGGER] DATABASE_URL not set")
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
        # ❗ 절대 raise 하지 않음 (엔진 안정성 최우선)
        print("❌ [INTENT_LOGGER] Failed to log intent:", e)
        return None


# ==================================================
# 2차 상담 대화 로그 저장 (dialog_logs)
# ==================================================
def log_dialog(
    intent_log_id: int,
    session_id: str,
    role: str,
    content: str,
    model: str,
    turn_index: int,
) -> None:
    """
    2차 상담(LLM) 대화를 dialog_logs 테이블에 저장한다.
    - role: 'user' | 'assistant'
    - 실패해도 엔진 중단 X (best-effort)
    """
    if not _db_url():
        print("⚠ [INTENT_LOGGER] DATABASE_URL not set")
        return
        
    if intent_log_id is None:
        print("⚠ [INTENT_LOGGER] intent_log_id is None → skip dialog log")
        return

    if role not in ("user", "assistant"):
        print("⚠ [INTENT_LOGGER] Invalid role:", role)
        return

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
