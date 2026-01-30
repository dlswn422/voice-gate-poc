# engine/intent_logger.py

import os
import psycopg2
from contextlib import contextmanager

# ??(蹂寃? env??import ?쒖젏??怨좎젙?섏? 留먭퀬, 留??몄텧留덈떎 媛?몄삤??def _db_url() -> str | None:
    return os.getenv("DATABASE_URL")


# ==================================================
# DB ?곌껐 ?ы띁
# ==================================================

@contextmanager
def get_conn():
    """
    PostgreSQL 而ㅻ꽖??而⑦뀓?ㅽ듃 留ㅻ땲?

    - 而ㅻ꽖???몃옖??뀡 ?덉쟾 泥섎━
    - ?덉쇅 諛쒖깮 ??rollback
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
# Intent 濡쒓렇 ?곸옱 (?숈뒿 ?곗씠?곗슜)
# ==================================================

def log_intent(
    utterance: str,
    predicted_intent: str,
    predicted_confidence: float,
    source: str | None = None,
    site_id: str | None = None,
) -> int | None:
    """
    1李??섎룄 遺꾨쪟 寃곌낵瑜??숈뒿 ?곗씠?곕줈 DB???곸옱?쒕떎.

    ?뺤콉:
    - ?ㅽ뙣?대룄 AppEngine ?먮쫫???덈? 留됱? ?딅뒗??    - 紐⑤뱺 ?덉쇅???대??먯꽌 泥섎━?섍퀬 swallow ?쒕떎

    諛섑솚:
    - ?깃났 ?? intent_logs??id(int)
    - ?ㅽ뙣 ?? None
    """
    if not _db_url():
        print("?좑툘 [INTENT_LOGGER] DATABASE_URL not set")
        return None  # ??(蹂寃? 紐낇솗??None 諛섑솚

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
                    (
                        utterance,
                        predicted_intent,
                        predicted_confidence,
                        source,
                        site_id,
                    ),
                )
                intent_log_id = cur.fetchone()[0]  # ??(異붽?) PK 諛섑솚
                return intent_log_id

    except Exception as e:
        # ???덈? raise ?섏? ?딆쓬 (?붿쭊 ?덉젙??理쒖슦??
        print("??[INTENT_LOGGER] Failed to log intent:", e)
        return None


# ==================================================
# 2截뤴깵 2李??곷떞 ???濡쒓렇 ???# ==================================================

def log_dialog(
    intent_log_id: int,
    session_id: str,
    role: str,
    content: str,
    model: str,
    turn_index: int,
) -> None:
    """
    2李??곷떞(?쇰쭏) ??붾? dialog_logs ?뚯씠釉붿뿉 ??ν븳??

    role:
        - 'user'
        - 'assistant'

    ?뺤콉:
    - ?ㅽ뙣?대룄 AppEngine ?먮쫫??留됱? ?딅뒗??best-effort)
    """
    if not _db_url():
        print("?좑툘 [INTENT_LOGGER] DATABASE_URL not set")
        return

    if role not in ("user", "assistant"):
        raise ValueError("role must be 'user' or 'assistant'")

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

    try:
        with get_conn() as conn:
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

    except Exception as e:
        print("??[INTENT_LOGGER] Failed to log dialog:", e)
