from src.nlu.intent_embedding import detect_intent_embedding
from src.nlu.intent_schema import Intent
from src.engine.intent_logger import log_intent, log_dialog
from src.nlu.dialog_llm_client import dialog_llm_chat

import uuid
import time
import re
from typing import Dict, Any, Optional


# ==================================================
# 정책 설정
# ==================================================
SITE_ID = "parkassist_local"

DONE_COOLDOWN_SEC = 1.2
SECOND_STAGE_HARD_TURN_LIMIT = 6

# 🔴 욕설/공격 감지 스위치 (ADD)
ENABLE_AGGRESSION_GUARD = True

MASKED_PROFANITY_PATTERNS = [
    r"씨[\*xX]+발",
    r"씨[\*xX]+",
    r"개[\*xX]+",
    r"[\*xX]+나",
]

# ==================================================
# NONE 시 안내 메시지
# ==================================================
NONE_RETRY_TEXT = (
    "말씀을 정확히 이해하지 못했어요. "
    "어떤 도움을 원하시는지 말씀해 주세요."
)


# ==================================================
# 종료 감지
# ==================================================
DONE_KEYWORDS = [
    "됐어요", "되었습니다", "해결", "괜찮아요",
    "그만", "종료", "끝", "마칠게",
    "고마워", "감사", "안녕",
]

FAREWELL_TEXT = "네, 이용해 주셔서 감사합니다. 안녕히 가세요."


# ==================================================
# 관리실 호출 인터럽트 (전역)
# ==================================================
CALL_ADMIN_KEYWORDS = [
    "관리실", "관리인",
    "직원", "사람",
    "불러", "불러줘", "와줘",
    "호출", "연결",
    "도와", "도움",
]

# 🔴 욕설/공격 감지 키워드 (ADD)
PROFANITY_KEYWORDS = [
    "씨발", "시발", "병신", "미친", "좆", "개새끼",
    "fuck", "shit", "asshole",
]

AGGRESSIVE_PATTERNS = [
    r"사람.*나와",
    r"책임자",
    r"당장.*불러",
    r"가만.*안",
    r"똑바로.*해",
]

def _contains_masked_profanity(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in MASKED_PROFANITY_PATTERNS)


def _normalize(text: str) -> str:
    return re.sub(r"[\s\.\,\!\?]+", "", text.strip().lower())


def _is_done_utterance(text: str) -> bool:
    t = _normalize(text)
    return any(_normalize(k) == t for k in DONE_KEYWORDS)


def _is_call_admin_utterance(text: str) -> bool:
    t = _normalize(text)
    return any(_normalize(k) in t for k in CALL_ADMIN_KEYWORDS)


# 🔴 욕설/공격 감지 함수 (ADD)
def _contains_profanity(text: str) -> bool:
    t = _normalize(text)
    return any(k in t for k in PROFANITY_KEYWORDS)


def _contains_aggression(text: str) -> bool:
    t = _normalize(text)
    return any(re.search(p, t) for p in AGGRESSIVE_PATTERNS)


class AppEngine:
    """
    AppEngine (REFINED)

    ✔ 전역 인터럽트(관리실 호출)
    ✔ 1차 의도 분류 -> 곧바로 dialog_llm_client로 메뉴얼 기반 응답
    ✔ 질문 생성 없음 (LLM이 질문하지 않음)
    ✔ PAYMENT일 때 DB(payment/payment_log) 조회 결과를 context로 전달
    ✔ LPR일 때 direction(ENTRY/EXIT) context로 전달하여 메뉴얼 후보 제한
    """

    def __init__(self):
        self._reset_all()

    def _reset_all(self):
        self.session_id = None
        self.state = "FIRST_STAGE"
        self.first_intent = None
        self.intent_log_id = None

        self.dialog_turn_index = 0
        self.dialog_history = []

        self._ignore_until_ts = 0.0

        self.second_turn_count_user = 0
        self.second_slots = {}
        self.second_pending_slot = None

    def _start_new_session(self):
        self.session_id = str(uuid.uuid4())
        self.state = "FIRST_STAGE"
        self.dialog_turn_index = 0
        self.dialog_history = []

        self.second_turn_count_user = 0
        self.second_slots = {}
        self.second_pending_slot = None

        print(f"[ENGINE] 🆕 New session started: {self.session_id}")

    def _end_session(self, reason: str):
        print(f"[ENGINE] 🛑 Session ended ({reason}): {self.session_id}")
        self._reset_all()
        self._ignore_until_ts = time.time() + DONE_COOLDOWN_SEC

    def _log_dialog(self, role, content, model="system"):
        self.dialog_turn_index += 1
        log_dialog(
            intent_log_id=self.intent_log_id,
            session_id=self.session_id,
            role=role,
            content=content,
            model=model,
            turn_index=self.dialog_turn_index,
        )

        if role in ("user", "assistant"):
            self.dialog_history.append({"role": role, "content": content})

    def _make_response(
        self,
        text: str,
        *,
        conversation_state: str,
        end_session: bool = False,
        intent: str | None = None,
        system_action: str | None = None,
    ):
        return {
            "type": "assistant_message",
            "text": text,
            "conversation_state": conversation_state,
            "end_session": end_session,
            "session_id": self.session_id,
            "intent": intent,
            "system_action": system_action,
        }

    def _handle_call_admin(self, text: str):
        self._log_dialog("user", text)
        reply = "불편을드려 죄송합니다.\n관리실에 연락했습니다.\n잠시만 기다려 주세요."
        self._log_dialog("assistant", reply)
        self._end_session("call_admin")

        return self._make_response(
            reply,
            conversation_state="ENDED",
            end_session=True,
            system_action="CALL_ADMIN",
        )

    def _fetch_payment_ctx(self) -> Optional[Dict[str, Any]]:
        try:
            from src import app_state
            from src.db.postgres import get_conn
        except Exception as e:
            print(f"[ENGINE][PAYMENT_CTX] import failed: {e}")
            return None

        psid = getattr(app_state, "current_parking_session_id", None)
        if not psid:
            return None

        ctx: Dict[str, Any] = {
            "parking_session_id": str(psid),
            "payment_id": None,
            "payment_status": None,
            "has_attempt": False,
            "log_result": None,
            "log_reason": None,
        }

        conn = None
        try:
            conn = get_conn()
            cur = conn.cursor()

            cur.execute(
                """
                SELECT id, payment_status
                FROM payment
                WHERE parking_session_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (psid,),
            )
            pay = cur.fetchone()
            if not pay:
                return ctx

            payment_id = pay["id"]
            ctx["payment_id"] = str(payment_id)
            ctx["payment_status"] = pay.get("payment_status")

            cur.execute(
                """
                SELECT result, reason
                FROM payment_log
                WHERE payment_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (payment_id,),
            )
            log = cur.fetchone()
            if log:
                ctx["has_attempt"] = True
                ctx["log_result"] = log.get("result")
                ctx["log_reason"] = log.get("reason")

            return ctx

        except Exception as e:
            print(f"[ENGINE][PAYMENT_CTX] query failed: {e}")
            return ctx
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass

    def _get_direction_ctx(self) -> Optional[str]:
        try:
            from src import app_state
            d = getattr(app_state, "current_direction", None)
            if d:
                return str(d).upper()
        except Exception:
            pass
        return None

    def _run_dialog(self, text: str) -> Dict[str, Any]:
        direction = self._get_direction_ctx()

        payment_ctx = None
        if (self.first_intent or "").upper() == "PAYMENT":
            payment_ctx = self._fetch_payment_ctx()

        res = dialog_llm_chat(
            text,
            history=self.dialog_history,
            context={
                "session_id": self.session_id,
                "intent": self.first_intent,
                "turn_count_user": self.second_turn_count_user,
                "hard_turn_limit": SECOND_STAGE_HARD_TURN_LIMIT,
                "slots": self.second_slots,
                "pending_slot": self.second_pending_slot,
                "direction": direction,
                "payment_ctx": payment_ctx,
            },
            debug=True,
        )

        reply = getattr(res, "reply", "") or "조금 더 자세히 말씀해 주실 수 있을까요?"
        self._log_dialog("assistant", reply, model="llama-3.1-8b")

        self.second_turn_count_user += 1
        self.second_slots = getattr(res, "slots", self.second_slots) or self.second_slots
        self.second_pending_slot = getattr(res, "pending_slot", None)

        action = getattr(res, "action", "SOLVE")
        if action in ("DONE", "ESCALATE_DONE"):
            self._end_session("llm_done")
            return self._make_response(
                reply,
                conversation_state="ENDED",
                end_session=True,
                intent=self.first_intent,
            )

        return self._make_response(
            reply,
            conversation_state="WAITING_USER",
            end_session=False,
            intent=self.first_intent,
        )

    def handle_text(self, text: Any) -> Dict[str, Any]:
        now = time.time()

        if isinstance(text, dict) and text.get("type") == "ui_keyword":
            text = text.get("text", "")

        if not isinstance(text, str) or not text.strip():
            return self._make_response(
                "다시 한 번 말씀해 주세요.",
                conversation_state="WAITING_USER",
            )

        if now < self._ignore_until_ts:
            return self._make_response("", conversation_state="WAITING_USER")

        if not self.session_id:
            self._start_new_session()

        # 🔴 욕설/공격 감지 → 즉시 관리실 호출 (ADD)
        if ENABLE_AGGRESSION_GUARD and (
            _contains_profanity(text) or _contains_aggression(text) or _contains_masked_profanity(text)
        ):
            return self._handle_call_admin(text)

        if _is_call_admin_utterance(text):
            return self._handle_call_admin(text)

        if _is_done_utterance(text):
            self._log_dialog("user", text)
            self._log_dialog("assistant", FAREWELL_TEXT)
            self._end_session("user_done")
            return self._make_response(
                FAREWELL_TEXT,
                conversation_state="ENDED",
                end_session=True,
            )

        if self.state == "FIRST_STAGE":
            result = detect_intent_embedding(text)

            self.intent_log_id = log_intent(
                utterance=text,
                predicted_intent=result.intent.value,
                predicted_confidence=result.confidence,
                source="kiosk",
                site_id=SITE_ID,
            )

            self.first_intent = result.intent.value
            self._log_dialog("user", text)

            intent_value = (result.intent.value or "").upper()
            none_enum = getattr(Intent, "NONE", None)
            is_none = (none_enum is not None and result.intent == none_enum) or (intent_value == "NONE")

            if is_none:
                self._log_dialog("assistant", NONE_RETRY_TEXT)
                return self._make_response(
                    NONE_RETRY_TEXT,
                    conversation_state="WAITING_USER",
                    end_session=False,
                    intent="NONE",
                )

            self.state = "SECOND_STAGE"
            return self._run_dialog(text)

        return self._run_dialog(text)
