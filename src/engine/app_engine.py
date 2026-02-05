from src.nlu.llm_client import detect_intent_llm
from src.nlu.intent_schema import Intent
from src.engine.intent_logger import log_intent, log_dialog
from src.nlu.dialog_llm_client import dialog_llm_chat

import uuid
import time
import re


# ==================================================
# 정책 설정
# ==================================================
CONFIDENCE_THRESHOLD = 0.75
SITE_ID = "parkassist_local"

IDLE_TIMEOUT_SEC = 15.0
DONE_COOLDOWN_SEC = 1.2

# ==================================================
# 2차: COMPLAINT 재분류(2차에서 의도 재확정)
# ==================================================
COMPLAINT_RECLASSIFY_MAX_TRIES = 2
COMPLAINT_RECLASSIFY_QUESTION = (
    "정확한 상황 판단을 위해, 지금 어떤 문제가 가장 크신가요? "
    "(결제/무료·할인/등록/입차/출차/기기오류)"
)

# ==================================================
# 2차 고도화 설정
# ==================================================
DEBUG_DIALOG = True
SECOND_STAGE_HARD_TURN_LIMIT = 6

REQUIRED_SLOTS_BY_INTENT = {
    "PAYMENT": ["symptom", "where"],
    "TIME_PRICE": ["symptom"],
    "REGISTRATION": ["symptom"],
    "ENTRY": ["symptom"],
    "EXIT": ["symptom", "where"],
    "FACILITY": ["symptom", "where"],
    "COMPLAINT": ["symptom"],
    "NONE": ["symptom"],
}

# ==================================================
# 원턴 응답 (⚠️ 질문형)
# ==================================================
ONE_TURN_RESPONSES = {
    Intent.EXIT: "출차하려면 요금 정산이 완료되어야 차단기가 열립니다. 혹시 정산은 이미 하셨나요?",
    Intent.ENTRY: "입차 시 차량이 인식되면 차단기가 자동으로 열립니다. 차량이 인식되지 않았다면 잠시 정차해 주세요.",
    Intent.PAYMENT: "주차 요금은 정산기나 출구에서 결제하실 수 있습니다. 이미 결제를 진행하셨나요?",
    Intent.REGISTRATION: "차량이나 방문자 등록은 키오스크에서 진행하실 수 있습니다. 아직 등록 전이신가요?",
    Intent.TIME_PRICE: "주차 시간과 요금은 키오스크 화면에서 확인하실 수 있습니다. 어느 부분이 궁금하신가요?",
    Intent.FACILITY: "기기나 차단기에 이상이 있는 경우 관리실 도움을 받으실 수 있습니다. 현재 어떤 문제가 발생했나요?",
}

NONE_RETRY_TEXT = (
    "말씀을 정확히 이해하지 못했어요. "
    "출차, 결제, 등록 중 어떤 도움을 원하시는지 말씀해 주세요."
)

DONE_KEYWORDS = [
    "됐어요", "되었습니다", "해결", "괜찮아요",
    "그만", "종료", "끝", "마칠게",
    "고마워", "감사", "안녕",
]

FAREWELL_TEXT = "네, 해결되셨다니 다행입니다. 이용해 주셔서 감사합니다. 안녕히 가세요."


# ==================================================
# 유틸
# ==================================================
def _normalize(text: str) -> str:
    return re.sub(r"[\s\.\,\!\?]+", "", text.strip().lower())


def _is_done_utterance(text: str) -> bool:
    t = _normalize(text)
    return any(_normalize(k) in t for k in DONE_KEYWORDS)


def _norm_intent_name(x) -> str:
    if x is None:
        return "NONE"
    if isinstance(x, Intent):
        return x.name
    s = str(x).strip().upper()
    # e.g. "Intent.PAYMENT"
    s = s.replace("INTENT.", "").replace(" ", "_")
    return s


def _merge_slots(prev: dict, new: dict) -> dict:
    out = dict(prev or {})
    for k, v in (new or {}).items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        out[k] = v
    return out


# ==================================================
# AppEngine
# ==================================================
class AppEngine:
    """
    ✔ 1차 의도 확정 후 세션 동안 의도 고정(기본) + 2차에서 intent 전환 허용(new_intent)
    ✔ 원턴(질문) → 다음 발화는 무조건 멀티턴
    ✔ FIRST_STAGE로 되돌아가지 않음
    ✔ idle-timeout 입력 중 종료 버그 해결
    ✔ ✅ 1차 COMPLAINT는 2차에서 재분류 질문 1개를 먼저 내보낸 뒤, 다음 user 답변으로 재분류
    """

    def __init__(self):
        self.state = "FIRST_STAGE"

        self.session_id = None
        self.first_intent = None
        self.current_intent = None        # ✅ 2차 현재 의도(전환 가능)
        self.intent_log_id = None

        self.dialog_turn_index = 0
        self.dialog_history = []

        self._none_retry_count = 0
        self._ignore_until_ts = 0.0
        self._last_activity_ts = 0.0

        self._last_handled_utterance_id = None
        self._just_one_turn = False

        # ✅ 1차에서 COMPLAINT로 잡힌 경우, 2차에서 재분류를 한 번 거친 뒤 정상 플로우로 진입
        self._pending_reclassify = False
        self._reclassify_try_count = 0

        # ✅ 2차 고도화: 슬롯/턴
        self.second_stage_slots = {}
        self.second_stage_user_turns = 0

    # --------------------------------------------------
    # 세션 시작
    # --------------------------------------------------
    def _start_new_session(self):
        self.session_id = str(uuid.uuid4())
        self.state = "FIRST_STAGE"

        self.first_intent = None
        self.current_intent = None
        self.intent_log_id = None
        self.dialog_turn_index = 0
        self.dialog_history = []

        self._none_retry_count = 0
        self._just_one_turn = False
        self._pending_reclassify = False
        self._reclassify_try_count = 0
        self._last_activity_ts = time.time()

        self.second_stage_slots = {}
        self.second_stage_user_turns = 0

        print(f"[ENGINE] 🆕 New session started: {self.session_id}")

    # --------------------------------------------------
    # 세션 종료
    # --------------------------------------------------
    def end_session(self, reason: str = ""):
        print(f"[ENGINE] 🛑 Session ended ({reason}): {self.session_id}")

        self.session_id = None
        self.state = "FIRST_STAGE"
        self.first_intent = None
        self.current_intent = None
        self.intent_log_id = None
        self.dialog_turn_index = 0
        self.dialog_history = []

        self._none_retry_count = 0
        self._just_one_turn = False
        self._pending_reclassify = False
        self._reclassify_try_count = 0
        self._last_handled_utterance_id = None

        self.second_stage_slots = {}
        self.second_stage_user_turns = 0

    # --------------------------------------------------
    # idle timeout (외부 watchdog용)
    # --------------------------------------------------
    def check_idle_timeout(self):
        if self.session_id and time.time() - self._last_activity_ts >= IDLE_TIMEOUT_SEC:
            self.end_session(reason="idle-timeout")

    # --------------------------------------------------
    # confidence
    # --------------------------------------------------
    def calculate_confidence(self, text: str, intent: Intent) -> float:
        score = 0.4
        KEYWORDS = {
            Intent.EXIT: ["출차", "나가", "차단기", "출구"],
            Intent.ENTRY: ["입차", "들어가", "입구"],
            Intent.PAYMENT: ["결제", "요금", "정산", "승인", "카드"],
            Intent.REGISTRATION: ["등록", "방문", "차량", "번호판"],
            Intent.TIME_PRICE: ["시간", "요금", "무료", "할인", "감면", "적용"],
            Intent.FACILITY: ["기계", "고장", "이상", "먹통", "오류", "통신", "서버"],
            Intent.COMPLAINT: ["왜", "안돼", "짜증", "말이돼", "불만"],
        }
        hits = sum(1 for k in KEYWORDS.get(intent, []) if k in text)
        score += 0.35 if hits else 0.15
        score += 0.05 if len(text) <= 4 else 0.2
        return round(min(score, 1.0), 2)

    # --------------------------------------------------
    # dialog log
    # --------------------------------------------------
    def _log_dialog(self, role, content, model="stt"):
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

    # --------------------------------------------------
    # context builder
    # --------------------------------------------------
    def _build_second_stage_context(self) -> dict:
        cur = _norm_intent_name(self.current_intent or self.first_intent)
        req = REQUIRED_SLOTS_BY_INTENT.get(cur, ["symptom"])

        return {
            "session_id": self.session_id,
            "intent_log_id": self.intent_log_id,

            # ✅ 최초 의도 + 현재 의도(전환 가능)
            "first_intent": _norm_intent_name(self.first_intent),
            "current_intent": cur,

            # ✅ 턴 제한
            "turn_count_user": self.second_stage_user_turns,
            "hard_turn_limit": SECOND_STAGE_HARD_TURN_LIMIT,

            # ✅ 슬롯 정형화
            "slots": self.second_stage_slots,
            "required_slots": req,
        }

    # --------------------------------------------------
    # 2차: COMPLAINT 재분류용 질문 출력
    # --------------------------------------------------
    def _ask_reclassify_question(self):
        self._log_dialog("assistant", COMPLAINT_RECLASSIFY_QUESTION, model="system")
        print(f"[DIALOG] {COMPLAINT_RECLASSIFY_QUESTION}")

    # --------------------------------------------------
    # 2차: COMPLAINT 재분류 수행 (user 답변 기반)
    # --------------------------------------------------
    def _try_reclassify_from_user_text(self, text: str) -> str | None:
        try:
            r = detect_intent_llm(text)
            r.confidence = self.calculate_confidence(text, r.intent)
            ni = _norm_intent_name(getattr(r, "intent", None))
            if ni in ("NONE", "COMPLAINT", ""):
                return None
            return ni
        except Exception as e:
            print(f"[ENGINE] ⚠️ reclassify failed: {e}")
            return None

    # --------------------------------------------------
    # SECOND_STAGE
    # --------------------------------------------------
    def _handle_second_stage(self, text: str, *, already_logged_user: bool = False):
        # 1) 사용자 종료 발화 → 종료
        if _is_done_utterance(text):
            if not already_logged_user:
                self._log_dialog("user", text)
            self._log_dialog("assistant", FAREWELL_TEXT, model="system")
            print(f"[DIALOG] {FAREWELL_TEXT}")
            self.end_session(reason="done")
            self._ignore_until_ts = time.time() + DONE_COOLDOWN_SEC
            return

        # 2) user 로그(중복 방지)
        if not already_logged_user:
            self._log_dialog("user", text)

        # history 중복 방지
        history_for_llm = self.dialog_history
        if history_for_llm and history_for_llm[-1]["role"] == "user" and history_for_llm[-1]["content"] == text:
            history_for_llm = history_for_llm[:-1]

        # 3) 2차 모델 호출
        res = dialog_llm_chat(
            text,
            history=history_for_llm,
            context=self._build_second_stage_context(),
            debug=DEBUG_DIALOG,
        )

        reply = getattr(res, "reply", "") or "조금 더 자세히 말씀해 주실 수 있을까요?"
        action = (getattr(res, "action", "") or "").strip().upper()
        new_intent = getattr(res, "new_intent", None)

        # 4) 슬롯 누적 merge (정형화 핵심)
        self.second_stage_slots = _merge_slots(self.second_stage_slots, getattr(res, "slots", {}) or {})

        # 5) ✅ intent 전환 허용 (new_intent 수신 시)
        if isinstance(new_intent, str):
            ni = _norm_intent_name(new_intent)
            if ni != "NONE" and ni != _norm_intent_name(self.current_intent):
                print(f"[ENGINE] 🔀 intent switched: {self.current_intent} -> {ni}")
                self.current_intent = ni

        # 6) assistant 로그/출력
        self._log_dialog("assistant", reply, model="llama-3.1-8b")
        print(f"[DIALOG] {reply}")

        # ✅ 이번 user 입력은 2차에서 1턴 소비한 것으로 카운트 증가
        self.second_stage_user_turns += 1

        # 7) 세션 종료 트리거들
        if action in ("DONE", "ESCALATE_DONE"):
            self.end_session(reason=action.lower())
            self._ignore_until_ts = time.time() + DONE_COOLDOWN_SEC
            return

    # --------------------------------------------------
    # STT 엔트리포인트
    # --------------------------------------------------
    def handle_text(self, text: str, *, utterance_id: str | None = None):
        now = time.time()

        if not text or not text.strip():
            return
        if now < self._ignore_until_ts:
            return

        # ✅ 입력이 들어왔으므로 활동 시간 갱신
        self._last_activity_ts = now

        # STT 중복 방지
        if utterance_id and utterance_id == self._last_handled_utterance_id:
            print("[ENGINE] ⚠️ duplicated utterance ignored")
            return
        self._last_handled_utterance_id = utterance_id

        if not self.session_id:
            self._start_new_session()

        print("=" * 50)
        print(f"[ENGINE] State={self.state}")
        print(f"[ENGINE] Text={text}")

        # ==================================================
        # 🔥 원턴 직후 후속 발화 → 무조건 멀티턴
        # ==================================================
        if self._just_one_turn:
            print("[ENGINE] 🔁 one-turn follow-up → SECOND_STAGE")
            self.state = "SECOND_STAGE"
            self._just_one_turn = False
            # 원턴 이후 진입은 2차 턴 카운트 0에서 시작
            self._handle_second_stage(text)
            return

        # --------------------------------------------------
        # FIRST_STAGE
        # --------------------------------------------------
        if self.state == "FIRST_STAGE":
            result = detect_intent_llm(text)
            result.confidence = self.calculate_confidence(text, result.intent)

            print(f"[ENGINE] Intent={result.intent.name}, confidence={result.confidence:.2f}")

            self.intent_log_id = log_intent(
                utterance=text,
                predicted_intent=result.intent.value,
                predicted_confidence=result.confidence,
                source="kiosk",
                site_id=SITE_ID,
            )

            self.first_intent = result.intent.value
            self.current_intent = result.intent.value
            self._log_dialog("user", text)

            if result.intent == Intent.NONE:
                self._none_retry_count += 1
                self._log_dialog("assistant", NONE_RETRY_TEXT, model="system")
                print(f"[ONE-TURN] {NONE_RETRY_TEXT}")
                return

            # ✅ 1차 COMPLAINT는 2차에서 의도 재분류 질문을 먼저 1회 출력
            if result.intent == Intent.COMPLAINT:
                self.state = "SECOND_STAGE"
                self._pending_reclassify = True
                self._reclassify_try_count = 0
                self._ask_reclassify_question()
                return

            if result.confidence < CONFIDENCE_THRESHOLD:
                self.state = "SECOND_STAGE"
                self._handle_second_stage(text, already_logged_user=True)
                return

            reply = ONE_TURN_RESPONSES.get(result.intent)
            self._log_dialog("assistant", reply, model="system")
            print(f"[ONE-TURN] {reply}")
            self._just_one_turn = True
            return

        # --------------------------------------------------
        # SECOND_STAGE
        # --------------------------------------------------
        if self.state == "SECOND_STAGE":
            # ✅ COMPLAINT 재분류 대기 상태: 이번 user 답변으로 의도를 재분류한 뒤 정상 플로우로 진입
            if self._pending_reclassify:
                # user 로그는 여기서 남기고, 같은 입력으로 바로 2차 플로우를 이어감
                self._log_dialog("user", text)

                ni = self._try_reclassify_from_user_text(text)
                if ni is None:
                    self._reclassify_try_count += 1
                    if self._reclassify_try_count >= COMPLAINT_RECLASSIFY_MAX_TRIES:
                        msg = (
                            "현재 말씀하신 내용만으로는 문제 유형을 정확히 분류하기 어렵습니다. "
                            "관리자를 호출해 도움을 받아주세요. 이용해 주셔서 감사합니다. 안전운전하세요."
                        )
                        self._log_dialog("assistant", msg, model="system")
                        print(f"[DIALOG] {msg}")
                        self.end_session(reason="escalate_done")
                        self._ignore_until_ts = time.time() + DONE_COOLDOWN_SEC
                        return

                    self._ask_reclassify_question()
                    return

                print(f"[ENGINE] 🔀 reclassified from COMPLAINT -> {ni}")
                self.current_intent = ni
                self._pending_reclassify = False

                # 같은 입력을 바로 2차 슬롯/질문 플로우에 반영 (이미 user 로그를 남겼으므로 중복 방지)
                self._handle_second_stage(text, already_logged_user=True)
                return

            self._handle_second_stage(text)
            return
