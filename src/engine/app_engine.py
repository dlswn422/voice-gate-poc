from __future__ import annotations

import re
import time
import uuid

from src.engine.intent_logger import log_dialog, log_intent
from src.nlu.dialog_llm_client import dialog_llm_chat
from src.nlu.intent_schema import Intent
from src.nlu.llm_client import detect_intent_llm


CONFIDENCE_THRESHOLD = 0.75
SITE_ID = "parkassist_local"

IDLE_TIMEOUT_SEC = 15.0
DONE_COOLDOWN_SEC = 1.2

SECOND_STAGE_HARD_TURN_LIMIT = 6


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

FAREWELL_TEXT = "네, 해결되셨다니 다행입니다. 이용해 주셔서 감사합니다. 안녕히 가세요."

DONE_KEYWORDS = [
    "됐어요", "되었습니다", "해결", "괜찮아요",
    "그만", "종료", "끝", "마칠게",
    "고마워", "감사", "안녕",
]


def _normalize(text: str) -> str:
    return re.sub(r"[\s\.\,\!\?]+", "", (text or "").strip().lower())


def _is_done_utterance(text: str) -> bool:
    """
    ✅ '안됐어요'가 DONE으로 오인되는 문제 차단
    """
    t = _normalize(text)
    neg_prefix = ("안", "못", "미")
    for kw in DONE_KEYWORDS:
        k = _normalize(kw)
        idx = t.find(k)
        if idx == -1:
            continue
        if idx > 0 and t[idx - 1] in neg_prefix:
            continue
        if t.startswith("안" + k) or t.startswith("못" + k) or t.startswith("미" + k):
            continue
        return True
    return False


class AppEngine:
    """
    - FIRST_STAGE: 1차 의도 분류 + 원턴 질문형 응답(가볍게)
    - SECOND_STAGE: 슬롯/체크리스트 기반으로 재질문 → 충분하면 SOLVE → follow-up
      * 2차에서는 intent 전환도 허용(명확히 다른 문제로 넘어가면)
      * 6턴 초과면 관리자 호출 + 종료
    """

    def __init__(self):
        self.state = "FIRST_STAGE"

        self.session_id = None
        self.intent_log_id = None

        self.first_intent: str | None = None
        self.current_intent: str | None = None

        self.dialog_turn_index = 0
        self.dialog_history = []

        self._none_retry_count = 0
        self._ignore_until_ts = 0.0
        self._last_activity_ts = 0.0
        self._last_handled_utterance_id = None

        self._just_one_turn = False

        # ✅ 2차 상태
        self.second_turn_user = 0
        self.second_slots: dict = {}
        self.second_phase: str = "CLARIFY"  # CLARIFY | SOLVED

    def _start_new_session(self):
        self.session_id = str(uuid.uuid4())
        self.state = "FIRST_STAGE"

        self.intent_log_id = None
        self.first_intent = None
        self.current_intent = None

        self.dialog_turn_index = 0
        self.dialog_history = []

        self._none_retry_count = 0
        self._just_one_turn = False
        self._last_activity_ts = time.time()
        self._last_handled_utterance_id = None

        # 2차 초기화
        self.second_turn_user = 0
        self.second_slots = {}
        self.second_phase = "CLARIFY"

        print(f"[ENGINE] 🆕 New session started: {self.session_id}")

    def end_session(self, reason: str = ""):
        print(f"[ENGINE] 🛑 Session ended ({reason}): {self.session_id}")

        self.session_id = None
        self.state = "FIRST_STAGE"

        self.intent_log_id = None
        self.first_intent = None
        self.current_intent = None

        self.dialog_turn_index = 0
        self.dialog_history = []

        self._none_retry_count = 0
        self._just_one_turn = False
        self._last_handled_utterance_id = None

        self.second_turn_user = 0
        self.second_slots = {}
        self.second_phase = "CLARIFY"

    def check_idle_timeout(self):
        if self.session_id and time.time() - self._last_activity_ts >= IDLE_TIMEOUT_SEC:
            self.end_session(reason="idle-timeout")

    def calculate_confidence(self, text: str, intent: Intent) -> float:
        score = 0.4
        KEYWORDS = {
            Intent.EXIT: ["출차", "나가", "차단기", "출구"],
            Intent.ENTRY: ["입차", "들어", "입구"],
            Intent.PAYMENT: ["결제", "요금", "정산", "승인"],
            Intent.REGISTRATION: ["등록", "방문", "번호판"],
            Intent.TIME_PRICE: ["시간", "요금", "무료", "할인"],
            Intent.FACILITY: ["기계", "고장", "이상", "먹통", "통신", "서버"],
            Intent.COMPLAINT: ["왜", "안돼", "짜증", "불만"],
        }
        hits = sum(1 for k in KEYWORDS.get(intent, []) if k in text)
        score += 0.35 if hits else 0.15
        score += 0.05 if len(text) <= 4 else 0.2
        return round(min(score, 1.0), 2)

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

    def _enter_second_stage(self):
        self.state = "SECOND_STAGE"
        self.second_turn_user = 0
        self.second_slots = {}
        self.second_phase = "CLARIFY"

    def _maybe_switch_intent(self, new_intent: str | None):
        """
        ✅ 2차에서 intent 전환 허용:
        - new_intent가 명확하면 current_intent를 교체하고 슬롯/턴 리셋
        """
        if not new_intent:
            return
        if not isinstance(new_intent, str):
            return
        if new_intent == "NONE":
            return
        if self.current_intent == new_intent:
            return

        print(f"[ENGINE] 🔀 intent switch: {self.current_intent} -> {new_intent}")
        self.current_intent = new_intent
        self.second_turn_user = 0
        self.second_slots = {}
        self.second_phase = "CLARIFY"

    def _handle_second_stage(self, text: str):
        # 사용자 종료
        if _is_done_utterance(text):
            self._log_dialog("user", text)
            self._log_dialog("assistant", FAREWELL_TEXT, model="system")
            print(f"[DIALOG] {FAREWELL_TEXT}")
            self.end_session(reason="done")
            self._ignore_until_ts = time.time() + DONE_COOLDOWN_SEC
            return

        self._log_dialog("user", text)

        # user turn count 증가(2차로 들어온 사용자 발화 기준)
        # * 첫 질문을 2차가 던지기 위해 turn_count_user=0에서 시작
        # * 지금은 "사용자 발화가 들어올 때" 증가시키는 방식
        ctx_turn = self.second_turn_user

        res = dialog_llm_chat(
            text,
            history=self.dialog_history,
            context={
                "session_id": self.session_id,
                "first_intent": self.first_intent,
                "current_intent": self.current_intent,
                "turn_count_user": ctx_turn,
                "hard_turn_limit": SECOND_STAGE_HARD_TURN_LIMIT,
                "slots": self.second_slots,
                "phase": self.second_phase,
            },
            debug=True,
        )

        # 반영(슬롯/intent/phase)
        if getattr(res, "slots", None):
            if isinstance(res.slots, dict):
                self.second_slots.update(res.slots)

        self._maybe_switch_intent(getattr(res, "new_intent", None))

        action = getattr(res, "action", "ASK") or "ASK"
        reply = getattr(res, "reply", "") or "조금 더 자세히 말씀해 주실 수 있을까요?"

        self._log_dialog("assistant", reply, model="llama-3.1-8b")
        print(f"[DIALOG] {reply}")

        if action in ("DONE", "ESCALATE_DONE"):
            self.end_session(reason="escalate_done" if action == "ESCALATE_DONE" else "done")
            self._ignore_until_ts = time.time() + DONE_COOLDOWN_SEC
            return

        # ✅ 여기서 사용자 턴 증가
        self.second_turn_user += 1

        if action == "SOLVE":
            self.second_phase = "SOLVED"

    def handle_text(self, text: str, *, utterance_id: str | None = None):
        now = time.time()

        if not text or not text.strip():
            return
        if now < self._ignore_until_ts:
            return

        self._last_activity_ts = now

        if utterance_id and utterance_id == self._last_handled_utterance_id:
            print("[ENGINE] ⚠️ duplicated utterance ignored")
            return
        self._last_handled_utterance_id = utterance_id

        if not self.session_id:
            self._start_new_session()

        print("=" * 50)
        print(f"[ENGINE] State={self.state}")
        print(f"[ENGINE] Text={text}")

        # 원턴 직후 -> 2차 진입
        if self._just_one_turn:
            print("[ENGINE] 🔁 one-turn follow-up → SECOND_STAGE")
            self._just_one_turn = False
            if self.state != "SECOND_STAGE":
                self._enter_second_stage()
            self._handle_second_stage(text)
            return

        # FIRST_STAGE
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

            # 불만/저신뢰는 바로 2차
            if result.intent == Intent.COMPLAINT or result.confidence < CONFIDENCE_THRESHOLD:
                self._enter_second_stage()
                self._handle_second_stage(text)
                return

            # 일반: 원턴 질문형 응답 → 다음 발화부터 2차
            reply = ONE_TURN_RESPONSES.get(result.intent)
            self._log_dialog("assistant", reply, model="system")
            print(f"[ONE-TURN] {reply}")
            self._just_one_turn = True
            return

        # SECOND_STAGE
        if self.state == "SECOND_STAGE":
            self._handle_second_stage(text)
            return
