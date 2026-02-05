from __future__ import annotations

import os
import re
import time
import uuid
from typing import Dict, Any, Optional

from src.engine.intent_logger import log_intent, log_dialog
from src.nlu.intent_schema import Intent
from src.nlu.llm_client import detect_intent_llm
from src.nlu.dialog_llm_client import dialog_llm_chat


# ==================================================
# 정책 설정
# ==================================================
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75") or 0.75)
SITE_ID = os.getenv("SITE_ID", "parkassist_local")

IDLE_TIMEOUT_SEC = float(os.getenv("IDLE_TIMEOUT_SEC", "15.0") or 15.0)
DONE_COOLDOWN_SEC = float(os.getenv("DONE_COOLDOWN_SEC", "1.2") or 1.2)

SECOND_STAGE_HARD_TURN_LIMIT = int(os.getenv("SECOND_STAGE_HARD_TURN_LIMIT", "6") or 6)
DEBUG_DIALOG = os.getenv("DEBUG_DIALOG", "0").strip().lower() in ("1", "true", "yes")


# ==================================================
# 원턴 응답 (1차에서 질문형으로만)
# ==================================================
ONE_TURN_RESPONSES = {
    Intent.EXIT: "출차하려면 요금 정산이 완료되어야 차단기가 열립니다. 혹시 정산은 이미 하셨나요?",
    Intent.ENTRY: "입차 시 차량이 인식되면 차단기가 자동으로 열립니다. 차량이 인식되지 않았다면 잠시 정차해 주세요.",
    Intent.PAYMENT: "주차 요금은 정산기나 출구에서 결제하실 수 있습니다. 이미 결제를 진행하셨나요?",
    Intent.REGISTRATION: "차량/방문자 등록은 키오스크에서 진행합니다. 지금 등록 과정에서 문제가 있으신가요?",
    Intent.TIME_PRICE: "주차 시간/요금은 키오스크 화면에서 확인할 수 있어요. 무료/할인 적용 문제인가요, 아니면 요금 확인이 필요하신가요?",
    Intent.FACILITY: "기기나 차단기에 이상이 있는 경우가 있어요. 지금 어떤 증상이 나타나나요?",
}

NONE_RETRY_TEXT = (
    "말씀을 정확히 이해하지 못했어요. "
    "출차, 결제, 등록, 요금/시간, 기기 문제 중 어떤 도움을 원하시는지 말씀해 주세요."
)

FAREWELL_TEXT = "네, 이용해 주셔서 감사합니다. 안녕히 가세요."


# ==================================================
# 종료 발화(부정형 오인 방지)
# ==================================================
_DONE_HARD = {
    "종료", "끝", "그만", "마칠게", "이만", "끊을게",
    "됐어요", "됐어", "됐습니다", "해결", "해결됨", "해결됐", "정상", "문제없",
}
_DONE_SOFT = {"고마워", "감사", "안녕", "수고", "잘가", "바이"}


def _normalize(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[\s\.\,\!\?\u3002\uFF0E\uFF0C\uFF01\uFF1F]+", "", t)
    return t


def _is_done_utterance(text: str) -> bool:
    t = _normalize(text)
    if not t:
        return False

    if any(_normalize(k) in t for k in _DONE_SOFT):
        return True

    if "안됐" in t or "안되" in t or "안돼" in t:
        return False

    for kw in _DONE_HARD:
        k = _normalize(kw)
        if t == k or t.endswith(k):
            return True

    return False


# ==================================================
# 2차 슬롯/필수 슬롯 정의 (뼈대)
# ==================================================
REQUIRED_SLOTS_BY_INTENT = {
    "PAYMENT": ["where", "symptom"],
    "EXIT": ["where", "symptom"],
    "ENTRY": ["where", "symptom"],
    "REGISTRATION": ["where", "symptom"],
    "TIME_PRICE": ["symptom"],
    "FACILITY": ["where", "symptom"],
    "COMPLAINT": ["where", "symptom"],
    "NONE": ["where", "symptom"],
}


# ==================================================
# AppEngine
# ==================================================
class AppEngine:
    """
    - FIRST_STAGE: 1차 의도분류 + 원턴 질문형 응답
    - SECOND_STAGE: 2차 LLM이 슬롯을 채우며 질문(ASK) → 슬롯 충족 시 메뉴얼 기반 해결(SOLVE)
    - 6턴 초과: 관리자 호출 선언 + 세션 종료
    """

    def __init__(self):
        self.state = "FIRST_STAGE"
        self.session_id: Optional[str] = None
        self.intent_log_id: Optional[str] = None

        self.first_intent: Optional[str] = None
        self.current_intent: Optional[str] = None

        self.dialog_turn_index = 0
        self.dialog_history = []

        self._ignore_until_ts = 0.0
        self._last_activity_ts = 0.0
        self._last_handled_utterance_id = None

        self._just_one_turn = False

        # 2차 상태
        self.second_stage_user_turns = 0
        self.second_stage_slots: Dict[str, Any] = {}

    def _start_new_session(self):
        self.session_id = str(uuid.uuid4())
        self.state = "FIRST_STAGE"

        self.intent_log_id = None
        self.first_intent = None
        self.current_intent = None

        self.dialog_turn_index = 0
        self.dialog_history = []

        self._ignore_until_ts = 0.0
        self._last_activity_ts = time.time()
        self._last_handled_utterance_id = None
        self._just_one_turn = False

        self.second_stage_user_turns = 0
        self.second_stage_slots = {}

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

        self._ignore_until_ts = 0.0
        self._last_activity_ts = 0.0
        self._last_handled_utterance_id = None
        self._just_one_turn = False

        self.second_stage_user_turns = 0
        self.second_stage_slots = {}

    def check_idle_timeout(self):
        if self.session_id and time.time() - self._last_activity_ts >= IDLE_TIMEOUT_SEC:
            self.end_session(reason="idle-timeout")

    def calculate_confidence(self, text: str, intent: Intent) -> float:
        score = 0.4
        KEYWORDS = {
            Intent.EXIT: ["출차", "나가", "차단기", "출구"],
            Intent.ENTRY: ["입차", "들어가", "입구"],
            Intent.PAYMENT: ["결제", "요금", "정산", "승인"],
            Intent.REGISTRATION: ["등록", "방문", "차량", "번호판"],
            Intent.TIME_PRICE: ["시간", "요금", "무료", "할인", "감면"],
            Intent.FACILITY: ["기계", "고장", "이상", "먹통", "무반응", "오류"],
            Intent.COMPLAINT: ["왜", "안돼", "짜증", "화나", "불만"],
        }
        hits = sum(1 for k in KEYWORDS.get(intent, []) if k in text)
        score += 0.35 if hits else 0.15
        score += 0.05 if len(text) <= 4 else 0.2
        return round(min(score, 1.0), 2)

    def _log_dialog(self, role: str, content: str, model: str = "stt"):
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

    def _handle_second_stage(self, text: str):
        if _is_done_utterance(text):
            self._log_dialog("user", text)
            self._log_dialog("assistant", FAREWELL_TEXT, model="system")
            print(f"[DIALOG] {FAREWELL_TEXT}")
            self.end_session(reason="done")
            self._ignore_until_ts = time.time() + DONE_COOLDOWN_SEC
            return

        self._log_dialog("user", text)

        history_for_llm = self.dialog_history
        if history_for_llm and history_for_llm[-1]["role"] == "user" and history_for_llm[-1]["content"] == text:
            history_for_llm = history_for_llm[:-1]

        cur_int = (self.current_intent or self.first_intent or "NONE")
        req_slots = REQUIRED_SLOTS_BY_INTENT.get(cur_int, ["where", "symptom"])

        res = dialog_llm_chat(
            text,
            history=history_for_llm,
            context={
                "session_id": self.session_id,
                "intent_log_id": self.intent_log_id,
                "first_intent": self.first_intent,
                "current_intent": self.current_intent or self.first_intent,
                "required_slots": req_slots,
                "slots": self.second_stage_slots,
                "hard_turn_limit": SECOND_STAGE_HARD_TURN_LIMIT,
                "turn_count_user": self.second_stage_user_turns,
            },
            debug=DEBUG_DIALOG,
        )

        if isinstance(res.slots, dict):
            self.second_stage_slots = res.slots

        # ✅ 2차에서 의도 전환 허용
        if getattr(res, "new_intent", None):
            self.current_intent = res.new_intent
        else:
            self.current_intent = cur_int

        reply = getattr(res, "reply", "") or "조금 더 자세히 말씀해 주실 수 있을까요?"
        self._log_dialog("assistant", reply, model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
        print(f"[DIALOG] {reply}")

        self.second_stage_user_turns += 1

        if getattr(res, "action", "") in ("DONE", "ESCALATE_DONE"):
            self.end_session(reason=str(res.action).lower())
            self._ignore_until_ts = time.time() + DONE_COOLDOWN_SEC

    def handle_text(self, text: str, *, utterance_id: Optional[str] = None):
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

        # 원턴 직후 후속 발화 → SECOND_STAGE
        if self._just_one_turn:
            print("[ENGINE] 🔁 one-turn follow-up → SECOND_STAGE")
            self.state = "SECOND_STAGE"
            self._just_one_turn = False
            self.second_stage_user_turns = 0
            self.second_stage_slots = {}
            self.current_intent = self.first_intent
            self._handle_second_stage(text)
            return

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
            self.current_intent = self.first_intent
            self._log_dialog("user", text)

            if result.intent == Intent.NONE:
                self._log_dialog("assistant", NONE_RETRY_TEXT, model="system")
                print(f"[ONE-TURN] {NONE_RETRY_TEXT}")
                return

            if result.intent == Intent.COMPLAINT or result.confidence < CONFIDENCE_THRESHOLD:
                self.state = "SECOND_STAGE"
                self.second_stage_user_turns = 0
                self.second_stage_slots = {}
                self._handle_second_stage(text)
                return

            reply = ONE_TURN_RESPONSES.get(result.intent) or NONE_RETRY_TEXT
            self._log_dialog("assistant", reply, model="system")
            print(f"[ONE-TURN] {reply}")
            self._just_one_turn = True
            return

        if self.state == "SECOND_STAGE":
            self._handle_second_stage(text)
            return
