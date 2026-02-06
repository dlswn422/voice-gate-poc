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

SECOND_STAGE_HARD_TURN_LIMIT = 6


# ==================================================
# 원턴 응답 (현상 질문)
# ==================================================
ONE_TURN_RESPONSES = {
    Intent.EXIT: "출차하려면 요금 정산이 완료되어야 차단기가 열립니다. 현재 어떤 문제가 발생했나요?",
    Intent.ENTRY: "입차 시 차량이 인식되면 차단기가 자동으로 열립니다. 현재 어떤 문제가 발생했나요?",
    Intent.PAYMENT: "주차 요금은 정산기나 출구에서 결제하실 수 있습니다. 현재 어떤 문제가 발생했나요?",
    Intent.REGISTRATION: "차량이나 방문자 등록은 키오스크에서 진행하실 수 있습니다. 현재 어떤 문제가 발생했나요?",
    Intent.TIME_PRICE: "주차 시간과 요금은 키오스크 화면에서 확인하실 수 있습니다. 현재 어떤 문제가 발생했나요?",
    Intent.FACILITY: "기기나 차단기에 이상이 있는 경우 관리실 도움을 받으실 수 있습니다. 현재 어떤 문제가 발생했나요?",
}

NONE_RETRY_TEXT = (
    "말씀을 정확히 이해하지 못했어요. "
    "출차, 결제, 등록 중 어떤 도움을 원하시는지 말씀해 주세요."
)

DONE_KEYWORDS = ["됐어요", "되었습니다", "해결", "괜찮아요", "그만", "종료", "끝", "마칠게", "고마워", "감사", "안녕"]
FAREWELL_TEXT = "네, 이용해 주셔서 감사합니다. 안녕히 가세요."


def _normalize(text: str) -> str:
    return re.sub(r"[\s\.\,\!\?]+", "", text.strip().lower())


def _is_done_utterance(text: str) -> bool:
    t = _normalize(text)
    return any(_normalize(k) in t for k in DONE_KEYWORDS)


class AppEngine:
    """
    요구 플로우:
    1) FIRST_STAGE: intent 분류
    2) intent별 원턴 질문(현상 질문) 출력
    3) 원턴 질문의 다음 사용자 답변을 symptom 슬롯에 "문장 통째로" 저장
    4) SECOND_STAGE: 남은 슬롯에 대해 LLM이 자율적으로 질문 생성
       - 사용자의 답변은 해당 슬롯에 "문장 통째로" 저장
    5) 슬롯이 다 차면 메뉴얼 기반 안내 문장 그대로 반환
    6) 사용자 6턴 초과: 관리자 호출 + 종료
    """

    def __init__(self):
        self.state = "FIRST_STAGE"

        self.session_id = None
        self.first_intent = None
        self.intent_log_id = None

        self.dialog_turn_index = 0
        self.dialog_history = []

        self._ignore_until_ts = 0.0
        self._last_activity_ts = 0.0
        self._last_handled_utterance_id = None

        self._just_one_turn = False

        # 2차 상태
        self.second_turn_count_user = 0
        self.second_slots = {}
        self.second_pending_slot = None

    def _start_new_session(self):
        self.session_id = str(uuid.uuid4())
        self.state = "FIRST_STAGE"

        self.first_intent = None
        self.intent_log_id = None
        self.dialog_turn_index = 0
        self.dialog_history = []

        self._just_one_turn = False

        self.second_turn_count_user = 0
        self.second_slots = {}
        self.second_pending_slot = None

        self._last_activity_ts = time.time()
        print(f"[ENGINE] 🆕 New session started: {self.session_id}")

    def end_session(self, reason: str = ""):
        print(f"[ENGINE] 🛑 Session ended ({reason}): {self.session_id}")

        self.session_id = None
        self.state = "FIRST_STAGE"
        self.first_intent = None
        self.intent_log_id = None
        self.dialog_turn_index = 0
        self.dialog_history = []

        self._just_one_turn = False
        self.second_turn_count_user = 0
        self.second_slots = {}
        self.second_pending_slot = None

        self._last_handled_utterance_id = None

    def check_idle_timeout(self):
        if self.session_id and time.time() - self._last_activity_ts >= IDLE_TIMEOUT_SEC:
            self.end_session(reason="idle-timeout")

    def calculate_confidence(self, text: str, intent: Intent) -> float:
        score = 0.4
        KEYWORDS = {
            Intent.EXIT: ["출차", "나가", "차단기"],
            Intent.ENTRY: ["입차", "들어가", "인식"],
            Intent.PAYMENT: ["결제", "요금", "정산", "승인"],
            Intent.REGISTRATION: ["등록", "방문", "할인", "차량"],
            Intent.TIME_PRICE: ["시간", "요금", "무료", "할인", "미적용"],
            Intent.FACILITY: ["기계", "고장", "먹통", "통신", "서버", "오류", "키오스크", "차단기"],
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

    def _handle_second_stage(self, text: str):
        if _is_done_utterance(text):
            self._log_dialog("user", text)
            self._log_dialog("assistant", FAREWELL_TEXT, model="system")
            print(f"[DIALOG] {FAREWELL_TEXT}")
            self.end_session(reason="done")
            self._ignore_until_ts = time.time() + DONE_COOLDOWN_SEC
            return

        self._log_dialog("user", text)

        res = dialog_llm_chat(
            text,
            history=self.dialog_history,
            context={
                "session_id": self.session_id,
                "intent": self.first_intent,  # ✅ 엔진의 현재 intent를 전달
                "turn_count_user": self.second_turn_count_user,
                "hard_turn_limit": SECOND_STAGE_HARD_TURN_LIMIT,
                "slots": self.second_slots,
                "pending_slot": self.second_pending_slot,
            },
            debug=True,
        )

        reply = getattr(res, "reply", "") or "조금 더 자세히 말씀해 주실 수 있을까요?"
        self._log_dialog("assistant", reply, model="llama-3.1-8b")
        print(f"[DIALOG] {reply}")

        # ✅ intent 갱신(complaint 재분류 / 내부 intent 전환을 엔진에 반영)
        new_intent = getattr(res, "new_intent", None)
        if new_intent and isinstance(new_intent, str) and new_intent.strip():
            if self.first_intent != new_intent.strip():
                self.first_intent = new_intent.strip()

        # 상태 갱신
        self.second_turn_count_user += 1
        if getattr(res, "slots", None) is not None:
            self.second_slots = res.slots
        self.second_pending_slot = getattr(res, "pending_slot", None)

        # ESCALATE_DONE / DONE면 종료
        if getattr(res, "action", "") in ("DONE", "ESCALATE_DONE"):
            self.end_session(reason=str(res.action).lower())
            self._ignore_until_ts = time.time() + DONE_COOLDOWN_SEC
            return

        # SOLVE는 "메뉴얼 안내 문장 그대로" 반환 정책
        # 세션 종료는 사용자의 종료 발화 또는 6턴 초과 정책으로 처리

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

        # ==================================================
        # 원턴 질문 직후 답변 -> symptom 슬롯에 raw 저장 후 2차 진입
        # ==================================================
        if self._just_one_turn:
            print("[ENGINE] 🔁 one-turn follow-up → SECOND_STAGE (symptom raw captured)")
            self.state = "SECOND_STAGE"
            self._just_one_turn = False

            if not isinstance(self.second_slots, dict):
                self.second_slots = {}

            # ✅ symptom은 "문장 통째로"
            self.second_slots["symptom"] = text.strip()

            self.second_pending_slot = None
            self.second_turn_count_user = 0

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
            self._log_dialog("user", text)

            if result.intent == Intent.NONE:
                self._log_dialog("assistant", NONE_RETRY_TEXT, model="system")
                print(f"[ONE-TURN] {NONE_RETRY_TEXT}")
                return

            # COMPLAINT/low confidence면 바로 2차
            if result.intent == Intent.COMPLAINT or result.confidence < CONFIDENCE_THRESHOLD:
                self.state = "SECOND_STAGE"
                self.second_turn_count_user = 0
                self.second_slots = {}
                self.second_pending_slot = None
                self._handle_second_stage(text)
                return

            # 일반 intent -> 원턴 질문(현상 질문)
            reply = ONE_TURN_RESPONSES.get(result.intent, "현재 어떤 문제가 발생했나요?")
            self._log_dialog("assistant", reply, model="system")
            print(f"[ONE-TURN] {reply}")

            # 다음 입력을 symptom으로 받을 준비
            self._just_one_turn = True
            return

        # --------------------------------------------------
        # SECOND_STAGE
        # --------------------------------------------------
        if self.state == "SECOND_STAGE":
            self._handle_second_stage(text)
            return
