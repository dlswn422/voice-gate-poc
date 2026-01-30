from nlu.llm_client import detect_intent_llm
from nlu.intent_schema import Intent
from engine.intent_logger import log_intent, log_dialog  # ✅ dialog_logs 적재
from nlu.dialog_llm_client import dialog_llm_chat        # ✅ 2차 LLM(+RAG) 호출

import uuid
import time
import re


# --------------------------------------------------
# 정책 설정
# --------------------------------------------------
CONFIDENCE_THRESHOLD = 0.75
SITE_ID = "parkassist_local"

# ✅ (추가) DONE 강제 종료 키워드(2차에서 우선 적용)
DONE_KEYWORDS = [
    "됐어요", "되었습니다", "해결", "괜찮아요", "그만", "종료", "끝", "마칠게", "고마워", "감사", "안녕"
]

# ✅ (추가) DONE 시 배웅 멘트 고정
FAREWELL_TEXT = "네, 해결되셨다니 다행입니다. 이용해 주셔서 감사합니다. 안녕히 가세요."

# ✅ (추가) DONE 직후 잔향/중복 STT를 무시하기 위한 쿨다운(초)
DONE_COOLDOWN_SEC = 1.2


def _normalize(text: str) -> str:
    # 공백/구두점 제거해서 키워드 판정 안정화
    t = text.strip().lower()
    t = re.sub(r"[\s\.\,\!\?\u3002\uFF0E\uFF0C\uFF01\uFF1F]+", "", t)
    return t


def _is_done_utterance(text: str) -> bool:
    t = _normalize(text)
    return any(k.replace(" ", "") in t for k in DONE_KEYWORDS)


class AppEngine:
    """
    주차장 키오스크 CX용 App Engine

    상태:
    - FIRST_STAGE  : 1차 의도 분류 단계
    - SECOND_STAGE : 2차 상담(라마) 단계
    """

    def __init__(self):
        self.state = "FIRST_STAGE"
        self.session_id = None

        # ✅ (추가) 2차 로그/세션 추적용
        self.intent_log_id = None
        self.dialog_turn_index = 0
        self.dialog_history = []   # ✅ (추가) 멀티턴 전달용(선택)

        # ✅ (추가) DONE 직후 쿨다운
        self._ignore_until_ts = 0.0

    # ==================================================
    # 🔧 confidence 계산 로직
    # ==================================================
    def calculate_confidence(self, text: str, intent: Intent) -> float:
        score = 0.0
        text = text.strip()

        # ✅ (수정) Intent enum이 바뀌어도 안전하도록 name 기반으로 매핑
        intent_name = getattr(intent, "name", str(intent))

        KEYWORDS_BY_INTENT_NAME = {
            "EXIT_FLOW_ISSUE": ["출차", "나가", "차단기", "안열려", "안 열려"],
            "ENTRY_FLOW_ISSUE": ["입차", "들어가", "차단기", "안열려", "안 열려"],
            "PAYMENT_ISSUE": ["결제", "요금", "카드", "정산", "승인"],
            "TIME_ISSUE": ["시간", "무료", "초과"],
            "PRICE_INQUIRY": ["얼마", "요금", "가격"],
            "HOW_TO_EXIT": ["어떻게", "출차", "나가"],
            "HOW_TO_REGISTER": ["등록", "어디", "방법"],
            # ✅ (추가) 너 로그처럼 HELP_REQUEST가 들어오는 경우를 대비(낮게 주고 2차로 넘기기 쉽게)
            "HELP_REQUEST": ["결제", "차단기", "출차", "입차", "등록", "오류", "안돼", "안 돼"],
        }

        hits = sum(1 for k in KEYWORDS_BY_INTENT_NAME.get(intent_name, []) if k in text)

        if hits >= 2:
            score += 0.45
        elif hits == 1:
            score += 0.30
        else:
            score += 0.10

        if len(text) < 3:
            score += 0.05
        elif any(f in text for f in ["어", "음", "..."]):
            score += 0.10
        else:
            score += 0.25

        INTENT_RISK_WEIGHT_BY_NAME = {
            "HOW_TO_EXIT": 1.0,
            "PRICE_INQUIRY": 1.0,
            "TIME_ISSUE": 0.9,
            "EXIT_FLOW_ISSUE": 0.7,
            "ENTRY_FLOW_ISSUE": 0.7,
            "PAYMENT_ISSUE": 0.7,
            "REGISTRATION_ISSUE": 0.6,
            "COMPLAINT": 0.5,
            "HELP_REQUEST": 0.7,
        }

        score *= INTENT_RISK_WEIGHT_BY_NAME.get(intent_name, 0.6)
        return round(min(score, 1.0), 2)

    # ==================================================
    # ✅ (추가) 2차 처리(로그 + LLM + DONE 강제 + 배웅 고정)
    # ==================================================
    def _handle_second_stage(self, text: str):
        # ✅ (추가) DONE 직후 중복 STT 무시
        if time.time() < self._ignore_until_ts:
            return

        try:
            # ✅ (추가) DONE 키워드면 LLM 호출 없이 강제 종료 + 배웅 멘트 고정
            if _is_done_utterance(text):
                self.dialog_turn_index += 1
                log_dialog(
                    intent_log_id=self.intent_log_id,
                    session_id=self.session_id,
                    role="user",
                    content=text,
                    model="stt",
                    turn_index=self.dialog_turn_index,
                )

                self.dialog_turn_index += 1
                log_dialog(
                    intent_log_id=self.intent_log_id,
                    session_id=self.session_id,
                    role="assistant",
                    content=FAREWELL_TEXT,
                    model="system",
                    turn_index=self.dialog_turn_index,
                )

                print(f"[DIALOG] {FAREWELL_TEXT}")
                self.end_second_stage()
                self._ignore_until_ts = time.time() + DONE_COOLDOWN_SEC
                return

            # 사용자 발화 로그
            self.dialog_turn_index += 1
            log_dialog(
                intent_log_id=self.intent_log_id,
                session_id=self.session_id,
                role="user",
                content=text,
                model="stt",
                turn_index=self.dialog_turn_index,
            )

            # ✅ (추가) 멀티턴 유지(선택)
            self.dialog_history.append({"role": "user", "content": text})

            # ✅ 2차 LLM(+RAG) 호출
            res = dialog_llm_chat(
                text,
                history=self.dialog_history,
                context={"session_id": self.session_id, "intent_log_id": self.intent_log_id},
                debug=True,
            )

            llama_reply = getattr(res, "reply", "") or ""
            action = getattr(res, "action", None)  # "ASK|SOLVE|DONE|..."

            # ✅ (추가) 모델이 DONE을 주면 배웅 멘트로 고정 후 종료
            if action == "DONE":
                llama_reply = FAREWELL_TEXT

            # 어시스턴트 로그
            self.dialog_turn_index += 1
            log_dialog(
                intent_log_id=self.intent_log_id,
                session_id=self.session_id,
                role="assistant",
                content=llama_reply,
                model="llama-3.1-8b",
                turn_index=self.dialog_turn_index,
            )

            self.dialog_history.append({"role": "assistant", "content": llama_reply})

            print(f"[DIALOG] {llama_reply}")

            if action == "DONE":
                self.end_second_stage()
                self._ignore_until_ts = time.time() + DONE_COOLDOWN_SEC

        except Exception as e:
            # ✅ STT 콜백이 죽지 않게 여기서 잡아먹음
            print(f"[ENGINE] 2nd-stage failed: {repr(e)}")

    # ==================================================
    # 🎙️ STT 텍스트 처리 엔트리포인트
    # ==================================================
    def handle_text(self, text: str):
        if not text or not text.strip():
            return

        # ✅ (추가) DONE 직후 중복 STT 무시
        if time.time() < self._ignore_until_ts:
            return

        print("=" * 50)
        print(f"[ENGINE] State={self.state}")
        print(f"[ENGINE] Text={text}")

        # ==================================================
        # 🟢 2차 상담 단계
        # ==================================================
        if self.state == "SECOND_STAGE":
            self._handle_second_stage(text)
            print("=" * 50)
            return

        # ==================================================
        # 🔵 1차 의도 분류 단계
        # ==================================================
        try:
            result = detect_intent_llm(text)
        except Exception as e:
            print("[ENGINE] LLM inference failed:", e)
            print("=" * 50)
            return

        result.confidence = self.calculate_confidence(text=text, intent=result.intent)

        print(f"[ENGINE] Intent={result.intent.name}, confidence={result.confidence:.2f}")

        # ✅ 1차 로그 적재 + PK 받아서 2차 dialog_logs FK로 사용
        self.intent_log_id = log_intent(
            utterance=text,
            predicted_intent=result.intent.value,
            predicted_confidence=result.confidence,
            source="kiosk",
            site_id=SITE_ID,
        )
        print(f"[ENGINE] intent_log_id={self.intent_log_id}")

        # intent_log_id가 None이면 dialog_logs NOT NULL 깨지므로 2차 자체를 스킵
        if self.intent_log_id is None:
            print("[ENGINE] intent_log_id is None → skip llama fallback")
            print("=" * 50)
            return

        if result.intent == Intent.NONE:
            print("[ENGINE] Decision: irrelevant utterance")
            print("=" * 50)
            return

        # ==================================================
        # confidence 기준 이하 → 2차(라마 + 로그)
        # ==================================================
        if result.confidence < CONFIDENCE_THRESHOLD:
            print("[ENGINE] Decision: low confidence → llama fallback")

            self.state = "SECOND_STAGE"
            self.session_id = str(uuid.uuid4())   # ✅ 요구사항: session_id 고유 생성
            self.dialog_turn_index = 0
            self.dialog_history = []

            print(f"[ENGINE] Session started: {self.session_id}")
            print("[ENGINE] Llama will handle this utterance (logging dialog)")

            # ✅ (수정) 재귀(handle_text 재호출) 금지 → 바로 2차 처리
            self._handle_second_stage(text)

            print("=" * 50)
            return

        print("[ENGINE] Decision: passed 1st-stage classification")
        print("[ENGINE] Action: defer execution to next stage")
        print("=" * 50)

    # ==================================================
    # 🔚 상담 종료 시 호출
    # ==================================================
    def end_second_stage(self):
        print(f"[ENGINE] Session ended: {self.session_id}")
        self.state = "FIRST_STAGE"
        self.session_id = None
        self.intent_log_id = None
        self.dialog_turn_index = 0
        self.dialog_history = []
