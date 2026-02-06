from src.nlu.llm_client import detect_intent_llm
from src.nlu.intent_schema import Intent, INTENT_TO_DOCS # intent_schema로 옮겨질 매핑 정보 import
from src.engine.intent_logger import log_intent, log_dialog
from src.nlu.dialog_llm_client import dialog_llm_chat
from src.rag.manual_rag import ManualRAG

import uuid
import time
import re

# ==================================================
# 정책 설정
# ==================================================
SITE_ID = "parkassist_local"
IDLE_TIMEOUT_SEC = 15.0

# ==================================================
# 유틸
# ==================================================
def _norm_intent_name(x) -> str:
    if x is None: return "NONE"
    if isinstance(x, Intent): return x.name
    s = str(x).strip().upper()
    return s.replace("INTENT.", "").replace(" ", "_")

def _merge_slots(prev: dict, new: dict) -> dict:
    out = dict(prev or {})
    for k, v in (new or {}).items():
        if v and isinstance(v, str) and v.strip():
            out[k] = v
    return out

# ==================================================
# AppEngine 
# ==================================================
class AppEngine:
    """
    모든 발화는 [Intent분류 -> RAG검색 -> DialogLLM] 단일 파이프라인을 통과
    세션 유무(active)만 판단
    하드코딩된 답변/종료/재분류 로직 제거 -> LLM Action에 전적으로 위임
    """

    def __init__(self):
        self.session_id = None
        self.current_intent = None
        self.intent_log_id = None
        
        self.dialog_turn_index = 0
        self.dialog_history = []
        self.slots = {} 

        self._last_activity_ts = 0.0
        self._last_handled_utterance_id = None
        
        # RAG 엔진 초기화
        self.rag = ManualRAG()
        self._is_processing = False

    

    # --------------------------------------------------
    # 세션 관리
    # --------------------------------------------------
    def _ensure_session(self):
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
            self.dialog_turn_index = 0
            self.dialog_history = []
            self.slots = {}
            self.current_intent = None
            self.intent_log_id = None
            self._last_activity_ts = time.time()
            print(f"[ENGINE] 🆕 New session started: {self.session_id}")

    def end_session(self, reason: str = ""):
        print(f"[ENGINE] Session ended ({reason}): {self.session_id}")
        self.session_id = None
        self.current_intent = None
        self.slots = {}
        self.dialog_history = []
        self._last_handled_utterance_id = None

    def check_idle_timeout(self):
        if self.session_id and time.time() - self._last_activity_ts >= IDLE_TIMEOUT_SEC:
            self.end_session(reason="idle-timeout")

    # --------------------------------------------------
    # 로깅 헬퍼
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
    # 메인 파이프라인 (handle_text)
    # --------------------------------------------------
    def handle_text(self, text: str, *, utterance_id: str | None = None):
        now = time.time()

        if not text or not text.strip(): return
        if self._is_processing:
            return
        
        # 중복 발화 필터링
        if utterance_id and utterance_id == self._last_handled_utterance_id:
            return
        self._last_handled_utterance_id = utterance_id

        # 세션 활성화 및 타임스탬프 갱신
        self._ensure_session()
        self._last_activity_ts = now

        print("=" * 50)
        print(f"[ENGINE] Input: {text}")

        # 1차 의도 분류 
        # 세션에 이미 의도가 있어도, 사용자가 주제를 바꿨을 수 있으므로 매 턴 체크 권장
        # 여기서는 정확도를 위해 매 턴 분류 수행 
        intent_res = detect_intent_llm(text)
        
        # 로그 기록 
        self.intent_log_id = log_intent(
            utterance=text,
            predicted_intent=intent_res.intent.value,
            predicted_confidence=0.0, 
            source="kiosk",
            site_id=SITE_ID,
        )

        # 의도 업데이트 
        # 여기서는 '현재 발화의 의도'를 우선시하되, LLM에게 이전 의도도 context로 줌
        detected_intent_name = _norm_intent_name(intent_res.intent)
        if not self.current_intent or detected_intent_name != "NONE":
            self.current_intent = detected_intent_name
        
        print(f"[ENGINE] Intent: {self.current_intent}")

        #사용자 발화 로깅
        self._log_dialog("user", text)

        # RAG 검색 
        # 현재 의도와 매핑된 문서를 우선 검색하도록 가이드
        # INTENT_TO_DOCS는 intent_schema.py에서 가져옴
        preferred_docs = INTENT_TO_DOCS.get(self.current_intent, [])
        
        # 검색 수행
        retrieved_docs = self.rag.retrieve(text, preferred_docs=preferred_docs)
        
        # 검색 결과를 텍스트로 변환 (LLM 입력용)
        manual_context_str = "\n\n".join([f"[{d.doc_id}] {d.text}" for d in retrieved_docs])

        #통합 Dialog LLM 호출
        res = dialog_llm_chat(
            text,
            history=self.dialog_history,
            context={
                "slots": self.slots,
                "current_intent": self.current_intent
            },
            manual_context=manual_context_str, 
            debug=True
        )

        # 결과 처리 및 상태 업데이트
        self.slots = _merge_slots(self.slots, res.slots)
        
        # LLM이 의도 변경을 감지했다면 반영
        if res.new_intent and res.new_intent != "NONE":
            self.current_intent = res.new_intent

        # 답변 출력 및 로깅
        final_reply = res.reply
        self._log_dialog("assistant", final_reply, model="llama-3.1-8b")
        print(f"[DIALOG] {final_reply}")

        # 액션 수행 
        action = res.action.upper()
        if action == "DONE":
            self.end_session(reason="done_by_llm")
        elif action == "ESCALATE":
            # 관리자 소환이 필요 한 경우 이 코드를 수정
            print("관리자 호출중")
            self.end_session(reason="escalate_by_llm")
        
       