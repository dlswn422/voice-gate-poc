from __future__ import annotations

import os
import re
import json
import asyncio
import time
from collections import deque
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI

load_dotenv()

app = FastAPI()

# ----------------------------
# Single-session guard (optional)
# ----------------------------
_ACTIVE_SESSION_LOCK = asyncio.Lock()


def make_aoai_client() -> AzureOpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
    if not endpoint or not api_key or not api_version:
        raise RuntimeError("AZURE_OPENAI_* env missing")
    return AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)


AOAI = make_aoai_client()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "parking-llm").strip()

SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "").strip()
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "").strip()
SPEECH_LANG = os.getenv("AZURE_SPEECH_LANGUAGE", "ko-KR").strip()
SPEECH_VOICE = os.getenv("AZURE_SPEECH_VOICE", "ko-KR-SunHiNeural").strip()

if not SPEECH_KEY or not SPEECH_REGION:
    raise RuntimeError("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION env missing")


# =========================================================
# ✅ 개선된 시스템 프롬프트
# - 문맥 유지
# - 짧은 답/차량번호를 직전 질문의 답으로 해석
# - 인사/감사/종료 발화 자연 처리
# - 불완전 차량번호는 재질문
# - 카메라/표정 언급 금지
# =========================================================
SYSTEM_PROMPT_BASE = """
너는 주차장 고객상담 AI다. 사용자는 주차장 입구/출구/사전정산기/무인정산기/관리실 호출 상황에서 짧게 말할 수 있으며,
음성 인식(STT) 결과에는 오탈자, 띄어쓰기 오류, 문장부호 보정, 말끝 흐림, 불완전 발화가 포함될 수 있다.

[목표]
- 사용자가 주차장 관련 문제를 빠르게 해결하도록 돕는다.
- 답변은 짧고 명확하게 한다.
- 필요할 때만 질문 1개를 한다.
- 기본 문맥은 “지금 주차장 관련 상담 중”이다.

[상담 범위]
- 차량 출입
- 주차 등록
- 차량번호 확인
- 사전정산 / 출차정산 / 영수증
- 요금 문의
- 할인 적용
- 정기권 / 방문 등록
- 차단기 / 정산기 / 호출벨 / 안내방송 / 소음 / 기기 오류
- 시설 고장 / 출차 불가 / 입차 불가

[말투/길이]
- 항상 한국어 존댓말.
- 기본적으로 1~2문장으로 답한다.
- 꼭 필요할 때만 최대 3문장까지 허용한다.
- 장황한 설명 금지.
- 사과는 필요할 때 1회만 짧게 한다.

[중요한 해석 규칙]
- 이 대화는 기본적으로 “주차장 상담”이다.
- 사용자의 짧은 답, 숫자, 차량번호 조각, 명사구는 직전 질문에 대한 답으로 우선 해석한다.
- 사용자가 이미 주차장 문맥 안에서 답하고 있는데도 “주차장 문의인지 확인해주세요” 같은 범위 확인 질문을 반복하지 않는다.
- 문장부호(?, !)는 STT 자동 보정일 수 있으므로 의도 해석에 과도하게 반영하지 않는다.
- “소음”, “삐 소리”, “경고음” 등은 병원/의료가 아니라 차단기/기기/안내방송 등 주차장 상황으로 우선 해석한다.
- 불완전한 문장이라도 주차장 문맥에서 의미를 최대한 복원한다.

[대화 문맥 유지 규칙]
- 직전에 네가 질문했고, 사용자가 짧게 답하면 그 답을 현재 질문의 답으로 우선 처리한다.
- 이미 진행 중인 문맥이 있으면 새 문의로 초기화하지 않는다.
- 단, 사용자가 명확히 다른 주제로 전환하면 그때만 새 문의로 해석한다.

[인사/감사/종료 발화 처리]
- 아래와 같은 사회적 발화는 주차 문의 여부를 다시 묻지 말고 짧게 자연스럽게 답한다.
- 예:
  - “안녕하세요” → “안녕하세요. 무엇을 도와드릴까요?”
  - “감사합니다” → “네, 감사합니다.”
  - “고마워요” → “네, 도움이 되어 다행입니다.”
  - “안녕히 계세요” / “수고하세요” → “네, 좋은 하루 되세요.”
- 이런 발화에 “주차 문의인지 확인해주세요”라고 답하지 않는다.

[차량번호 처리 규칙]
- 차량번호가 들어오면 현재 문맥상 차량번호를 요구하던 상황인지 우선 확인하고, 그렇다면 답변으로 처리한다.
- 차량번호가 완전하지 않거나 불분명하면 추측해서 확정하지 않는다.
- 말끝이 흐리거나 일부만 들린 경우에는 반드시 짧게 다시 요청한다.
- 예:
  - “12가 어버버…”
  - “차량번호를 정확히 이해하지 못했습니다. 다시 한 번만 말씀해주세요.”
- 차량번호처럼 들리지만 확신이 낮으면 역시 재질문한다.

[불완전 발화 / 무응답 대응]
- 사용자의 발화가 너무 짧거나 끊겼더라도 침묵하지 말고 반드시 응답한다.
- 이해가 어려우면 아래처럼 짧게 재질문한다.
  - “이해를 잘 못했습니다. 다시 한 번만 말씀해주세요.”
  - “차량번호를 다시 또렷하게 말씀해주세요.”
  - “지금 어떤 문제가 있는지 한 번만 더 말씀해주세요.”
- 응답 불가 상태처럼 보이더라도 빈 응답을 하지 않는다.

[질문 정책]
- 정보가 부족하면 질문은 한 번에 1개만 한다.
- 질문 우선순위:
  1) 지금 단계 확인(입차/출차/정산/등록 중 무엇인지)
  2) 차량번호
  3) 기기 화면 문구 또는 오류 내용
  4) 결제 여부 / 할인 여부
- 이미 질문했고 사용자가 답했다면, 같은 질문을 반복하지 말고 다음 단계 안내로 이어간다.

[범위 확인 규칙]
- “주차장 문의인지 확인해주세요” 같은 범위 확인은 정말로 무관한 주제일 때만 사용한다.
- 인사말, 감사 표현, 짧은 대답, 차량번호, 숫자열, 오류 문구, 단답형 응답에는 범위 확인을 하지 않는다.
- 아래처럼 주차장과 무관함이 명백할 때만 1문장으로 확인한다:
  “주차장 이용 관련 문의 맞으실까요? 어떤 문제가 있으신가요?”

[오류/고장 대응]
- “안 돼요”, “먹통이에요”, “차단기가 안 열려요”, “결제가 안 돼요”처럼 모호해도 바로 범위 확인부터 하지 않는다.
- 현재 문맥 또는 주차장 상황으로 우선 해석하고, 필요한 질문 1개만 한다.

[안전 규칙]
- 사용자의 감정/상태를 단정하거나 진단하지 마라.
- 내부 신호가 있더라도 사용자에게 언급하지 마라.
- 카메라, 얼굴, 표정, 감정 감지 여부를 절대 언급하지 마라.

[출력 규칙]
- 답변만 출력한다.
- 불필요한 설명, 내부 판단, 분류 결과를 출력하지 않는다.
- 목록이 필요하면 1, 2 정도로 아주 짧게만 사용한다.
""".strip()

NO_MATCH_FALLBACK = "이해를 잘 못했습니다. 다시 한 번만 말씀해주세요."
MAX_HISTORY_MESSAGES = 12


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _normalize_match_text(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[\s\.\,\!\?\~…]+", "", s)
    return s


def try_get_social_reply(text: str) -> str:
    norm = _normalize_match_text(text)

    greetings = {
        "안녕하세요", "안녕", "여보세요", "네안녕하세요"
    }
    thanks = {
        "감사합니다", "고맙습니다", "고마워요", "고마워", "감사해요"
    }
    closings = {
        "안녕히계세요", "수고하세요", "안녕히가세요", "들어가세요", "바이", "bye"
    }

    if norm in greetings:
        return "안녕하세요. 무엇을 도와드릴까요?"
    if norm in thanks:
        return "네, 감사합니다."
    if norm in closings:
        return "네, 좋은 하루 되세요."
    return ""


def extract_full_plate(text: str) -> str:
    compact = re.sub(r"\s+", "", text or "")
    m = re.search(r"(?<!\d)(\d{2,3}[가-힣]\d{4})(?!\d)", compact)
    return m.group(1) if m else ""


def is_incomplete_plate_like(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")

    if extract_full_plate(compact):
        return False

    # 예: 12가 / 12가1 / 12가12 / 12가123 / 123가 / 123가1 ...
    if re.search(r"(?<!\d)\d{1,3}[가-힣]\d{0,3}(?!\d)", compact):
        return True

    # 예: "12가 어버버" 같은 경우 일부라도 차량번호 형태면 재질문 유도
    if re.search(r"\d{1,3}[가-힣]", compact):
        return True

    return False


def infer_task_from_text(text: str) -> str | None:
    norm = _normalize_match_text(text)

    if not norm:
        return None

    if any(k in norm for k in ["주차등록", "차량등록", "방문등록", "정기권등록", "등록"]):
        return "registration"

    if any(k in norm for k in ["결제", "정산", "요금", "사전정산", "출차정산", "영수증"]):
        return "payment"

    if any(k in norm for k in ["차단기", "입차", "출차", "안열", "안열려", "출차가안", "입차가안"]):
        return "barrier"

    if any(k in norm for k in ["고장", "오류", "먹통", "정산기", "호출벨", "인터폰", "기기", "안내방송", "소음"]):
        return "facility"

    return None


def looks_like_question(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False

    question_markers = [
        "알려주세요", "말씀해주세요", "말씀해 주세요", "어떤", "무엇", "맞으실까요",
        "인가요", "하셨나요", "되셨나요", "보이시나요", "필요하신가요", "있으신가요",
        "시도하셨나요", "확인해 주세요", "가능하실까요", "겠어요"
    ]
    return s.endswith("?") or any(m in s for m in question_markers)


def build_dialog_context(dialog_state: dict) -> str:
    current_task = dialog_state.get("current_task") or "unknown"
    pending_slot = dialog_state.get("pending_slot") or "none"
    last_question = dialog_state.get("last_question") or "none"
    conversation_phase = dialog_state.get("conversation_phase") or "opening"

    return f"""
[현재 상담 상태]
- current_task: {current_task}
- pending_slot: {pending_slot}
- last_question: {last_question}
- conversation_phase: {conversation_phase}

[문맥 유지 추가 규칙]
- 현재 사용자 발화는 최근 대화와 현재 상담 상태를 함께 보고 해석한다.
- pending_slot이 car_number면, 짧은 숫자/문자열도 차량번호 답변 후보로 우선 해석한다.
- 직전에 차량번호를 요청한 상태에서 번호가 불완전하면 반드시 다시 또렷하게 말씀해달라고 답한다.
- 이미 주차장 상담 문맥이면 범위 확인 질문을 반복하지 않는다.
""".strip()


def build_expression_policy(vision: dict | None) -> str:
    """
    Expression-only policy for LLM (tone/strategy only).
    3-tier confidence:
      - conf < 0.25: ignore
      - 0.25~0.45: weak (tone only)
      - >=0.45: strong (tone + strategy)
    """
    if not isinstance(vision, dict):
        return ""

    expr = None
    if vision.get("type") == "vision_expression":
        expr = vision.get("expression") or {}
    elif isinstance(vision.get("expression"), dict):
        expr = vision.get("expression") or {}

    if not isinstance(expr, dict) or not expr:
        return ""

    label = str(expr.get("label", "neutral"))
    conf = _safe_float(expr.get("confidence", 0))
    valence = _safe_float(expr.get("valence", 0))
    arousal = _safe_float(expr.get("arousal", 0))

    # low => ignore
    if conf < 0.25:
        return (
            "\n[표정 신호(추정)] confidence가 낮다. 표정 정보는 무시하고 기본 톤으로 답하라.\n"
        )

    # mid => tone only
    if conf < 0.45:
        return (
            "\n[표정 신호(추정, 약한 참고)] "
            f"label={label}, conf={conf:.3f}, valence={valence:.3f}, arousal={arousal:.3f}.\n"
            "규칙: 오차 가능성이 크다. 사실로 단정하지 말고 말투만 약하게 조정하라.\n"
            "- 더 짧게, 더 명확하게.\n"
            "- 확인 질문은 최대 1개.\n"
        )

    # high => strong strategy
    if label in ("angry", "frustrated"):
        return (
            "\n[표정 신호(추정, 강한 참고)] "
            f"label={label}, conf={conf:.3f}, valence={valence:.3f}, arousal={arousal:.3f}.\n"
            "규칙: 절대 단정/진단하지 말고, 대화 전략만 조정하라.\n"
            "전략:\n"
            "1) 사과는 1회만 짧게.\n"
            "2) 해결 절차를 바로 제시(가장 빠른 조치).\n"
            "3) 확인 질문은 딱 1개만.\n"
            "4) 장황한 설명 금지.\n"
        )
    if label == "confused":
        return (
            "\n[표정 신호(추정, 강한 참고)] "
            f"label={label}, conf={conf:.3f}, valence={valence:.3f}, arousal={arousal:.3f}.\n"
            "규칙: 절대 단정/진단하지 말고, 대화 전략만 조정하라.\n"
            "전략:\n"
            "1) 한 단계씩 쉽게 안내.\n"
            "2) 확인 질문은 1개만.\n"
        )
    if label == "positive":
        return (
            "\n[표정 신호(추정, 강한 참고)] "
            f"label={label}, conf={conf:.3f}, valence={valence:.3f}, arousal={arousal:.3f}.\n"
            "규칙: 절대 단정/진단하지 말고, 친절하지만 짧게 안내하라.\n"
        )

    return (
        "\n[표정 신호(추정, 강한 참고)] "
        f"label={label}, conf={conf:.3f}, valence={valence:.3f}, arousal={arousal:.3f}.\n"
        "규칙: 절대 단정/진단하지 말고 기본 톤으로 답하라.\n"
    )


def prepare_state_for_current_turn(dialog_state: dict, user_text: str) -> dict:
    state = dict(dialog_state)
    guessed_task = infer_task_from_text(user_text)
    if guessed_task:
        state["current_task"] = guessed_task
    return state


def should_expect_car_number(dialog_state: dict) -> bool:
    if dialog_state.get("pending_slot") == "car_number":
        return True
    last_question = dialog_state.get("last_question") or ""
    return "차량번호" in last_question


def maybe_handle_fastpath(user_text: str, dialog_state: dict) -> str:
    # 1) 인사/감사/종료 발화는 deterministic 처리
    social = try_get_social_reply(user_text)
    if social:
        return social

    # 2) 차량번호를 기다리는 상황이면 deterministic 보강
    if should_expect_car_number(dialog_state):
        plate = extract_full_plate(user_text)
        if plate:
            task = dialog_state.get("current_task")
            if task == "registration":
                return f"네, 차량번호 {plate}로 확인했습니다. 등록을 도와드릴게요."
            if task == "payment":
                return f"네, 차량번호 {plate}로 확인했습니다. 요금 확인을 도와드릴게요."
            return f"네, 차량번호 {plate}로 확인했습니다. 계속 도와드릴게요."

        if is_incomplete_plate_like(user_text):
            return "차량번호를 정확히 이해하지 못했습니다. 다시 한 번만 말씀해주세요."

    return ""


def update_dialog_state(dialog_state: dict, user_text: str, bot_text: str) -> None:
    user_task = infer_task_from_text(user_text)
    norm_user = _normalize_match_text(user_text)
    bot_text = (bot_text or "").strip()

    if user_task:
        dialog_state["current_task"] = user_task

    if norm_user in {"안녕하세요", "안녕", "여보세요", "네안녕하세요"}:
        dialog_state["conversation_phase"] = "opening"
    elif norm_user in {"감사합니다", "고맙습니다", "고마워요", "고마워", "감사해요", "안녕히계세요", "수고하세요"}:
        dialog_state["conversation_phase"] = "closing"
    elif looks_like_question(bot_text):
        dialog_state["conversation_phase"] = "collecting_info"
    else:
        dialog_state["conversation_phase"] = "guiding"

    # assistant가 차량번호를 요청하는 경우
    if "차량번호" in bot_text and any(k in bot_text for k in ["알려주세요", "말씀해주세요", "말씀해 주세요", "다시", "부탁드립니다"]):
        dialog_state["pending_slot"] = "car_number"
        dialog_state["last_question"] = bot_text
        return

    # assistant가 차량번호를 확인 완료한 경우
    if "차량번호" in bot_text and ("확인했습니다" in bot_text or "확인됐" in bot_text or "확인했습니다." in bot_text):
        dialog_state["pending_slot"] = None

    # 기타 질문 슬롯
    if "결제 수단" in bot_text or "어떤 결제 수단" in bot_text:
        dialog_state["pending_slot"] = "payment_method"
    elif "문구" in bot_text or "오류 내용" in bot_text:
        dialog_state["pending_slot"] = "error_message"
    elif not looks_like_question(bot_text):
        # 명시적 질문이 아니면 슬롯 해제
        if dialog_state.get("pending_slot") != "car_number":
            dialog_state["pending_slot"] = None

    dialog_state["last_question"] = bot_text if looks_like_question(bot_text) else None


def llm_reply_sync(
    user_text: str,
    vision_state: dict | None,
    history_messages: list[dict],
    dialog_state: dict,
) -> str:
    """Blocking call (run via asyncio.to_thread)."""
    expr_policy = build_expression_policy(vision_state)
    context_prompt = build_dialog_context(dialog_state)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_BASE + "\n\n" + context_prompt + expr_policy},
    ]

    if history_messages:
        messages.extend(history_messages[-MAX_HISTORY_MESSAGES:])

    messages.append({"role": "user", "content": user_text})

    resp = AOAI.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        temperature=0.3,
        max_tokens=220,
    )
    return (resp.choices[0].message.content or "").strip()


def synth_wav_sync(text: str) -> bytes:
    """Blocking TTS (run via asyncio.to_thread). Returns WAV bytes."""
    try:
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
        speech_config.speech_synthesis_voice_name = SPEECH_VOICE
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
        )
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        r = synthesizer.speak_text_async(text).get()
        if r.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            return b""
        return r.audio_data
    except Exception as e:
        print(f"[TTS_ERROR] {e}")
        return b""


@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    # (선택) 단일 세션만 허용
    if _ACTIVE_SESSION_LOCK.locked():
        await ws.accept()
        await ws.send_text(json.dumps({"type": "error", "message": "이미 상담이 진행 중입니다. 잠시 후 다시 시도하세요."}, ensure_ascii=False))
        await ws.close(code=1008)
        return

    await _ACTIVE_SESSION_LOCK.acquire()
    await ws.accept()

    # ✅ Latest expression-only camera signal (per websocket session)
    last_vision_state: dict | None = None
    _last_expr_log_ts = 0.0

    # ✅ STT 애매 인식 fallback용 타임스탬프
    last_partial_ts = 0.0
    last_no_match_queue_ts = 0.0

    # ✅ 최근 대화 히스토리 / 상태
    chat_history = deque(maxlen=MAX_HISTORY_MESSAGES)
    dialog_state = {
        "current_task": None,
        "pending_slot": None,
        "last_question": None,
        "conversation_phase": "opening",
    }

    # Azure STT: 브라우저 PCM16(16k mono) 스트림 수신
    stream_format = speechsdk.audio.AudioStreamFormat(samples_per_second=16000, bits_per_sample=16, channels=1)
    push_stream = speechsdk.audio.PushAudioInputStream(stream_format)
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language = SPEECH_LANG
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    loop = asyncio.get_running_loop()
    final_queue: asyncio.Queue[dict] = asyncio.Queue()

    # ----------------------------
    # Turn cancel mechanism
    # ----------------------------
    turn_id = 0
    turn_lock = asyncio.Lock()

    async def send_json(payload: dict):
        try:
            await ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            pass

    def bump_turn_and_barge_in():
        """User started speaking -> cancel current turn + notify frontend."""
        nonlocal turn_id
        turn_id += 1
        loop.call_soon_threadsafe(lambda: asyncio.create_task(send_json({"type": "barge_in"})))

    def on_recognizing(evt):
        nonlocal last_partial_ts
        text = (evt.result.text or "").strip()
        if not text:
            return
        last_partial_ts = time.time()
        bump_turn_and_barge_in()
        loop.call_soon_threadsafe(lambda: asyncio.create_task(send_json({"type": "partial", "text": text})))

    def on_recognized(evt):
        nonlocal last_partial_ts, last_no_match_queue_ts

        reason = evt.result.reason

        if reason == speechsdk.ResultReason.RecognizedSpeech:
            text = (evt.result.text or "").strip()
            if not text:
                return

            loop.call_soon_threadsafe(
                final_queue.put_nowait,
                {"kind": "speech", "text": text, "turn_id": turn_id},
            )
            return

        # ✅ 애매하게 끊긴 발화도 무응답으로 끝내지 않기
        if reason == speechsdk.ResultReason.NoMatch:
            now = time.time()
            if (now - last_partial_ts) < 2.5 and (now - last_no_match_queue_ts) > 1.5:
                last_no_match_queue_ts = now
                loop.call_soon_threadsafe(
                    final_queue.put_nowait,
                    {"kind": "no_match", "turn_id": turn_id},
                )

    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.start_continuous_recognition()

    async def final_consumer():
        nonlocal turn_id, last_vision_state

        while True:
            item = await final_queue.get()
            kind = item.get("kind")
            my_turn = item.get("turn_id", turn_id)

            # --------------------------------
            # 1) STT no-match fallback
            # --------------------------------
            if kind == "no_match":
                async with turn_lock:
                    if my_turn != turn_id:
                        continue

                    bot = NO_MATCH_FALLBACK
                    await send_json({"type": "bot_text", "text": bot})

                    wav = await asyncio.to_thread(synth_wav_sync, bot)

                    if my_turn != turn_id:
                        continue

                    if wav:
                        try:
                            await ws.send_bytes(wav)
                        except Exception:
                            pass
                continue

            # --------------------------------
            # 2) normal recognized speech
            # --------------------------------
            user_text = (item.get("text") or "").strip()
            if not user_text:
                continue

            async with turn_lock:
                if my_turn != turn_id:
                    continue

                await send_json({"type": "final", "text": user_text})

                state_for_turn = prepare_state_for_current_turn(dialog_state, user_text)

                # ✅ deterministic fast-path
                bot = maybe_handle_fastpath(user_text, state_for_turn)

                # ✅ fallback to LLM with recent history + state
                if not bot:
                    try:
                        bot = await asyncio.to_thread(
                            llm_reply_sync,
                            user_text,
                            last_vision_state,
                            list(chat_history),
                            state_for_turn,
                        )
                    except Exception as e:
                        print(f"[LLM_ERROR] {e}")
                        bot = NO_MATCH_FALLBACK

                if not bot:
                    bot = NO_MATCH_FALLBACK

                if my_turn != turn_id:
                    continue

                # ✅ 다음 턴이 직전 문맥을 이어받도록 먼저 반영
                chat_history.append({"role": "user", "content": user_text})
                chat_history.append({"role": "assistant", "content": bot})
                update_dialog_state(dialog_state, user_text, bot)

                await send_json({"type": "bot_text", "text": bot})

                wav = await asyncio.to_thread(synth_wav_sync, bot)

                if my_turn != turn_id:
                    continue

                if wav:
                    try:
                        await ws.send_bytes(wav)
                    except Exception:
                        pass

    consumer_task = asyncio.create_task(final_consumer())

    try:
        while True:
            msg = await ws.receive()

            # control
            if "text" in msg and msg["text"]:
                try:
                    ctrl = json.loads(msg["text"])
                    ctype = ctrl.get("type")

                    if ctype == "stop":
                        break

                    if ctype == "barge_in":
                        bump_turn_and_barge_in()
                        continue

                    # ✅ 카메라 표정 신호 수신
                    if ctype == "vision_expression":
                        last_vision_state = ctrl

                        # throttled server log
                        now = time.time()
                        if now - _last_expr_log_ts > 2.0:
                            _last_expr_log_ts = now
                            e = ctrl.get("expression") or {}
                            print(
                                f"[VISION_EXPR] label={e.get('label')} "
                                f"conf={e.get('confidence')} v={e.get('valence')} a={e.get('arousal')}"
                            )
                        continue

                except Exception:
                    pass

            # audio bytes
            if "bytes" in msg and msg["bytes"]:
                push_stream.write(msg["bytes"])

    except WebSocketDisconnect:
        pass
    finally:
        # cleanup
        try:
            push_stream.close()
        except Exception:
            pass

        try:
            recognizer.stop_continuous_recognition()
        except Exception:
            pass

        try:
            consumer_task.cancel()
        except Exception:
            pass

        try:
            if _ACTIVE_SESSION_LOCK.locked():
                _ACTIVE_SESSION_LOCK.release()
        except Exception:
            pass


@app.get("/health")
def health():
    return {"ok": True}
