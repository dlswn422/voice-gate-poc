# server.py
from __future__ import annotations

import os
import re
import json
import asyncio
import time
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
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


AOAI = make_aoai_client()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "parking-llm").strip()

SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "").strip()
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "").strip()
SPEECH_LANG = os.getenv("AZURE_SPEECH_LANGUAGE", "ko-KR").strip()
SPEECH_VOICE = os.getenv("AZURE_SPEECH_VOICE", "ko-KR-SunHiNeural").strip()

if not SPEECH_KEY or not SPEECH_REGION:
    raise RuntimeError("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION env missing")


# ----------------------------
# Prompt
# ----------------------------
SYSTEM_PROMPT_BASE = (
    "너는 주차장 고객상담 AI다. 한국어로 짧고 명확하게 안내한다. "
    "필요한 정보가 있으면 한 번에 1개만 질문한다. "
    "답변은 1~2문장으로, 너무 길게 말하지 않는다. "
    "말투는 공손하고, 예의있게 답변해라. "
    "규칙: 입력 문장의 문장부호(?, !)는 음성 인식 자동 보정 결과일 수 있으므로 의도 해석 시 과도하게 반영하지 마라. "
    "추가 규칙: 너의 역할은 '주차장/차량 출입/결제/요금/차단기/정기권/등록/시설 고장' 관련 상담만 한다. "
    "사용자 발화가 주차장 운영과 무관한 일반 지식(의료/병원/법률/투자/연애/정치 등)으로 해석될 가능성이 있으면, "
    "그 방향으로 절대 답하지 말고 '주차장 문의인지'를 한 문장으로 확인 질문을 해라. "
    "예: '소음' 같은 단어는 병원/의료로 연결하지 말고, "
    "'차단기/기기/경고음/안내방송 소리' 같은 주차장 상황으로 먼저 재해석한다. "
    "그래도 주차장과 무관하면: '주차장 문의만 가능해요. 어떤 문제인가요?' 라고 답한다. "
    "핵심 해석 규칙: 사용자가 짧고 명령형으로 말하면, 일반 대화로 해석하지 말고 먼저 주차장 운영 의도로 해석하라. "
    "특히 '문 열어', '문열어', '문 열어줘', '열어줘', '열어주세요', "
    "'차단기 열어줘', '차단봉 올려', '차단봉 올려주세요', '바 올려', '게이트 열어줘', "
    "'출구 열어줘', '입구 열어줘' 같은 표현은 우선적으로 "
    "'차단기/게이트 개방 요청 또는 개방 불가 문제'로 해석하라. "
    "여기서 '문'은 일반 문보다 주차장 출입 차단기/게이트를 우선 의미하는 것으로 본다. "
    "이런 발화가 들어오면 일반적인 되묻기보다, "
    "차단기 개방 요청인지 또는 차단기가 열리지 않는 상황인지 중심으로 짧게 확인하라. "
    "응답 규칙: "
    "1) 개방 요청으로 보이면 '차단기 개방 요청이신가요? 차량번호를 말씀해 주세요.'처럼 답하라. "
    "2) 열리지 않는 문제로 보이면 "
    "'차단기가 열리지 않나요? 입차 중인지 출차 중인지와 차량번호를 말씀해 주세요.'처럼 답하라. "
    "3) 항상 주차장 운영 맥락으로 먼저 재해석하라. "
    "중요: 카메라/얼굴/표정 인식이나 감정 감지 사실을 사용자에게 절대 언급하지 마라. "
    "또한 사용자의 감정/상태를 단정하거나 진단하지 마라."
)

# few-shot examples
FEW_SHOT_MESSAGES = [
    {"role": "user", "content": "문 열어"},
    {"role": "assistant", "content": "차단기 개방 요청이신가요? 차량번호를 말씀해 주세요."},
    {"role": "user", "content": "문열어줘"},
    {"role": "assistant", "content": "차단기 개방 요청으로 이해했습니다. 차량번호를 말씀해 주세요."},
    {"role": "user", "content": "차단봉 올려주세요"},
    {"role": "assistant", "content": "차단기 개방 요청이신가요? 차량번호를 말씀해 주세요."},
    {"role": "user", "content": "안 열리는데요"},
    {
        "role": "assistant",
        "content": "차단기가 열리지 않나요? 입차 중인지 출차 중인지와 차량번호를 말씀해 주세요.",
    },
    {"role": "user", "content": "출구 막혔어요"},
    {
        "role": "assistant",
        "content": "출차 차단기 문제이신가요? 차량번호를 말씀해 주세요.",
    },
]


# ----------------------------
# Helpers
# ----------------------------
def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _compact_text(text: str) -> str:
    text = _normalize_text(text)
    # 공백/문장부호 제거 후 현장 발화 매칭용으로 축약
    text = re.sub(r"[\s\?\!\.\,\~\-_/]+", "", text)
    return text


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


# ----------------------------
# Short utterance normalization
# ----------------------------
_GATE_REQUEST_EXACT = {
    "문열어",
    "문열어줘",
    "문열어주세요",
    "열어",
    "열어줘",
    "열어주세요",
    "차단기열어",
    "차단기열어줘",
    "차단기열어주세요",
    "차단봉올려",
    "차단봉올려줘",
    "차단봉올려주세요",
    "게이트열어",
    "게이트열어줘",
    "게이트열어주세요",
    "바올려",
    "바올려줘",
    "바올려주세요",
    "입구열어",
    "입구열어줘",
    "입구열어주세요",
    "출구열어",
    "출구열어줘",
    "출구열어주세요",
}

_GATE_FAILURE_EXACT = {
    "안열려",
    "안열리는데",
    "안열리네요",
    "문이안열려",
    "문이안열리는데",
    "차단기가안열려",
    "차단기가안열리는데",
    "차단봉이안올라가",
    "차단봉이안올라가요",
    "게이트가안열려",
    "게이트가안열리는데",
    "출구가막혔어",
    "출구가막혔어요",
    "입구가막혔어",
    "입구가막혔어요",
    "출차가안돼",
    "출차가안되",
    "출차가안돼요",
    "출차가안됩니다",
    "입차가안돼",
    "입차가안되",
    "입차가안돼요",
    "입차가안됩니다",
}

_GATE_WORDS = (
    "문",
    "차단기",
    "차단봉",
    "게이트",
    "바",
    "입구",
    "출구",
    "입차",
    "출차",
)

_OPEN_VERBS = (
    "열어",
    "올려",
    "개방",
)

_FAILURE_MARKERS = (
    "안열",
    "안올라",
    "막혔",
    "막혀",
    "안돼",
    "안되",
    "오류",
    "고장",
)


def classify_short_gate_intent(user_text: str) -> dict | None:
    """
    짧은 현장 발화를 차단기 개방 요청 / 차단기 개방 실패로 정규화.
    너무 넓게 잡지 않기 위해 '짧은 발화' 또는 '강한 exact match' 위주로만 동작.
    """
    raw = _normalize_text(user_text)
    compact = _compact_text(raw)
    if not compact:
        return None

    is_short = (len(raw) <= 20) or (len(compact) <= 12)

    # 1) failure 먼저 체크
    if compact in _GATE_FAILURE_EXACT:
        return {
            "intent": "gate_open_failure",
            "confidence": 0.99,
            "source": "exact_failure",
            "normalized_text": compact,
        }

    if is_short and any(marker in compact for marker in _FAILURE_MARKERS) and any(
        gate_word in compact for gate_word in _GATE_WORDS
    ):
        return {
            "intent": "gate_open_failure",
            "confidence": 0.93,
            "source": "short_failure_pattern",
            "normalized_text": compact,
        }

    # 2) open request 체크
    if compact in _GATE_REQUEST_EXACT:
        return {
            "intent": "gate_open_request",
            "confidence": 0.99,
            "source": "exact_request",
            "normalized_text": compact,
        }

    if is_short and compact in {"열어", "열어줘", "열어주세요", "올려", "올려줘", "올려주세요"}:
        return {
            "intent": "gate_open_request",
            "confidence": 0.95,
            "source": "short_request_exact",
            "normalized_text": compact,
        }

    if is_short and any(open_verb in compact for open_verb in _OPEN_VERBS) and any(
        gate_word in compact for gate_word in _GATE_WORDS
    ):
        return {
            "intent": "gate_open_request",
            "confidence": 0.92,
            "source": "short_request_pattern",
            "normalized_text": compact,
        }

    return None


def build_operation_hint(intent_result: dict | None) -> str:
    if not isinstance(intent_result, dict):
        return ""

    intent = intent_result.get("intent")
    conf = _safe_float(intent_result.get("confidence", 0.0))
    source = intent_result.get("source", "")

    if intent == "gate_open_request":
        return (
            "\n[운영 의도 힌트] "
            f"서버 전처리 결과: gate_open_request (confidence={conf:.2f}, source={source}).\n"
            "이 입력은 현장형 짧은 발화로 보이며, 차단기/게이트 개방 요청으로 우선 해석하라.\n"
            "가능하면 일반적인 되묻기 대신, 차량번호 요청 또는 입차/출차 확인 중심으로 답하라.\n"
        )

    if intent == "gate_open_failure":
        return (
            "\n[운영 의도 힌트] "
            f"서버 전처리 결과: gate_open_failure (confidence={conf:.2f}, source={source}).\n"
            "이 입력은 차단기/게이트가 열리지 않거나 막힌 상황으로 우선 해석하라.\n"
            "가능하면 일반적인 되묻기 대신, 입차/출차 상태와 차량번호 확인 중심으로 답하라.\n"
        )

    return ""


def render_direct_response(intent_result: dict | None) -> str | None:
    """
    확실한 short-utterance 케이스는 LLM을 거치지 않고 템플릿 응답.
    운영 안정성 확보용.
    """
    if not isinstance(intent_result, dict):
        return None

    intent = intent_result.get("intent")
    conf = _safe_float(intent_result.get("confidence", 0.0))

    # 너무 약한 분류면 direct response 사용하지 않음
    if conf < 0.90:
        return None

    if intent == "gate_open_request":
        return "차단기 개방 요청이신가요? 차량번호를 말씀해 주세요."

    if intent == "gate_open_failure":
        return "차단기가 열리지 않나요? 입차 중인지 출차 중인지와 차량번호를 말씀해 주세요."

    return None


# ----------------------------
# LLM / TTS
# ----------------------------
def llm_reply_sync(
    user_text: str,
    vision_state: dict | None,
    intent_result: dict | None = None,
) -> str:
    """Blocking call (run via asyncio.to_thread)."""
    expr_policy = build_expression_policy(vision_state)
    op_hint = build_operation_hint(intent_result)

    resp = AOAI.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_BASE + expr_policy + op_hint},
            *FEW_SHOT_MESSAGES,
            {"role": "user", "content": user_text},
        ],
        temperature=0.3,
        max_tokens=220,
    )
    return (resp.choices[0].message.content or "").strip()


def synth_wav_sync(text: str) -> bytes:
    """Blocking TTS (run via asyncio.to_thread). Returns WAV bytes."""
    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION,
    )
    speech_config.speech_synthesis_voice_name = SPEECH_VOICE
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
    )
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=None,
    )
    r = synthesizer.speak_text_async(text).get()
    if r.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        return b""
    return r.audio_data


@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    # (선택) 단일 세션만 허용
    if _ACTIVE_SESSION_LOCK.locked():
        await ws.accept()
        await ws.send_text(
            json.dumps(
                {
                    "type": "error",
                    "message": "이미 상담이 진행 중입니다. 잠시 후 다시 시도하세요.",
                },
                ensure_ascii=False,
            )
        )
        await ws.close(code=1008)
        return

    await _ACTIVE_SESSION_LOCK.acquire()
    await ws.accept()

    # Latest expression-only camera signal (per websocket session)
    last_vision_state: dict | None = None
    _last_expr_log_ts = 0.0

    # Azure STT: 브라우저 PCM16(16k mono) 스트림 수신
    stream_format = speechsdk.audio.AudioStreamFormat(
        samples_per_second=16000,
        bits_per_sample=16,
        channels=1,
    )
    push_stream = speechsdk.audio.PushAudioInputStream(stream_format)
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION,
    )
    speech_config.speech_recognition_language = SPEECH_LANG
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
    )

    loop = asyncio.get_running_loop()
    final_queue: asyncio.Queue[str] = asyncio.Queue()

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
        loop.call_soon_threadsafe(
            lambda: asyncio.create_task(send_json({"type": "barge_in"}))
        )

    def on_recognizing(evt):
        text = (evt.result.text or "").strip()
        if not text:
            return
        bump_turn_and_barge_in()
        loop.call_soon_threadsafe(
            lambda: asyncio.create_task(send_json({"type": "partial", "text": text}))
        )

    def on_recognized(evt):
        if evt.result.reason != speechsdk.ResultReason.RecognizedSpeech:
            return
        text = (evt.result.text or "").strip()
        if not text:
            return
        loop.call_soon_threadsafe(final_queue.put_nowait, text)

    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.start_continuous_recognition()

    async def final_consumer():
        nonlocal turn_id, last_vision_state
        while True:
            user_text = await final_queue.get()

            if len(user_text) <= 1:
                continue

            my_turn = turn_id

            async with turn_lock:
                if my_turn != turn_id:
                    continue

                await send_json({"type": "final", "text": user_text})

                # 1) short utterance normalization
                intent_result = classify_short_gate_intent(user_text)
                if intent_result:
                    print(
                        "[INTENT_HINT]",
                        f"text={user_text!r}",
                        f"intent={intent_result.get('intent')}",
                        f"conf={intent_result.get('confidence')}",
                        f"source={intent_result.get('source')}",
                    )

                # 2) direct response for strong, high-confidence cases
                bot = render_direct_response(intent_result)

                # 3) fallback to LLM if no direct response
                if not bot:
                    bot = await asyncio.to_thread(
                        llm_reply_sync,
                        user_text,
                        last_vision_state,
                        intent_result,
                    )

                if not bot:
                    bot = "죄송합니다. 다시 한 번 말씀해 주세요."

                if my_turn != turn_id:
                    continue

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

                    # 카메라 표정 신호 수신
                    if ctype == "vision_expression":
                        last_vision_state = ctrl

                        # throttled server log
                        now = time.time()
                        if now - _last_expr_log_ts > 2.0:
                            _last_expr_log_ts = now
                            e = ctrl.get("expression") or {}
                            print(
                                f"[VISION_EXPR] label={e.get('label')} "
                                f"conf={e.get('confidence')} "
                                f"v={e.get('valence')} "
                                f"a={e.get('arousal')}"
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
