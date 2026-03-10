# server.py 원본
from __future__ import annotations

import os
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
    return AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)


AOAI = make_aoai_client()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "parking-llm").strip()

SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "").strip()
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "").strip()
SPEECH_LANG = os.getenv("AZURE_SPEECH_LANGUAGE", "ko-KR").strip()
SPEECH_VOICE = os.getenv("AZURE_SPEECH_VOICE", "ko-KR-SunHiNeural").strip()

if not SPEECH_KEY or not SPEECH_REGION:
    raise RuntimeError("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION env missing")


# ✅ 기본 시스템 프롬프트 (너 기존 정책 유지 + 카메라 언급 금지 명시)
SYSTEM_PROMPT_BASE = (
    "너는 주차장 고객상담 AI다. 한국어로 짧고 명확하게 안내한다. "
    "필요한 정보가 있으면 한 번에 1개만 질문한다. "
    "답변은 1~2문장으로, 너무 길게 말하지 않는다. "
    "말투는 공손하고, 예의있게 답변해라. "
    "규칙: 입력 문장의 문장부호(?, !)는 음성 인식 자동 보정 결과일 수 있으므로 의도 해석 시 과도하게 반영하지 마라. "
    "추가 규칙: 너의 역할은 '주차장/차량 출입/결제/요금/차단기/정기권/등록/시설 고장' 관련 상담만 한다. "
    "사용자 발화가 주차장 운영과 무관한 일반 지식(의료/병원/법률/투자/연애/정치 등)으로 해석될 가능성이 있으면, "
    "그 방향으로 절대 답하지 말고 '주차장 문의인지'를 한 문장으로 확인 질문을 해라. "
    "예: '소음' 같은 단어는 병원/의료로 연결하지 말고, '차단기/기기/경고음/안내방송 소리' 같은 주차장 상황으로 먼저 재해석한다. "
    "그래도 주차장과 무관하면: '주차장 문의만 가능해요. 어떤 문제인가요?' 라고 답한다. "
    # ✅ 카메라/표정 감지 언급 금지
    "중요: 카메라/얼굴/표정 인식이나 감정 감지 사실을 사용자에게 절대 언급하지 마라. "
    "또한 사용자의 감정/상태를 단정하거나 진단하지 마라."
)


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


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


def llm_reply_sync(user_text: str, vision_state: dict | None) -> str:
    """Blocking call (run via asyncio.to_thread)."""
    expr_policy = build_expression_policy(vision_state)

    resp = AOAI.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_BASE + expr_policy},
            {"role": "user", "content": user_text},
        ],
        temperature=0.3,
        max_tokens=220,
    )
    return (resp.choices[0].message.content or "").strip()


def synth_wav_sync(text: str) -> bytes:
    """Blocking TTS (run via asyncio.to_thread). Returns WAV bytes."""
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


@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    # (선택) 단일 세션만 허용
    if _ACTIVE_SESSION_LOCK.locked():
        await ws.accept()
        await ws.send_text(json.dumps({"type": "error", "message": "이미 상담이 진행 중입니다. 잠시 후 다시 시도하세요."}))
        await ws.close(code=1008)
        return

    await _ACTIVE_SESSION_LOCK.acquire()
    await ws.accept()

    # ✅ Latest expression-only camera signal (per websocket session)
    last_vision_state: dict | None = None
    _last_expr_log_ts = 0.0

    # Azure STT: 브라우저 PCM16(16k mono) 스트림 수신
    stream_format = speechsdk.audio.AudioStreamFormat(samples_per_second=16000, bits_per_sample=16, channels=1)
    push_stream = speechsdk.audio.PushAudioInputStream(stream_format)
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language = SPEECH_LANG
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

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
        loop.call_soon_threadsafe(lambda: asyncio.create_task(send_json({"type": "barge_in"})))

    def on_recognizing(evt):
        text = (evt.result.text or "").strip()
        if not text:
            return
        bump_turn_and_barge_in()
        loop.call_soon_threadsafe(lambda: asyncio.create_task(send_json({"type": "partial", "text": text})))

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

                # ✅ LLM에 표정 신호 전달
                bot = await asyncio.to_thread(llm_reply_sync, user_text, last_vision_state)
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
