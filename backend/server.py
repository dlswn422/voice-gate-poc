from __future__ import annotations

import os
import json
import asyncio
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


def llm_reply_sync(user_text: str) -> str:
    """Blocking call (run via asyncio.to_thread)."""
    resp = AOAI.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "너는 주차장 고객상담 AI다. 한국어로 1~2문장으로 짧고 명확히 답한다. "
                    "필요하면 질문 1개만 한다. "
                    "입력 문장의 문장부호(?, !)는 음성 인식 자동 보정 결과일 수 있으므로 "
                    "의도 해석 시 과도하게 반영하지 마세요."
                ),
            },
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
    turn_id = 0                      # 새 발화/끼어들기 시 증가
    turn_lock = asyncio.Lock()       # 턴 처리 직렬화
    last_partial_ts = 0.0            # partial flood 방지 (선택)

    async def send_json(payload: dict):
        try:
            await ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            pass

    def bump_turn_and_barge_in():
        """User started speaking -> cancel current turn + notify frontend."""
        nonlocal turn_id
        turn_id += 1
        # 프론트가 재생 중인 오디오를 즉시 stop 하도록 힌트
        loop.call_soon_threadsafe(lambda: asyncio.create_task(send_json({"type": "barge_in"})))

    # 이벤트 핸들러 (STT 콜백은 별도 스레드에서 올 수 있으니 thread-safe)
    def on_recognizing(evt):
        # 사용자가 말하기 시작 = 끼어들기 가능성 -> turn 취소
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
        nonlocal turn_id
        while True:
            user_text = await final_queue.get()

            # 너무 짧은 잡음 컷
            if len(user_text) <= 1:
                continue

            # 이 발화에 대한 "내 턴 번호"
            my_turn = turn_id

            async with turn_lock:
                # 턴 락 잡기 전에 끼어들기 발생했으면 무효
                if my_turn != turn_id:
                    continue

                await send_json({"type": "final", "text": user_text})

                # LLM (thread로 분리)
                bot = await asyncio.to_thread(llm_reply_sync, user_text)
                if not bot:
                    bot = "죄송합니다. 다시 한 번 말씀해 주세요."

                # LLM 끝나는 동안 끼어들기 발생했으면 무효
                if my_turn != turn_id:
                    continue

                await send_json({"type": "bot_text", "text": bot})

                # TTS (thread로 분리)
                wav = await asyncio.to_thread(synth_wav_sync, bot)

                # TTS 중에도 끼어들기 발생했으면 무효 (전송 안 함)
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
                    if ctrl.get("type") == "stop":
                        break
                    # 프론트가 명시적으로 barge_in을 보내는 방식도 가능(옵션)
                    if ctrl.get("type") == "barge_in":
                        bump_turn_and_barge_in()
                except Exception:
                    pass

            # audio bytes
            if "bytes" in msg and msg["bytes"]:
                # 오디오가 들어오는 동안에도 "사용자가 말하는 중"이므로 기존 턴 취소를 더 공격적으로 하고 싶으면 아래 1줄 켜도 됨.
                # bump_turn_and_barge_in()
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
