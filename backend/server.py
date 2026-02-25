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
# Single-session guard (global)
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


def llm_reply(user_text: str) -> str:
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


def synth_wav(text: str) -> bytes:
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_synthesis_voice_name = SPEECH_VOICE
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
    )

    # audio_config=None -> 메모리로 WAV 바이트 반환
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    r = synthesizer.speak_text_async(text).get()

    if r.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        return b""
    return r.audio_data


@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    # 1) 단일 세션만 허용: 이미 누가 쓰고 있으면 즉시 거절
    if _ACTIVE_SESSION_LOCK.locked():
        # accept는 해야 close/send가 안정적으로 됨
        await ws.accept()
        await ws.send_text(json.dumps({"type": "error", "message": "이미 상담이 진행 중입니다. 잠시 후 다시 시도하세요."}))
        await ws.close(code=1008)  # Policy Violation
        return

    await _ACTIVE_SESSION_LOCK.acquire()
    await ws.accept()

    # PushAudioInputStream: 브라우저 PCM16(16k mono) 넣는 스트림
    stream_format = speechsdk.audio.AudioStreamFormat(
        samples_per_second=16000, bits_per_sample=16, channels=1
    )
    push_stream = speechsdk.audio.PushAudioInputStream(stream_format)
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language = SPEECH_LANG

    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    loop = asyncio.get_running_loop()
    final_queue: asyncio.Queue[str] = asyncio.Queue()

    # 2) 턴 처리(LLM+TTS)를 절대 겹치지 않게 직렬화
    turn_lock = asyncio.Lock()

    # 이벤트 핸들러(별도 스레드에서 콜백될 수 있으므로 thread-safe로 loop에 넘김)
    def on_recognizing(evt):
        text = (evt.result.text or "").strip()
        if not text:
            return

        async def _send_partial():
            try:
                await ws.send_text(json.dumps({"type": "partial", "text": text}))
            except Exception:
                # ws 끊긴 경우 무시
                pass

        loop.call_soon_threadsafe(lambda: asyncio.create_task(_send_partial()))

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
        while True:
            text = await final_queue.get()

            async with turn_lock:
                # (선택) 너무 짧은 잡음/호흡은 컷
                if len(text) <= 1:
                    continue

                await ws.send_text(json.dumps({"type": "final", "text": text}))

                # LLM (블로킹 호출이지만 데모에서는 OK / 필요하면 to_thread로 분리 가능)
                bot = llm_reply(text)
                if not bot:
                    bot = "죄송합니다. 다시 한 번 말씀해 주세요."

                await ws.send_text(json.dumps({"type": "bot_text", "text": bot}))

                # TTS -> WAV bytes 전송
                wav = synth_wav(bot)
                if wav:
                    await ws.send_bytes(wav)

    consumer_task = asyncio.create_task(final_consumer())

    try:
        while True:
            msg = await ws.receive()

            # control message
            if "text" in msg and msg["text"]:
                try:
                    ctrl = json.loads(msg["text"])
                    if ctrl.get("type") == "stop":
                        break
                except Exception:
                    pass

            # audio chunk
            if "bytes" in msg and msg["bytes"]:
                push_stream.write(msg["bytes"])

    except WebSocketDisconnect:
        pass
    finally:
        # 정리
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

        # 세션 락 해제
        try:
            if _ACTIVE_SESSION_LOCK.locked():
                _ACTIVE_SESSION_LOCK.release()
        except Exception:
            pass


@app.get("/health")
def health():
    return {"ok": True}
