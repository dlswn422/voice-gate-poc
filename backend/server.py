from __future__ import annotations

import os
import json
import asyncio
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI

load_dotenv()

app = FastAPI()


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


SYSTEM_PROMPT = (
    "너는 주차장 고객상담 AI다. 한국어로 1~2문장으로 짧고 명확히 답한다. "
    "필요하면 질문 1개만 한다. "
    "입력 문장의 문장부호(?, !)는 음성 인식 자동 보정 결과일 수 있으므로 "
    "의도 해석 시 과도하게 반영하지 마세요."
)


def llm_reply_sync(user_text: str) -> str:
    resp = AOAI.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        temperature=0.3,
        max_tokens=220,
    )
    return (resp.choices[0].message.content or "").strip()


def synth_wav_sync(text: str) -> bytes:
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_synthesis_voice_name = SPEECH_VOICE
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
    )
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    r = synthesizer.speak_text_async(text).get()
    if r.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        return b""
    return r.audio_data  # WAV bytes


@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    # ✅ 멀티 세션 허용 (락 제거)
    await ws.accept()

    # 세션 ID(디버깅용)
    session_id = uuid.uuid4().hex[:8]

    # --- STT 세션별 객체 생성 (절대 전역 공유 금지) ---
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

    # 한 WS 세션 안에서 턴 처리(LLM+TTS)는 직렬화
    turn_lock = asyncio.Lock()

    # TTS “재생 세대” 관리: 새로운 답변이 시작되면 playback_id 증가
    playback_id = 0

    def safe_create_task(coro):
        try:
            asyncio.create_task(coro)
        except RuntimeError:
            # loop closed 등
            pass

    # 이벤트 핸들러(스레드 콜백 가능 → thread-safe로 loop에 올림)
    def on_recognizing(evt):
        text = (evt.result.text or "").strip()
        if not text:
            return

        async def _send_partial():
            try:
                await ws.send_text(json.dumps({"type": "partial", "text": text, "sid": session_id}))
            except Exception:
                pass

        loop.call_soon_threadsafe(lambda: safe_create_task(_send_partial()))

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
        nonlocal playback_id
        while True:
            text = await final_queue.get()

            async with turn_lock:
                # 너무 짧은 잡음 컷
                if len(text) <= 1:
                    continue

                # FINAL 텍스트
                try:
                    await ws.send_text(json.dumps({"type": "final", "text": text, "sid": session_id}))
                except Exception:
                    return

                # ✅ 새 답변 시작: playback_id 증가 + 프론트에 stopPlayback 신호
                playback_id += 1
                current_pid = playback_id
                try:
                    await ws.send_text(json.dumps({"type": "stopPlayback", "playback_id": current_pid, "sid": session_id}))
                except Exception:
                    return

                # LLM (블로킹 → 스레드로)
                try:
                    bot = await asyncio.to_thread(llm_reply_sync, text)
                except Exception:
                    bot = ""

                if not bot:
                    bot = "죄송합니다. 다시 한 번 말씀해 주세요."

                # bot 텍스트
                try:
                    await ws.send_text(json.dumps({"type": "bot_text", "text": bot, "playback_id": current_pid, "sid": session_id}))
                except Exception:
                    return

                # TTS (블로킹 → 스레드로)
                try:
                    wav = await asyncio.to_thread(synth_wav_sync, bot)
                except Exception:
                    wav = b""

                # ✅ 혹시 다음 턴이 먼저 시작됐으면(끼어들기) 이 wav는 버림
                if current_pid != playback_id:
                    continue

                if wav:
                    try:
                        await ws.send_bytes(wav)
                    except Exception:
                        return

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


@app.get("/health")
def health():
    return {"ok": True}
