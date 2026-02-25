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
    # 1) 단일 세션만 허용
    if _ACTIVE_SESSION_LOCK.locked():
        await ws.accept()
        await ws.send_text(
            json.dumps({"type": "error", "message": "이미 상담이 진행 중입니다. 잠시 후 다시 시도하세요."})
        )
        await ws.close(code=1008)
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

    # 2) 턴 처리(LLM+TTS)를 직렬화
    turn_lock = asyncio.Lock()

    # ----------------------------
    # ✅ barge-in + 턴 무효화 가드
    # ----------------------------
    speech_active = False        # 사용자가 현재 말하고 있는지(Recognizing 구간)
    active_turn_id = 0           # 현재 유효한 턴(바징인 발생 시 증가)

    async def send_json(obj: dict):
        try:
            await ws.send_text(json.dumps(obj))
        except Exception:
            pass

    # 이벤트 핸들러(별도 스레드에서 콜백될 수 있으므로 thread-safe로 loop에 넘김)
    def on_recognizing(evt):
        nonlocal speech_active, active_turn_id

        text = (evt.result.text or "").strip()
        if not text:
            return

        # ✅ 사용자가 말 시작한 "첫 순간"에만 barge_in + 현재 턴 무효화
        if not speech_active:
            speech_active = True
            active_turn_id += 1  # 진행 중이던 bot 응답/tts 전송을 무효화(턴 ID 증가)

            loop.call_soon_threadsafe(lambda: asyncio.create_task(send_json({"type": "barge_in"})))

        # partial 전송
        loop.call_soon_threadsafe(
            lambda: asyncio.create_task(send_json({"type": "partial", "text": text}))
        )

    def on_recognized(evt):
        nonlocal speech_active
        # ✅ final이 떨어지면 발화 구간 종료로 간주
        speech_active = False

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
        nonlocal active_turn_id
        while True:
            text = await final_queue.get()

            async with turn_lock:
                # 턴 시작 시점의 id 스냅샷
                my_turn_id = active_turn_id

                # (선택) 너무 짧은 잡음/호흡은 컷
                if len(text) <= 1:
                    continue

                await send_json({"type": "final", "text": text})

                # ✅ LLM/TTS는 블로킹이라 to_thread로 분리(이벤트루프 끊김 방지)
                bot = await asyncio.to_thread(llm_reply, text)
                if not bot:
                    bot = "죄송합니다. 다시 한 번 말씀해 주세요."

                # ✅ 바징인이 발생해서 turn_id가 바뀌었으면(사용자가 끼어듦) 이 턴 결과는 버림
                if my_turn_id != active_turn_id:
                    continue

                await send_json({"type": "bot_text", "text": bot})

                # TTS
                wav = await asyncio.to_thread(synth_wav, bot)

                # ✅ TTS 생성 중에도 바징인 발생 가능 → 다시 확인 후 전송
                if my_turn_id != active_turn_id:
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
