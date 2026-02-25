from __future__ import annotations

import os
import json
import asyncio
import struct
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI

load_dotenv()

app = FastAPI()

# ----------------------------
# NOTE: 멀티 유저 허용(단일세션 락 없음)
# 세션은 "웹소켓 연결 1개 = 1세션"으로 격리됨
# ----------------------------

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


def llm_reply(user_text: str) -> str:
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


def synth_wav(text: str) -> bytes:
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


def pack_wav_frame(playback_id: int, wav_bytes: bytes) -> bytes:
    """
    WS 바이너리 프레임:
      [0:4]  b'WAV0'
      [4:8]  uint32 playback_id (little-endian)
      [8:]   wav bytes
    """
    return b"WAV0" + struct.pack("<I", playback_id) + wav_bytes


@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    await ws.accept()

    # --- Azure Speech (STT) 준비 ---
    stream_format = speechsdk.audio.AudioStreamFormat(samples_per_second=16000, bits_per_sample=16, channels=1)
    push_stream = speechsdk.audio.PushAudioInputStream(stream_format)
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language = SPEECH_LANG

    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    loop = asyncio.get_running_loop()
    final_queue: asyncio.Queue[str] = asyncio.Queue()

    # --- 끼어들기/겹침 방지 상태 ---
    playback_id = 0                  # 답변 "세대" 번호
    current_task: asyncio.Task | None = None
    task_lock = asyncio.Lock()       # task 교체/취소 동기화

    last_bump_ts = 0.0             # stopPlayback 폭주 방지(초)

    async def bump_playback_and_stop(force: bool = False):
        """
        새 발화/새 답변이 시작되면:
        1) playback_id 증가
        2) 프론트에 stopPlayback 통지
        3) 이전 답변 task가 있으면 취소
        """
        nonlocal playback_id, current_task

        # 부분인식이 연속으로 들어오면 stopPlayback이 폭주할 수 있어 디바운스
        nonlocal last_bump_ts
        now = loop.time()
        if (not force) and (now - last_bump_ts < 0.4):
            return playback_id
        last_bump_ts = now

        async with task_lock:
            playback_id += 1
            try:
                await ws.send_text(json.dumps({"type": "stopPlayback", "playback_id": playback_id}))
            except Exception:
                pass

            if current_task and not current_task.done():
                current_task.cancel()
            current_task = None

        return playback_id

    # recognizing(부분인식) 들어오면: "사용자가 말하기 시작" => 즉시 stopPlayback
    def on_recognizing(evt):
        text = (evt.result.text or "").strip()
        if not text:
            return

        async def _send_partial():
            # 사용자가 말 시작하면 즉시 끼어들기 처리
            await bump_playback_and_stop()
            try:
                await ws.send_text(json.dumps({"type": "partial", "text": text}))
            except Exception:
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

    async def handle_turn(text: str, my_id: int):
        """
        한 턴(LLM -> TTS -> send audio)
        my_id == 현재 playback_id일 때만 '의미 있는' 오디오로 취급됨.
        프론트도 playback_id로 필터링하므로, 늦게 도착해도 재생 안됨.
        """
        try:
            await ws.send_text(json.dumps({"type": "final", "text": text, "playback_id": my_id}))

            # LLM (blocking) -> 필요하면 asyncio.to_thread로 빼도 됨
            bot = await asyncio.to_thread(llm_reply, text)
            if not bot:
                bot = "죄송합니다. 다시 한 번 말씀해 주세요."

            await ws.send_text(json.dumps({"type": "bot_text", "text": bot, "playback_id": my_id}))

            wav = await asyncio.to_thread(synth_wav, bot)
            if not wav:
                return

            frame = pack_wav_frame(my_id, wav)
            await ws.send_bytes(frame)

        except asyncio.CancelledError:
            # 새 발화가 들어와서 이전 턴이 취소된 경우
            return
        except Exception as e:
            try:
                await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
            except Exception:
                pass

    async def final_consumer():
        nonlocal current_task
        while True:
            text = await final_queue.get()

            # 너무 짧은 잡음 컷(선택)
            if len(text) <= 1:
                continue

            # 새 답변 시작 => 이전 task 취소 + playback_id 증가
            my_id = await bump_playback_and_stop()

            async with task_lock:
                current_task = asyncio.create_task(handle_turn(text, my_id))

    consumer_task = asyncio.create_task(final_consumer())

    try:
        while True:
            msg = await ws.receive()

            if "text" in msg and msg["text"]:
                try:
                    ctrl = json.loads(msg["text"])
                    if ctrl.get("type") == "stop":
                        break


                    if ctrl.get("type") == "barge_in":
                        # 프론트가 '끼어들기' 감지 -> 현재 턴 즉시 중단
                        await bump_playback_and_stop(force=True)
                        continue
                except Exception:
                    pass

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
        async with task_lock:
            if current_task and not current_task.done():
                current_task.cancel()


@app.get("/health")
def health():
    return {"ok": True}
