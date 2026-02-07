from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

import numpy as np
import time
import json
import asyncio

import src.app_state as app_state
from src.speech.whisper_service import transcribe_pcm_chunks
from src.speech.tts import synthesize

router = APIRouter()

# ==================================================
# 🎧 Outdoor parking lot tuning (FINAL)
# ==================================================
SILENCE_RMS_THRESHOLD = 0.008
END_SILENCE_SEC = 0.12
PRERUN_SILENCE_SEC = 0.25
MIN_AUDIO_SEC = 0.6
CUT_AUDIO_SEC = 0.25
SAMPLE_RATE = 16000

MIN_SPEECH_FRAMES = 4
IGNORE_INPUT_AFTER_TTS_SEC = 0.6

MAX_SPEECH_SEC = 3.0
POST_SPEECH_IGNORE_SEC = 0.5


# ==================================================
# 🔒 WS utils
# ==================================================
async def safe_send(ws: WebSocket, payload: dict):
    if ws.application_state == WebSocketState.CONNECTED:
        await ws.send_json(payload)


async def safe_close(ws: WebSocket):
    if ws.application_state == WebSocketState.CONNECTED:
        await ws.close()


@router.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket):
    await websocket.accept()
    print("[WS] 🔌 Client connected")

    # IO 상태 (엔진 상태와 완전히 분리)
    io_state = "LISTENING"   # LISTENING | SPEAKING

    pcm_buffer: list[np.ndarray] = []
    collecting = False

    speech_start_ts = 0.0
    last_non_silence_ts = 0.0
    ignore_until_ts = 0.0

    speech_frame_count = 0
    prerun_task: asyncio.Task | None = None

    try:
        while True:
            message = await websocket.receive()

            # --------------------------------------------------
            # 프론트 → TTS 종료
            # --------------------------------------------------
            if "text" in message:
                try:
                    msg = json.loads(message["text"])
                    if msg.get("type") == "tts_end":
                        print("[WS] 🔁 TTS ended → LISTENING")
                        io_state = "LISTENING"
                        collecting = False
                        pcm_buffer.clear()
                        prerun_task = None
                        speech_frame_count = 0
                        ignore_until_ts = time.time() + IGNORE_INPUT_AFTER_TTS_SEC
                        continue
                except Exception:
                    pass

            # --------------------------------------------------
            # 오디오 프레임
            # --------------------------------------------------
            if "bytes" not in message:
                continue

            if io_state == "SPEAKING":
                continue

            now = time.time()
            if now < ignore_until_ts:
                continue

            pcm = np.frombuffer(message["bytes"], dtype=np.float32)
            if pcm.size == 0:
                continue

            rms = float(np.sqrt(np.mean(pcm * pcm)))

            # --------------------------------------------------
            # 🎤 발화 시작
            # --------------------------------------------------
            if not collecting:
                if rms > SILENCE_RMS_THRESHOLD:
                    speech_frame_count += 1
                else:
                    speech_frame_count = 0

                if speech_frame_count >= MIN_SPEECH_FRAMES:
                    collecting = True
                    pcm_buffer.clear()
                    prerun_task = None
                    speech_frame_count = 0
                    speech_start_ts = now
                    last_non_silence_ts = now
                    print("[WS] 🎤 Speech started")
                continue

            # --------------------------------------------------
            # 발화 수집
            # --------------------------------------------------
            pcm_buffer.append(pcm)

            if rms > SILENCE_RMS_THRESHOLD:
                last_non_silence_ts = now

            silence_time = now - last_non_silence_ts
            speech_duration = now - speech_start_ts

            # --------------------------------------------------
            # STT pre-run
            # --------------------------------------------------
            if prerun_task is None and silence_time >= PRERUN_SILENCE_SEC:
                audio = np.concatenate(pcm_buffer).astype(np.float32)
                cut = int(SAMPLE_RATE * CUT_AUDIO_SEC)
                if audio.size > cut:
                    audio = audio[:-cut]

                prerun_task = asyncio.create_task(
                    asyncio.to_thread(
                        transcribe_pcm_chunks,
                        [audio],
                        app_state.whisper_model,
                    )
                )
                print("[WS] ⚡ STT pre-run")

            # --------------------------------------------------
            # 🛑 발화 종료
            # --------------------------------------------------
            if silence_time >= END_SILENCE_SEC or speech_duration >= MAX_SPEECH_SEC:
                collecting = False
                ignore_until_ts = time.time() + POST_SPEECH_IGNORE_SEC
                print("[WS] 🛑 Speech ended")

                total_samples = sum(len(c) for c in pcm_buffer)
                if total_samples / SAMPLE_RATE < MIN_AUDIO_SEC:
                    pcm_buffer.clear()
                    prerun_task = None
                    continue

                await safe_send(websocket, {
                    "type": "assistant_state",
                    "state": "THINKING",
                })

                if prerun_task:
                    try:
                        text = await prerun_task
                    except Exception:
                        text = ""
                else:
                    text = transcribe_pcm_chunks(
                        pcm_buffer,
                        whisper_model=app_state.whisper_model,
                    )

                pcm_buffer.clear()
                prerun_task = None

                if not text:
                    continue

                print(f"[STT] {text}")

                # ==================================================
                # 🧠 AppEngine (단일 진입점)
                # ==================================================
                result = app_state.app_engine.handle_text(text)

                reply_text = result.get("text", "")
                end_session = result.get("end_session", False)

                if reply_text:
                    io_state = "SPEAKING"
                    tts_url = synthesize(reply_text)

                    # ⭐ 핵심 수정: AppEngine 결과 그대로 전달
                    payload = dict(result)
                    payload["tts_url"] = tts_url

                    await safe_send(websocket, payload)

                if end_session:
                    print("[WS] 🛑 Conversation ended")
                    await safe_close(websocket)
                    break

    except WebSocketDisconnect:
        print("[WS] ❌ Client disconnected")

    except Exception as e:
        print("[WS] 💥 Error:", repr(e))
        await safe_close(websocket)
