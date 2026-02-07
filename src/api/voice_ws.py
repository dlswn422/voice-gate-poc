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
# 🔧 튜닝 포인트 (웹 마이크 기준 확정값)
# ==================================================
SILENCE_RMS_THRESHOLD = 0.003
END_SILENCE_SEC = 0.25
PRERUN_SILENCE_SEC = 0.2
MIN_AUDIO_SEC = 0.5
CUT_AUDIO_SEC = 0.2
SAMPLE_RATE = 16000

MIN_SPEECH_FRAMES = 3
IGNORE_INPUT_AFTER_TTS_SEC = 0.35


# ==================================================
# 🔒 WebSocket 안전 유틸
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

    pcm_buffer: list[np.ndarray] = []
    collecting = False
    last_non_silence_ts = 0.0

    prerun_task: asyncio.Task | None = None
    speech_frame_count = 0
    ignore_until_ts = 0.0

    app_state.app_engine.state = "LISTENING"

    try:
        while True:
            message = await websocket.receive()

            # --------------------------------------------------
            # 디버그: 메시지 수신 확인
            # --------------------------------------------------
            if "bytes" not in message and "text" not in message:
                continue

            # --------------------------------------------------
            # 프론트 제어 메시지
            # --------------------------------------------------
            if "text" in message:
                try:
                    msg = json.loads(message["text"])
                    if msg.get("type") == "tts_end":
                        print("[WS] 🔁 TTS ended → LISTENING")
                        app_state.app_engine.state = "LISTENING"
                        collecting = False
                        pcm_buffer.clear()
                        prerun_task = None
                        speech_frame_count = 0
                        ignore_until_ts = time.time() + IGNORE_INPUT_AFTER_TTS_SEC
                        continue
                except Exception:
                    continue

            # --------------------------------------------------
            # 오디오 프레임
            # --------------------------------------------------
            if "bytes" not in message:
                continue

            if time.time() < ignore_until_ts:
                continue

            pcm = np.frombuffer(message["bytes"], dtype=np.float32)
            if pcm.size == 0:
                continue

            now = time.time()
            rms = np.sqrt(np.mean(pcm * pcm))

            # 🔍 디버그 로그 (지금 상태 바로 보이게)
            print(f"[DEBUG] rms={rms:.5f}, collecting={collecting}")

            # --------------------------------------------------
            # THINKING / SPEAKING 중 입력 무시
            # --------------------------------------------------
            if app_state.app_engine.state in ("THINKING", "SPEAKING"):
                collecting = False
                pcm_buffer.clear()
                prerun_task = None
                speech_frame_count = 0
                continue

            # --------------------------------------------------
            # 🎤 발화 시작 감지 (RMS ONLY)
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
                    last_non_silence_ts = now
                    print("[WS] 🎤 Speech started")
                continue

            # --------------------------------------------------
            # 발화 수집 중
            # --------------------------------------------------
            if rms > SILENCE_RMS_THRESHOLD:
                pcm_buffer.append(pcm)
                last_non_silence_ts = now
                continue

            silence_time = now - last_non_silence_ts

            # --------------------------------------------------
            # STT pre-run
            # --------------------------------------------------
            if collecting and prerun_task is None and silence_time >= PRERUN_SILENCE_SEC:
                print("[WS] ⚡ STT pre-run")

                audio = np.concatenate(pcm_buffer).astype(np.float32)
                cut_samples = int(SAMPLE_RATE * CUT_AUDIO_SEC)
                if audio.size > cut_samples:
                    audio = audio[:-cut_samples]

                prerun_task = asyncio.create_task(
                    asyncio.to_thread(
                        transcribe_pcm_chunks,
                        [audio],
                        app_state.whisper_model,
                    )
                )

            # --------------------------------------------------
            # 🛑 발화 종료
            # --------------------------------------------------
            if collecting and silence_time >= END_SILENCE_SEC:
                collecting = False
                print("[WS] 🛑 Speech ended")

                total_samples = sum(len(c) for c in pcm_buffer)
                total_audio_sec = total_samples / SAMPLE_RATE

                if total_audio_sec < MIN_AUDIO_SEC:
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
                        print("[WS] ⚡ pre-run STT reused")
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
                    app_state.app_engine.state = "LISTENING"
                    continue

                print(f"[STT] {text}")

                # --------------------------------------------------
                # AppEngine
                # --------------------------------------------------
                app_state.app_engine.state = "THINKING"
                result = app_state.app_engine.handle_text(text)

                reply_text = result.get("text", "")
                conversation_state = result.get("conversation_state", "WAITING_USER")
                end_session = result.get("end_session", False)

                # --------------------------------------------------
                # TTS
                # --------------------------------------------------
                tts_url = None
                if reply_text:
                    app_state.app_engine.state = "SPEAKING"
                    tts_url = synthesize(reply_text)

                await safe_send(websocket, {
                    "type": "assistant_message",
                    "text": reply_text,
                    "tts_url": tts_url,
                    "conversation_state": conversation_state,
                    "end_session": end_session,
                })

                if end_session:
                    print("[WS] 🛑 Conversation ended")
                    app_state.app_engine.state = "IDLE"
                    await safe_close(websocket)
                    break

    except WebSocketDisconnect:
        print("[WS] ❌ Client disconnected")

    except Exception as e:
        print("[WS] 💥 Error:", repr(e))
        await safe_close(websocket)