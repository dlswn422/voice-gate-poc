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
# 🎧 Audio tuning (소음 무시 최종값)
# ==================================================
SILENCE_RMS_THRESHOLD = 0.003   # 시작 감지용 (종료에는 거의 영향 없음)
END_SILENCE_SEC = 0.1          # 조용할 때 빠른 종료용
PRERUN_SILENCE_SEC = 0.2
MIN_AUDIO_SEC = 0.5
CUT_AUDIO_SEC = 0.2
SAMPLE_RATE = 16000

MIN_SPEECH_FRAMES = 3
IGNORE_INPUT_AFTER_TTS_SEC = 0.35

MAX_SPEECH_SEC = 3.5            # ⭐⭐⭐ 핵심: 소음 있어도 무조건 종료 ⭐⭐⭐


# ==================================================
# 🔒 WebSocket utils
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

    # IO 상태 (엔진 상태랑 완전히 분리)
    io_state = "LISTENING"   # LISTENING | SPEAKING

    pcm_buffer: list[np.ndarray] = []
    collecting = False
    last_non_silence_ts = 0.0
    speech_start_ts = 0.0

    prerun_task: asyncio.Task | None = None
    speech_frame_count = 0
    ignore_until_ts = 0.0

    try:
        while True:
            message = await websocket.receive()

            # --------------------------------------------------
            # 프론트 제어 메시지 (TTS 종료)
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

            # 🔴 말하는 중엔 마이크 완전 무시
            if io_state == "SPEAKING":
                continue

            if time.time() < ignore_until_ts:
                continue

            pcm = np.frombuffer(message["bytes"], dtype=np.float32)
            if pcm.size == 0:
                continue

            now = time.time()
            rms = np.sqrt(np.mean(pcm * pcm))

            # --------------------------------------------------
            # 🎤 발화 시작 감지
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
                    speech_start_ts = now
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
            # 🛑 발화 종료 (침묵 OR 시간 강제 종료)
            # --------------------------------------------------
            if (
                silence_time >= END_SILENCE_SEC
                or speech_duration >= MAX_SPEECH_SEC
            ):
                collecting = False
                print("[WS] 🛑 Speech ended")

                total_samples = sum(len(c) for c in pcm_buffer)
                total_audio_sec = total_samples / SAMPLE_RATE

                if total_audio_sec < MIN_AUDIO_SEC:
                    pcm_buffer.clear()
                    prerun_task = None
                    continue

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
                    continue

                print(f"[STT] {text}")

                # ==================================================
                # 🧠 AppEngine 단일 진입점 (원래 구조 유지)
                # ==================================================
                result = app_state.app_engine.handle_text(text)

                reply_text = result.get("text", "")
                conversation_state = result.get("conversation_state", "WAITING_USER")
                end_session = result.get("end_session", False)

                if reply_text:
                    io_state = "SPEAKING"
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
                    await safe_close(websocket)
                    break

    except WebSocketDisconnect:
        print("[WS] ❌ Client disconnected")

    except Exception as e:
        print("[WS] 💥 Error:", repr(e))
        await safe_close(websocket)
