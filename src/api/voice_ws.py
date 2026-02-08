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
# 🎧 Outdoor Parking Lot Voice Tuning (FINAL - STABLE)
# ==================================================
# ⚠️ 기준 환경
# - 실외 주차장
# - 차량 엔진음 / 바람 / 주변 대화 존재
# - 키오스크 마이크 (AGC / NS / EC 켜짐)

# ▶ 무음 판단 RMS 기준
# - 0.003~0.004 : 실내
# - 0.004~0.005 : 실외(권장)
# - 0.006↑      : 작은 목소리 인식 실패 가능
SILENCE_RMS_THRESHOLD = 0.0045

# ▶ 발화 종료로 판단하는 침묵 시간 (초)
# - 너무 짧으면 문장 중간 끊김
# - 너무 길면 응답 느림
END_SILENCE_SEC = 0.25

# ▶ 발화 중 잠깐 멈췄을 때 STT pre-run 시작 시점
# - 체감 응답 속도 개선용
PRERUN_SILENCE_SEC = 0.3

# ▶ 최소 음성 길이 (초)
# - 이보다 짧으면 의미 없는 소리로 판단
MIN_AUDIO_SEC = 0.5

# ▶ STT pre-run 시 뒤쪽 잡음 컷 (초)
CUT_AUDIO_SEC = 0.2

SAMPLE_RATE = 16000

# ▶ 발화 시작으로 인정할 최소 연속 프레임 수
# - 값이 클수록 소음에 강함, 대신 반응 느림
MIN_SPEECH_FRAMES = 2

# ▶ TTS 종료 직후 입력 무시 시간
# - TTS 자기 음성 재인식 방지
# - 너무 길면 사용자가 바로 말해도 안 잡힘
IGNORE_INPUT_AFTER_TTS_SEC = 0.2

# ▶ 최대 발화 허용 시간 (초)
# - 너무 길면 강제 종료
MAX_SPEECH_SEC = 4.0

# ▶ 발화 종료 직후 짧은 무시 구간
POST_SPEECH_IGNORE_SEC = 0.25

# ▶ 아무 말도 없을 때 자동 종료 시간
NO_INPUT_TIMEOUT_SEC = 8.0


# ==================================================
# 🔒 WS utils
# ==================================================
async def safe_send(ws: WebSocket, payload: dict):
    if ws.application_state == WebSocketState.CONNECTED:
        await ws.send_json(payload)


async def safe_close(ws: WebSocket):
    if ws.application_state == WebSocketState.CONNECTED:
        await ws.close()


# ==================================================
# 🧠 의미 없는 발화 필터
# ==================================================
def is_meaningful_text(text: str) -> bool:
    """
    STT 결과가 실제 의미 있는 발화인지 판단
    - 너무 짧은 발화 제거
    - 추임새 / 감탄사 제거
    """
    if not text:
        return False

    t = text.strip()

    # 글자 수 기준
    if len(t) < 3:
        return False

    meaningless = {
        "어", "음", "아", "네", "예",
        "어어", "음음", "응", "어?", "음?"
    }

    if t in meaningless:
        return False

    return True


# ==================================================
# 🎤 Voice WebSocket
# ==================================================
@router.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket):
    await websocket.accept()
    print("[WS] 🔌 Client connected")

    # ▶ 서버 기준 IO 상태
    # LISTENING : 마이크 입력 허용
    # THINKING  : STT / LLM 처리 중
    # SPEAKING  : TTS 재생 중
    io_state = "LISTENING"

    pcm_buffer: list[np.ndarray] = []
    collecting = False

    speech_start_ts = 0.0
    last_non_silence_ts = 0.0
    ignore_until_ts = 0.0

    speech_frame_count = 0
    prerun_task: asyncio.Task | None = None

    # ▶ 마지막 사용자 활동 시간
    last_activity_ts = time.time()

    try:
        while True:
            # --------------------------------------------------
            # 🕒 무응답 자동 종료
            # --------------------------------------------------
            if io_state == "LISTENING":
                if time.time() - last_activity_ts > NO_INPUT_TIMEOUT_SEC:
                    await safe_send(websocket, {
                        "type": "assistant_message",
                        "text": "응답이 없어 안내를 종료할게요.",
                        "end_session": True,
                    })
                    await safe_close(websocket)
                    break

            message = await websocket.receive()

            # --------------------------------------------------
            # 🔁 프론트 → TTS 종료 알림
            # --------------------------------------------------
            if "text" in message:
                try:
                    msg = json.loads(message["text"])
                    if msg.get("type") == "tts_end":
                        io_state = "LISTENING"
                        collecting = False
                        pcm_buffer.clear()
                        prerun_task = None
                        speech_frame_count = 0
                        ignore_until_ts = time.time() + IGNORE_INPUT_AFTER_TTS_SEC
                        last_activity_ts = time.time()
                        continue
                except Exception:
                    pass

            # --------------------------------------------------
            # 🎧 오디오 프레임
            # --------------------------------------------------
            if "bytes" not in message:
                continue

            if io_state != "LISTENING":
                continue

            now = time.time()
            if now < ignore_until_ts:
                continue

            pcm = np.frombuffer(message["bytes"], dtype=np.float32)
            if pcm.size == 0:
                continue

            rms = float(np.sqrt(np.mean(pcm * pcm)))

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
                    speech_start_ts = now
                    last_non_silence_ts = now
                continue

            # --------------------------------------------------
            # 🎙 발화 수집
            # --------------------------------------------------
            pcm_buffer.append(pcm)

            if rms > SILENCE_RMS_THRESHOLD:
                last_non_silence_ts = now

            silence_time = now - last_non_silence_ts
            speech_duration = now - speech_start_ts

            # --------------------------------------------------
            # ⚡ STT pre-run
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

            # --------------------------------------------------
            # 🛑 발화 종료 판단
            # --------------------------------------------------
            if silence_time >= END_SILENCE_SEC or speech_duration >= MAX_SPEECH_SEC:
                collecting = False
                io_state = "THINKING"
                ignore_until_ts = time.time() + POST_SPEECH_IGNORE_SEC

                total_samples = sum(len(c) for c in pcm_buffer)
                if total_samples / SAMPLE_RATE < MIN_AUDIO_SEC:
                    pcm_buffer.clear()
                    prerun_task = None
                    io_state = "LISTENING"
                    last_activity_ts = time.time()
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

                if not is_meaningful_text(text):
                    io_state = "LISTENING"
                    last_activity_ts = time.time()
                    continue

                last_activity_ts = time.time()
                print(f"[STT] {text}")

                # --------------------------------------------------
                # 🧠 AppEngine
                # --------------------------------------------------
                result = app_state.app_engine.handle_text(text)

                reply_text = result.get("text", "")
                end_session = result.get("end_session", False)

                if reply_text:
                    io_state = "SPEAKING"
                    tts_url = synthesize(reply_text)

                    payload = dict(result)
                    payload["tts_url"] = tts_url

                    await safe_send(websocket, payload)

                if end_session:
                    await safe_close(websocket)
                    break

    except WebSocketDisconnect:
        print("[WS] ❌ Client disconnected")

    except Exception as e:
        print("[WS] 💥 Error:", repr(e))
        await safe_close(websocket)
