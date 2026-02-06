from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np
import time

import src.app_state as app_state
from src.speech.vad import VoiceActivityDetector
from src.speech.whisper_service import transcribe_pcm_chunks


router = APIRouter()


@router.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket):
    """
    WebSocket 기반 상시 음성 대화 엔드포인트

    흐름:
    - 브라우저에서 PCM(Float32) 오디오 스트림 수신
    - VAD로 발화 시작/종료 판단
    - 발화 종료 시 Whisper STT
    - AppEngine으로 텍스트 전달
    - 응답 텍스트를 WebSocket으로 반환
    """

    await websocket.accept()
    print("[WS] 🔌 Client connected")

    vad = VoiceActivityDetector()
    pcm_buffer: list[np.ndarray] = []

    last_voice_ts = 0.0
    collecting = False

    try:
        while True:
            # ==================================================
            # 1️⃣ 오디오 chunk 수신 (Float32Array)
            # ==================================================
            data = await websocket.receive_bytes()

            pcm = np.frombuffer(data, dtype=np.float32)

            if pcm.size == 0:
                continue

            # ==================================================
            # 2️⃣ VAD 처리
            # ==================================================
            is_speech = vad.is_speech(pcm)

            now = time.time()

            if is_speech:
                if not collecting:
                    collecting = True
                    pcm_buffer.clear()
                    print("[WS] 🎤 Speech started")

                pcm_buffer.append(pcm)
                last_voice_ts = now

            # ==================================================
            # 3️⃣ 발화 종료 판단 (무음 지속)
            # ==================================================
            if collecting and not is_speech:
                if now - last_voice_ts >= vad.end_silence_sec:
                    print("[WS] 🛑 Speech ended")

                    collecting = False

                    # ==================================================
                    # 4️⃣ STT (발화 단위)
                    # ==================================================
                    text = transcribe_pcm_chunks(
                        pcm_buffer,
                        whisper_model=app_state.whisper_model,
                    )

                    pcm_buffer.clear()

                    if not text:
                        continue

                    print(f"[STT] {text}")

                    # ==================================================
                    # 5️⃣ AppEngine 처리
                    # ==================================================
                    reply = app_state.app_engine.handle_text(text)

                    print(f"[BOT] {reply}")

                    # ==================================================
                    # 6️⃣ 응답 전송
                    # ==================================================
                    await websocket.send_json(
                        {
                            "type": "bot_text",
                            "text": reply,
                        }
                    )

    except WebSocketDisconnect:
        print("[WS] ❌ Client disconnected")

    except Exception as e:
        print("[WS] 💥 Error:", repr(e))
        await websocket.close()
