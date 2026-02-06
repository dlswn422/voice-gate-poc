from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np
import time

import src.app_state as app_state
from src.speech.vad import VoiceActivityDetector
from src.speech.whisper_service import transcribe_pcm_chunks
from src.speech.tts import synthesize  # gTTS

router = APIRouter()


@router.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket):
    """
    WebSocket 기반 상시 음성 대화 엔드포인트

    흐름:
    - Float32 PCM 스트림 수신
    - VAD 발화 감지
    - STT
    - AppEngine
    - gTTS → static 파일 생성
    - text + tts_url 전송
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
            # 1️⃣ 오디오 chunk 수신
            # ==================================================
            data = await websocket.receive_bytes()
            pcm = np.frombuffer(data, dtype=np.float32)

            if pcm.size == 0:
                continue

            now = time.time()

            # ==================================================
            # 2️⃣ VAD 판단
            # ==================================================
            is_speech = vad.is_speech(pcm)

            if is_speech:
                if not collecting:
                    collecting = True
                    pcm_buffer.clear()
                    print("[WS] 🎤 Speech started")

                pcm_buffer.append(pcm)
                last_voice_ts = now

            # ==================================================
            # 3️⃣ 발화 종료 판단
            # ==================================================
            if collecting and not is_speech:
                if now - last_voice_ts >= vad.end_silence_sec:
                    print("[WS] 🛑 Speech ended")
                    collecting = False

                    # ==================================================
                    # 4️⃣ STT
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
                    # 5️⃣ AppEngine
                    # ==================================================
                    reply = app_state.app_engine.handle_text(text)
                    print(f"[BOT] {reply}")

                    # ==================================================
                    # 6️⃣ TTS (gTTS → static/tts/*.mp3)
                    # ==================================================
                    tts_url = synthesize(reply)
                    # 예: /static/tts/abcd1234.mp3

                    # ==================================================
                    # 7️⃣ 프론트로 전송 (🔥 타입 중요)
                    # ==================================================
                    await websocket.send_json(
                        {
                            "type": "bot_text",   # 🔥 프론트와 맞춤
                            "text": reply,
                            "tts_url": tts_url,
                        }
                    )

    except WebSocketDisconnect:
        print("[WS] ❌ Client disconnected")

    except Exception as e:
        print("[WS] 💥 Error:", repr(e))
        await websocket.close()