import os
import json
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI

load_dotenv()

app = FastAPI()

def make_aoai_client():
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

def llm_reply(user_text: str) -> str:
    resp = AOAI.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role":"system","content":"너는 주차장 고객상담 AI다. 한국어로 1~2문장으로 짧고 명확히 답한다. 필요하면 질문 1개만 한다."},
            {"role":"user","content":user_text},
        ],
        temperature=0.3,
        max_tokens=220,
    )
    return (resp.choices[0].message.content or "").strip()

def synth_wav(text: str) -> bytes:
    # 스피커 재생이 아니라 "메모리로 wav"를 뽑아서 프론트로 보내는 방식
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
    await ws.accept()

    # PushAudioInputStream: 브라우저 PCM16(16k mono) 넣는 스트림
    stream_format = speechsdk.audio.AudioStreamFormat(samples_per_second=16000, bits_per_sample=16, channels=1)
    push_stream = speechsdk.audio.PushAudioInputStream(stream_format)
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language = SPEECH_LANG

    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    loop = asyncio.get_event_loop()
    final_queue: asyncio.Queue[str] = asyncio.Queue()

    # 이벤트 핸들러(별도 스레드에서 콜백될 수 있어 loop.call_soon_threadsafe 사용)
    def on_recognizing(evt):
        text = evt.result.text or ""
        if text:
            loop.call_soon_threadsafe(asyncio.create_task, ws.send_text(json.dumps({"type":"partial","text":text})))

    def on_recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = (evt.result.text or "").strip()
            if text:
                loop.call_soon_threadsafe(final_queue.put_nowait, text)

    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)

    recognizer.start_continuous_recognition()

    async def final_consumer():
        while True:
            text = await final_queue.get()
            await ws.send_text(json.dumps({"type":"final","text":text}))
            # LLM
            bot = llm_reply(text)
            await ws.send_text(json.dumps({"type":"bot_text","text":bot}))
            # TTS -> WAV bytes 전송
            wav = synth_wav(bot)
            if wav:
                await ws.send_bytes(wav)

    consumer_task = asyncio.create_task(final_consumer())

    try:
        while True:
            msg = await ws.receive()
            if "text" in msg and msg["text"]:
                try:
                    ctrl = json.loads(msg["text"])
                    if ctrl.get("type") == "stop":
                        break
                except Exception:
                    pass

            if "bytes" in msg and msg["bytes"]:
                # 브라우저에서 온 PCM16 chunk
                push_stream.write(msg["bytes"])

    except WebSocketDisconnect:
        pass
    finally:
        try:
            push_stream.close()
        except Exception:
            pass
        recognizer.stop_continuous_recognition()
        consumer_task.cancel()
