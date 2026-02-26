import os
import time
from dotenv import load_dotenv

import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI


def make_speech_recognizer():
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    lang = os.getenv("AZURE_SPEECH_LANGUAGE", "ko-KR")
    if not key or not region:
        raise RuntimeError("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION 가 .env에 없습니다.")

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_recognition_language = lang

    # (선택) 주차장 도메인 힌트 - 인식 품질에 도움
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    phrase_list = speechsdk.PhraseListGrammar.from_recognizer(recognizer)
    for p in ["차단기", "게이트", "문", "출구", "입구", "결제", "요금", "정기권", "영수증", "할인", "문이 안 열려요", "차단기가 안 열려요"]:
        phrase_list.addPhrase(p)

    return recognizer


def make_speech_synthesizer():
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    voice = os.getenv("AZURE_SPEECH_VOICE", "ko-KR-SunHiNeural")
    if not key or not region:
        raise RuntimeError("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION 가 .env에 없습니다.")

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_synthesis_voice_name = voice

    # ✅ 파일 저장 없이 스피커로 바로 출력
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    return speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)


def make_azure_openai_client():
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()

    if not endpoint or not api_key or not api_version:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY / AZURE_OPENAI_API_VERSION 누락")

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def llm_reply(client: AzureOpenAI, deployment: str, history: list[dict], user_text: str) -> str:
    messages = history + [{"role": "user", "content": user_text}]
    resp = client.chat.completions.create(
        model=deployment,  # ✅ deployment name
        messages=messages,
        temperature=0.3,
        max_tokens=220,
    )
    return (resp.choices[0].message.content or "").strip()


def main():
    load_dotenv()

    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
    if not deployment:
        raise RuntimeError("AZURE_OPENAI_DEPLOYMENT(배포 이름)이 .env에 없습니다.")

    recognizer = make_speech_recognizer()
    synthesizer = make_speech_synthesizer()
    client = make_azure_openai_client()

    # 최소 상담 톤
    history: list[dict] = [{
        "role": "system",
        "content": (
            "너는 주차장 고객상담 AI다. 한국어로 짧고 명확하게 안내한다. "
            "필요한 정보가 있으면 한 번에 1개만 질문한다. "
            "답변은 1~2문장으로, 너무 길게 말하지 않는다."
            "규칙: 입력 문장의 문장부호(?, !)는 음성 인식 자동 보정 결과일 수 있으므로 의도 해석 시 과도하게 반영하지 마라. "
            "추가 규칙: 너의 역할은 '주차장/차량 출입/결제/요금/차단기/정기권/등록/시설 고장' 관련 상담만 한다. "
            "사용자 발화가 주차장 운영과 무관한 일반 지식(의료/병원/법률/투자/연애/정치 등)으로 해석될 가능성이 있으면, "
            "그 방향으로 절대 답하지 말고 '주차장 문의인지'를 한 문장으로 확인 질문을 해라. "
            "예: '소음' 같은 단어는 병원/의료로 연결하지 말고, '차단기/기기/경고음/안내방송 소리' 같은 주차장 상황으로 먼저 재해석한다. "
            "그래도 주차장과 무관하면: '주차장 이용 관련 문의만 도와드릴 수 있어요. 어떤 주차장 문제이신가요?' 라고 답한다."
        )
    }]

    print("🎤 말하면 STT → LLM → TTS로 응답합니다. (종료: '종료'라고 말하기)")
    synthesizer.speak_text_async("안녕하세요. 무엇을 도와드릴까요?").get()

    while True:
        print("\n[LISTEN] 말씀하세요...")
        stt = recognizer.recognize_once_async().get()

        if stt.reason != speechsdk.ResultReason.RecognizedSpeech or not stt.text:
            print("[STT] 인식 실패/무응답")
            synthesizer.speak_text_async("잘 들리지 않았어요. 다시 말씀해 주세요.").get()
            continue

        user_text = stt.text.strip()
        print("[USER]", user_text)

        if "종료" in user_text:
            synthesizer.speak_text_async("테스트를 종료합니다.").get()
            break

        try:
            answer = llm_reply(client, deployment, history, user_text)
        except Exception as e:
            print("[LLM ERROR]", repr(e))
            synthesizer.speak_text_async("죄송합니다. 잠시 후 다시 시도해 주세요.").get()
            continue

        if not answer:
            answer = "죄송합니다. 다시 한 번 말씀해 주세요."

        print("[AI]", answer)

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": answer})

        synthesizer.speak_text_async(answer).get()
        time.sleep(0.1)


if __name__ == "__main__":
    main()
