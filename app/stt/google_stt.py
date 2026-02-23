from google.oauth2 import service_account
from google.cloud import speech


def run_google_stt_streaming(
    json_key_path: str,
    rate: int,
    lang: str,
    audio_generator,
):
    """
    Google STT 스트리밍을 실행하고,
    partial / final 텍스트를 yield하는 제너레이터.

    반환 이벤트 형태:
      ("partial", "문 열어")
      ("final",   "문 열어")
    """
    creds = service_account.Credentials.from_service_account_file(json_key_path)
    client = speech.SpeechClient(credentials=creds)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=rate,
        language_code=lang,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
        single_utterance=False,
    )

    requests = (
        speech.StreamingRecognizeRequest(audio_content=content)
        for content in audio_generator
    )

    responses = client.streaming_recognize(streaming_config, requests)

    for response in responses:
        for result in response.results:
            transcript = result.alternatives[0].transcript.strip()
            if not transcript:
                continue

            if result.is_final:
                yield ("final", transcript)
            else:
                yield ("partial", transcript)