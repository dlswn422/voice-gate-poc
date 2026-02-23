import io
from google.oauth2 import service_account
from google.cloud import speech

client_file = 'speech_demo.json'  # JSON 파일명
credentials = service_account.Credentials.from_service_account_file(client_file)
client = speech.SpeechClient(credentials=credentials)

# 테스트 오디오 파일
audio_file = '사무실1층.wav'  

with io.open(audio_file, 'rb') as f:
    content = f.read()
    audio = speech.RecognitionAudio(content=content)

config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=48000,   # wav 파일에 맞게 수정
    language_code='ko-KR'      # 한국어면 ko-KR
)

response = client.recognize(config=config, audio=audio)

for result in response.results:
    print("Transcript:", result.alternatives[0].transcript)
