import queue
import pyaudio


class MicrophoneStream:
    """
    발화 → 20ms 오디오 조각 생성
    오디오 바이트가 self._buff (Queue)에 저장됨
    """

    def __init__(self, rate: int, chunk: int):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self._audio_interface = None
        self._audio_stream = None
        self.closed = True
          # ✅ 추가: TTS 중 마이크를 "무음"으로 보내기 위한 플래그
        self.muted = False

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,   # 16bit PCM
            channels=1,               # mono
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        try:
            self._audio_stream.stop_stream()
            self._audio_stream.close()
        except Exception:
            pass
        self.closed = True
        self._buff.put(None)  # 종료 신호
        try:
            self._audio_interface.terminate()
        except Exception:
            pass

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        #  muted면 같은 길이의 무음 바이트로 덮어쓰기
        if self.muted and in_data:
            in_data = b"\x00" * len(in_data)

        # 마이크에서 들어온 프레임(bytes)을 큐에 저장
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        """
       Q_audio → Google STT로 전달
       Queue에서 오디오 조각 꺼냄
       여러 개 묶어서 하나의 byte stream으로 yield
        """
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return

            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)