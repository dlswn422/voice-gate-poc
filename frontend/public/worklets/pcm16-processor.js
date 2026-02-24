// 48k(or device rate) -> 16k mono PCM16로 다운샘플 후 chunk 단위로 메인으로 전달
class PCM16Processor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetRate = 16000;
    this._buffer = [];
  }

  // naive downsample: inputRate -> 16k
  downsampleTo16k(input, inputRate) {
    if (inputRate === this.targetRate) return input;

    const ratio = inputRate / this.targetRate;
    const newLength = Math.round(input.length / ratio);
    const output = new Float32Array(newLength);

    let offsetResult = 0;
    let offsetBuffer = 0;
    while (offsetResult < output.length) {
      const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
      let accum = 0, count = 0;
      for (let i = offsetBuffer; i < nextOffsetBuffer && i < input.length; i++) {
        accum += input[i];
        count++;
      }
      output[offsetResult] = count > 0 ? accum / count : 0;
      offsetResult++;
      offsetBuffer = nextOffsetBuffer;
    }
    return output;
  }

  floatToPCM16(float32) {
    const buffer = new ArrayBuffer(float32.length * 2);
    const view = new DataView(buffer);
    for (let i = 0; i < float32.length; i++) {
      let s = Math.max(-1, Math.min(1, float32[i]));
      // scale to int16
      view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return buffer;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const mono = input[0]; // first channel
    const down = this.downsampleTo16k(mono, sampleRate);
    const pcm16 = this.floatToPCM16(down);

    // 메인 스레드로 전송 (ArrayBuffer)
    this.port.postMessage(pcm16, [pcm16]);
    return true;
  }
}

registerProcessor("pcm16-processor", PCM16Processor);
