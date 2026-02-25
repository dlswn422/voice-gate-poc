"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import MicCards from "./MicCard"
import RealtimeLogCard from "./RealtimeLogCard"
import GuideCards from "./GuideCards"

type Status = "OFF" | "LISTENING" | "THINKING" | "SPEAKING"

// ===============================
// WS URL (Production은 env 필수)
// ===============================
const ENV_WS = process.env.NEXT_PUBLIC_BACKEND_WS?.trim()
const WS_URL = ENV_WS || "ws://localhost:8000/ws/voice"

// ===============================
// Audio helpers (Float32 48k -> PCM16 16k)
// ===============================
function downsampleBuffer(input: Float32Array, inRate: number, outRate: number) {
  if (outRate === inRate) return input
  if (outRate > inRate) throw new Error("outRate must be <= inRate")

  const ratio = inRate / outRate
  const newLen = Math.round(input.length / ratio)
  const result = new Float32Array(newLen)

  let offsetResult = 0
  let offsetInput = 0
  while (offsetResult < result.length) {
    const nextOffsetInput = Math.round((offsetResult + 1) * ratio)
    let sum = 0
    let count = 0
    for (let i = offsetInput; i < nextOffsetInput && i < input.length; i++) {
      sum += input[i]
      count++
    }
    result[offsetResult] = count > 0 ? sum / count : 0
    offsetResult++
    offsetInput = nextOffsetInput
  }
  return result
}

function floatTo16BitPCM(input: Float32Array) {
  const buffer = new ArrayBuffer(input.length * 2)
  const view = new DataView(buffer)
  for (let i = 0; i < input.length; i++) {
    let s = Math.max(-1, Math.min(1, input[i]))
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return buffer
}

export default function Home() {
  const [isRunning, setIsRunning] = useState(false)
  const [status, setStatus] = useState<Status>("OFF")

  const [partialText, setPartialText] = useState("")
  const [finalText, setFinalText] = useState("")
  const [botText, setBotText] = useState("")

  const wsRef = useRef<WebSocket | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)

  // (참고) useMemo에 ref.current 넣어봤자 렌더 트리거가 아니라 의미가 약함.
  // 그래도 디버깅 용도로 남겨둠.
  const isWsOpen = useMemo(
    () => wsRef.current?.readyState === WebSocket.OPEN,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [isRunning]
  )

  // WAV 재생용(서버가 send_bytes로 보냄)
  const playWavBytes = async (wavBytes: ArrayBuffer) => {
    try {
      setStatus("SPEAKING")
      const blob = new Blob([wavBytes], { type: "audio/wav" })
      const url = URL.createObjectURL(blob)
      const audio = new Audio(url)
      await audio.play()
      audio.onended = () => {
        URL.revokeObjectURL(url)
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          setStatus("LISTENING")
        } else {
          setStatus("OFF")
        }
      }
    } catch {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) setStatus("LISTENING")
      else setStatus("OFF")
    }
  }

  const stopAll = async () => {
    // UI 먼저 OFF
    setIsRunning(false)
    setStatus("OFF")

    // WS stop signal
    try {
      const ws = wsRef.current
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "stop" }))
      }
    } catch {}

    // close ws (OPEN/CONNECTING에서만)
    try {
      const ws = wsRef.current
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        ws.close()
      }
    } catch {}
    wsRef.current = null

    // stop mic
    try {
      mediaStreamRef.current?.getTracks().forEach((t) => t.stop())
    } catch {}
    mediaStreamRef.current = null

    // close audio graph
    try {
      processorRef.current?.disconnect()
    } catch {}
    processorRef.current = null

    try {
      await audioCtxRef.current?.close()
    } catch {}
    audioCtxRef.current = null
  }

  const startAll = async () => {
    // ✅ env가 없으면 바로 티나게 (배포에서 특히 중요)
    if (!ENV_WS && typeof window !== "undefined" && window.location.protocol === "https:") {
      console.error("❌ NEXT_PUBLIC_BACKEND_WS is missing on HTTPS (production).")
    }

    // 1) WS 연결
    const ws = new WebSocket(WS_URL)
    wsRef.current = ws // ✅ 아주 중요: onopen 전에 ref 먼저 세팅 (audio loop에서 ref 참조함)
    ws.binaryType = "arraybuffer"

    ws.onopen = async () => {
      setIsRunning(true)
      setStatus("LISTENING")

      // 2) 마이크 캡처 + PCM16(16k)로 변환해서 ws.send(bytes)
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        })
        mediaStreamRef.current = stream

        const AudioCtx = window.AudioContext || (window as any).webkitAudioContext
        const audioCtx: AudioContext = new AudioCtx()
        audioCtxRef.current = audioCtx

        const source = audioCtx.createMediaStreamSource(stream)

        // ScriptProcessor는 구식이지만 “빠르게 붙이기”엔 제일 간단함
        const processor = audioCtx.createScriptProcessor(4096, 1, 1)
        processorRef.current = processor

        processor.onaudioprocess = (e) => {
          const sock = wsRef.current
          if (!sock || sock.readyState !== WebSocket.OPEN) return

          const input = e.inputBuffer.getChannelData(0)
          const inRate = e.inputBuffer.sampleRate
          const down = downsampleBuffer(input, inRate, 16000)
          const pcm16 = floatTo16BitPCM(down)
          sock.send(pcm16)
        }

        source.connect(processor)
        processor.connect(audioCtx.destination)
      } catch (err) {
        // 마이크 권한/장치 오류
        await stopAll()
      }
    }

    ws.onmessage = async (ev) => {
      // 3) 서버 응답 처리: JSON(text) or WAV(bytes)
      if (typeof ev.data === "string") {
        try {
          const msg = JSON.parse(ev.data)
          if (msg.type === "partial") {
            setPartialText(msg.text ?? "")
          }
          if (msg.type === "final") {
            setFinalText(msg.text ?? "")
            setPartialText("")
            setStatus("THINKING")
          }
          if (msg.type === "bot_text") {
            setBotText(msg.text ?? "")
          }
        } catch {
          // ignore
        }
        return
      }

      // binary (wav)
      if (ev.data instanceof ArrayBuffer) {
        await playWavBytes(ev.data)
        return
      }

      // 일부 브라우저는 Blob으로 올 수 있음
      if (ev.data instanceof Blob) {
        const ab = await ev.data.arrayBuffer()
        await playWavBytes(ab)
      }
    }

    ws.onerror = async () => {
      await stopAll()
    }

    ws.onclose = async () => {
      await stopAll()
    }
  }

  const onToggle = async () => {
    if (isRunning) {
      await stopAll()
      return
    }

    setPartialText("")
    setFinalText("")
    setBotText("")
    await startAll()
  }

  // 페이지 이탈 시 정리
  useEffect(() => {
    return () => {
      stopAll()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <main className="min-h-screen px-6 flex justify-center">
      <div className="w-full max-w-[1200px]">
        <header className="pt-10 text-center select-none">
          <h1 className="text-4xl font-semibold tracking-[0.35em]">PARKMATE</h1>
          <p className="mt-1 text-xs tracking-[0.35em] text-neutral-400 uppercase">
            Parking Guidance Kiosk
          </p>
        </header>

        <section className="mt-14 flex flex-col items-center">
          <div className="w-full">
            <MicCards isRunning={isRunning} status={status} onToggle={onToggle} />
          </div>

          <RealtimeLogCard
            partialText={partialText}
            finalText={finalText}
            botText={botText}
            wsUrl={WS_URL}
            wsRef={wsRef}
            isWsOpen={isWsOpen}
          />

          <GuideCards />

          <p className="mt-8 text-center text-xs text-neutral-400">
            * 버튼을 누르면 브라우저 마이크를 캡처해 WebSocket으로 백엔드에 전송합니다.
          </p>
        </section>
      </div>
    </main>
  )
}