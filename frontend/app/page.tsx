"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import MicCards from "./MicCards"

type Status = "OFF" | "LISTENING" | "THINKING" | "SPEAKING"

const WS_URL = process.env.NEXT_PUBLIC_WS_URL!

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

  const isWsOpen = useMemo(() => wsRef.current?.readyState === WebSocket.OPEN, [isRunning])

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
        // 말 끝나면 다시 듣는 상태로
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          setStatus("LISTENING")
        } else {
          setStatus("OFF")
        }
      }
    } catch {
      // 재생 실패 시에도 상태 복구
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

    // close ws
    try {
      wsRef.current?.close()
    } catch {}
    wsRef.current = null

    // stop mic
    try {
      mediaStreamRef.current?.getTracks().forEach((t) => t.stop())
    } catch {}
    mediaStreamRef.current = null

    // close audio context
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
    // 1) WS 연결
    const ws = new WebSocket(WS_URL)
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
        const audioCtx = new AudioCtx()
        audioCtxRef.current = audioCtx

        const source = audioCtx.createMediaStreamSource(stream)

        // ScriptProcessor는 구식이지만 “빠르게 붙이기”엔 제일 간단함
        const processor = audioCtx.createScriptProcessor(4096, 1, 1)
        processorRef.current = processor

        processor.onaudioprocess = (e) => {
          const sock = wsRef.current
          if (!sock || sock.readyState !== WebSocket.OPEN) return

          // 브라우저 입력은 보통 48k float32
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
            // bot_text가 오면 곧바로 wav bytes가 올 수도 있고, 안 올 수도 있음
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
      // 사용자가 stop 누른 경우도 여기로 옴
      await stopAll()
    }

    wsRef.current = ws
  }

  const onToggle = async () => {
    if (isRunning) {
      await stopAll()
      return
    }
    // 시작
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

        <section className="mt-24 flex flex-col items-center">
          <div className="w-full">
            <MicCards isRunning={isRunning} status={status} onToggle={onToggle} />
          </div>

          {/* ✅ STT/LLM 상태 표시(연결 확인용) */}
          <div className="mt-8 w-full rounded-2xl border border-white/60 bg-white/70 p-6 shadow-sm backdrop-blur">
            <div className="text-sm font-semibold text-neutral-900">실시간 로그</div>

            <div className="mt-4 space-y-3 text-sm">
              <div className="flex gap-3">
                <div className="w-24 shrink-0 text-neutral-500">PARTIAL</div>
                <div className="text-neutral-800">{partialText || <span className="text-neutral-400">-</span>}</div>
              </div>

              <div className="flex gap-3">
                <div className="w-24 shrink-0 text-neutral-500">FINAL</div>
                <div className="text-neutral-800">{finalText || <span className="text-neutral-400">-</span>}</div>
              </div>

              <div className="flex gap-3">
                <div className="w-24 shrink-0 text-neutral-500">BOT</div>
                <div className="text-neutral-800">{botText || <span className="text-neutral-400">-</span>}</div>
              </div>

              <div className="pt-2 text-xs text-neutral-400">
                WS: {WS_URL} · 연결상태:{" "}
                {wsRef.current
                  ? ["CONNECTING", "OPEN", "CLOSING", "CLOSED"][wsRef.current.readyState] ?? "UNKNOWN"
                  : "NONE"}
              </div>
            </div>
          </div>

          <div className="mt-20 w-full grid grid-cols-1 gap-8 sm:grid-cols-3">
            <GuideChip title="사용 방법" items={["마이크 시작 누르기", "문의하기", "안내 듣기"]} icon="🧭" />
            <GuideChip title="지원 항목" items={["요금/정산", "출차/입차", "등록/오류 안내"]} icon="🧩" />
            <GuideChip
              title="안내"
              items={[
                "음성 인식 후 자동으로 안내 시작",
                "결제 오류 시 사유 안내 가능",
                "필요 시 직원 호출이 가능",
              ]}
              icon="ℹ️"
            />
          </div>

          <p className="mt-12 text-center text-xs text-neutral-400">
            * 버튼을 누르면 브라우저 마이크를 캡처해 WebSocket으로 백엔드에 전송합니다.
          </p>
        </section>
      </div>
    </main>
  )
}

function GuideChip({
  title,
  items,
  icon,
}: {
  title: string
  items: string[]
  icon: string
}) {
  return (
    <div className="min-h-[180px] rounded-2xl border border-white/60 bg-white/70 p-7 shadow-sm backdrop-blur">
      <div className="flex items-center gap-3">
        <span className="text-xl" aria-hidden="true">
          {icon}
        </span>
        <div className="text-base font-semibold text-neutral-900">{title}</div>
      </div>

      <ul className="mt-5 space-y-3 text-sm text-neutral-600">
        {items.map((t) => (
          <li key={t} className="flex items-start gap-3">
            <span className="mt-[7px] inline-block size-2 rounded-full bg-neutral-300" />
            <span>{t}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}