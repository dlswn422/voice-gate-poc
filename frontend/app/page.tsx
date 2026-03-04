"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import MicCards from "./MicCard"
import RealtimeLogCard from "./RealtimeLogCard"
import GuideCards from "./GuideCards"
import { useAvatar } from "./useAvatar"

type Status = "OFF" | "LISTENING" | "THINKING" | "SPEAKING"

const ENV_WS = process.env.NEXT_PUBLIC_BACKEND_WS?.trim()
const WS_URL = ENV_WS || "ws://localhost:8000/ws/voice"
const USE_AVATAR = process.env.NEXT_PUBLIC_USE_AVATAR === "true"

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
  const zeroGainRef = useRef<GainNode | null>(null)

  // ✅ Viewer iframe ref
  const viewerRef = useRef<HTMLIFrameElement | null>(null)
  const avatar = useAvatar()

  const isWsOpen = useMemo(() => wsRef.current?.readyState === WebSocket.OPEN, [isRunning])

  const playWavBytesNonAvatar = async (wavBytes: ArrayBuffer) => {
    try {
      setStatus("SPEAKING")
      const blob = new Blob([wavBytes], { type: "audio/wav" })
      const url = URL.createObjectURL(blob)
      const audio = new Audio(url)
      await audio.play()
      audio.onended = () => {
        URL.revokeObjectURL(url)
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) setStatus("LISTENING")
        else setStatus("OFF")
      }
    } catch {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) setStatus("LISTENING")
      else setStatus("OFF")
    }
  }

  const stopAll = async () => {
    setIsRunning(false)
    setStatus("OFF")

    try {
      const ws = wsRef.current
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "stop" }))
    } catch {}

    try {
      const ws = wsRef.current
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) ws.close()
    } catch {}
    wsRef.current = null

    try {
      mediaStreamRef.current?.getTracks().forEach((t) => t.stop())
    } catch {}
    mediaStreamRef.current = null

    try {
      processorRef.current?.disconnect()
    } catch {}
    processorRef.current = null

    try {
      zeroGainRef.current?.disconnect()
    } catch {}
    zeroGainRef.current = null

    try {
      await audioCtxRef.current?.close()
    } catch {}
    audioCtxRef.current = null

    try {
      await avatar.stop()
    } catch {}
  }

  const startAll = async () => {
    if (!ENV_WS && typeof window !== "undefined" && window.location.protocol === "https:") {
      console.error("❌ NEXT_PUBLIC_BACKEND_WS is missing on HTTPS (production).")
    }

    // ✅ 아바타(=viewer iframe) 준비
    if (USE_AVATAR) {
      try {
        await avatar.start(viewerRef)
      } catch (e) {
        console.error("[VIEWER] start failed:", e)
      }
    }

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws
    ws.binaryType = "arraybuffer"

    ws.onopen = async () => {
      setIsRunning(true)
      setStatus("LISTENING")

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        })
        mediaStreamRef.current = stream

        const AudioCtx = window.AudioContext || (window as any).webkitAudioContext
        const audioCtx: AudioContext = new AudioCtx()
        audioCtxRef.current = audioCtx

        const source = audioCtx.createMediaStreamSource(stream)

        // ✅ ScriptProcessor가 멈추는 브라우저 대응: 0 gain으로 destination 연결(무음)
        const zero = audioCtx.createGain()
        zero.gain.value = 0
        zeroGainRef.current = zero
        zero.connect(audioCtx.destination)

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
        processor.connect(zero) // ✅ silent output path
      } catch (err) {
        console.error("mic start failed:", err)
        await stopAll()
      }
    }

    ws.onmessage = async (ev) => {
      // text json
      if (typeof ev.data === "string") {
        try {
          const msg = JSON.parse(ev.data)

          if (msg.type === "partial") setPartialText(msg.text ?? "")

          if (msg.type === "final") {
            setFinalText(msg.text ?? "")
            setPartialText("")
            setStatus("THINKING")
          }

          if (msg.type === "bot_text") {
            setBotText(msg.text ?? "")
            // ✅ TTS는 서버 WAV로 옴 (여기서 추가로 호출하지 않음)
          }

          if (msg.type === "barge_in") {
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) setStatus("LISTENING")
          }
        } catch {}
        return
      }

      // wav bytes
      const handleWav = async (ab: ArrayBuffer) => {
        if (USE_AVATAR) {
          try {
            setStatus("SPEAKING")
            await avatar.playWav(ab) // ✅ 오디오 재생 + mouth(입) 신호 전송
          } finally {
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) setStatus("LISTENING")
            else setStatus("OFF")
          }
        } else {
          await playWavBytesNonAvatar(ab)
        }
      }

      if (ev.data instanceof ArrayBuffer) {
        await handleWav(ev.data)
        return
      }

      if (ev.data instanceof Blob) {
        const ab = await ev.data.arrayBuffer()
        await handleWav(ab)
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

  useEffect(() => {
    return () => {
      stopAll()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <main className="min-h-screen px-6 pb-10">
      <div className="w-full max-w-[1400px] mx-auto">
        <header className="pt-6 text-center select-none">
          <h1 className="text-4xl font-semibold tracking-[0.35em]">TABLEMATE</h1>
          <p className="mt-1 text-xs tracking-[0.35em] text-neutral-400 uppercase">DINING GUIDANCE KIOSK</p>
        </header>

        <section className="mt-6">
          <div className="grid grid-cols-12 gap-6 items-start">
            <div className="col-span-12 md:col-span-3">
              <GuideCards />
            </div>

            <div className="col-span-12 md:col-span-6 flex flex-col gap-4">
              {/* Avatar */}
              <div className="h-[540px] rounded-3xl border border-white/70 bg-white/50 shadow-xl backdrop-blur overflow-hidden">
                <div className="px-6 py-4 flex items-center justify-between">
                  <div className="text-sm font-semibold text-neutral-900">Avatar</div>
                  <div className="text-xs text-neutral-400">{USE_AVATAR ? "ON" : "OFF"}</div>
                </div>

                <div className="px-6 pb-6 h-[calc(100%-56px)]">
                  {/* ✅ 공식 Cubism Demo 엔진을 iframe으로 렌더 */}
                  <iframe
                    ref={viewerRef}
                    src="/live2d_viewer/index.html"
                    className="w-full h-full rounded-2xl bg-black/10"
                    style={{
                      border: "0",
                      background: "#111",
                      // ✅ 터치 필요없다 했으니 이벤트 차단(에러/드래그 방지)
                      pointerEvents: "none",
                    }}
                  />
                </div>
              </div>

              {/* Mic */}
              <div className="h-[210px] overflow-hidden">
                <MicCards isRunning={isRunning} status={status} onToggle={onToggle} />
              </div>
            </div>

            {/* Right */}
            <div className="col-span-12 md:col-span-3 flex flex-col gap-4">
              <div className="h-[250px]">
                <RealtimeLogCard
                  partialText={partialText}
                  finalText={finalText}
                  botText={botText}
                  wsUrl={WS_URL}
                  wsRef={wsRef}
                  isWsOpen={isWsOpen}
                  className="h-full"
                  scroll
                />
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  )
}