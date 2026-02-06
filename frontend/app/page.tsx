"use client"

import { useRef, useState } from "react"

type Status = "idle" | "listening" | "thinking" | "speaking"

const STATUS_TEXT: Record<Status, string> = {
  idle: "시작 버튼을 눌러 주세요",
  listening: "말씀을 듣고 있어요",
  thinking: "잠시만 기다려주세요",
  speaking: "안내를 시작할게요"
}

const WS_BASE = "ws://127.0.0.1:8000/ws/voice"
const API_BASE = "http://127.0.0.1:8000"

export default function Home() {
  /* ===============================
     상태(UI) + 상태 Ref(로직)
  =============================== */
  const [status, _setStatus] = useState<Status>("idle")
  const statusRef = useRef<Status>("idle")
  const setStatus = (s: Status) => {
    statusRef.current = s
    _setStatus(s)
  }

  const [botText, setBotText] = useState("")
  const [active, setActive] = useState(false)

  /* ===============================
     Refs
  =============================== */
  const wsRef = useRef<WebSocket | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  /* ===============================
     🎤 마이크 제어
  =============================== */
  const startMicGraph = () => {
    if (!audioCtxRef.current || !processorRef.current || !sourceRef.current) return
    sourceRef.current.connect(processorRef.current)
    processorRef.current.connect(audioCtxRef.current.destination)
  }

  const stopMicGraph = () => {
    try {
      sourceRef.current?.disconnect()
      processorRef.current?.disconnect()
    } catch {}
  }

  // 🔥 물리적 마이크 OFF/ON
  const muteMicHard = () => {
    streamRef.current?.getAudioTracks().forEach(t => (t.enabled = false))
  }
  const unmuteMicHard = () => {
    streamRef.current?.getAudioTracks().forEach(t => (t.enabled = true))
  }

  /* ===============================
     ▶️ 음성 시작
  =============================== */
  const startVoice = async () => {
    if (active) return

    setActive(true)
    setBotText("")
    setStatus("listening")

    // 1️⃣ WebSocket
    const ws = new WebSocket(WS_BASE)
    ws.binaryType = "arraybuffer"
    wsRef.current = ws

    ws.onopen = () => console.log("[WS] connected")

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.type === "bot_text") {
          setBotText(data.text)

          // 🔴 TTS 시작 → STT 완전 차단
          stopMicGraph()
          muteMicHard()
          setStatus("speaking")

          if (data.tts_url) {
            const audio = new Audio(
              data.tts_url.startsWith("http")
                ? data.tts_url
                : `${API_BASE}${data.tts_url}`
            )

            audio.onended = () => {
              // 🔔 백엔드에 TTS 종료 알림 (핵심)
              wsRef.current?.send(
                JSON.stringify({ type: "tts_end" })
              )

              // 잔향 방지 딜레이 후 listening 복귀
              setTimeout(() => {
                setStatus("listening")
                unmuteMicHard()
                startMicGraph()
              }, 400)
            }

            audio.play().catch(() => {
              wsRef.current?.send(
                JSON.stringify({ type: "tts_end" })
              )
              setStatus("listening")
              unmuteMicHard()
              startMicGraph()
            })
          } else {
            wsRef.current?.send(
              JSON.stringify({ type: "tts_end" })
            )
            setStatus("listening")
            unmuteMicHard()
            startMicGraph()
          }
        }
      } catch (e) {
        console.error("WS parse error", e)
      }
    }

    ws.onclose = () => stopVoice()

    // 2️⃣ 마이크
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }
    })
    streamRef.current = stream

    const audioCtx = new AudioContext({ sampleRate: 16000 })
    audioCtxRef.current = audioCtx

    const source = audioCtx.createMediaStreamSource(stream)
    sourceRef.current = source

    const processor = audioCtx.createScriptProcessor(4096, 1, 1)
    processorRef.current = processor

    processor.onaudioprocess = (e) => {
      // 🔥 실시간 제어는 반드시 ref로
      if (statusRef.current !== "listening") return
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return

      const input = e.inputBuffer.getChannelData(0)
      wsRef.current.send(input.buffer)
    }

    startMicGraph()
  }

  /* ===============================
     ⏹ 종료
  =============================== */
  const stopVoice = () => {
    setActive(false)
    setStatus("idle")

    wsRef.current?.close()
    wsRef.current = null

    stopMicGraph()
    muteMicHard()
    processorRef.current = null
    sourceRef.current = null

    audioCtxRef.current?.close()
    audioCtxRef.current = null

    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
  }

  const toggle = () => (active ? stopVoice() : startVoice())

  const ringStyle = {
    idle: "from-emerald-300 to-sky-400",
    listening: "from-sky-400 to-blue-500 animate-pulse",
    thinking: "from-amber-300 to-orange-400",
    speaking: "from-purple-400 to-pink-400 animate-pulse"
  }[status]

  return (
    <main className="min-h-screen bg-gradient-to-br from-emerald-50 via-sky-50 to-white flex flex-col items-center justify-center px-8 text-neutral-800">
      <header className="absolute top-14 text-center select-none">
        <h1 className="text-4xl font-semibold tracking-[0.28em] text-neutral-800/80">PARKING</h1>
        <p className="mt-2 text-xs tracking-[0.35em] text-neutral-400 uppercase">voice assistant</p>
      </header>

      <p className="mb-10 text-lg text-neutral-500">{STATUS_TEXT[status]}</p>

      <button
        onClick={toggle}
        className={`relative w-44 h-44 rounded-full bg-gradient-to-br ${ringStyle}
          flex items-center justify-center shadow-xl transition-all duration-300`}
      >
        <div className="w-32 h-32 bg-white rounded-full flex items-center justify-center shadow-inner">
          <span className="text-3xl font-semibold text-neutral-700">
            {active ? "STOP" : "START"}
          </span>
        </div>
      </button>

      <section className="mt-14 w-full max-w-xl space-y-5">
        {botText && (
          <div className="bg-emerald-100/70 backdrop-blur p-5 rounded-2xl shadow-sm">
            <p className="text-xs tracking-wide text-emerald-600 mb-2">SYSTEM RESPONSE</p>
            <p className="text-lg font-semibold">{botText}</p>
          </div>
        )}
      </section>

      <footer className="absolute bottom-10 text-center text-sm text-neutral-500">
        시작을 누른 뒤 자연스럽게 말씀해 주세요<br />
        출차 · 요금 · 정산 문제를 도와드립니다
      </footer>
    </main>
  )
}
