"use client"

import { useEffect, useRef, useState } from "react"

type Status = "idle" | "listening" | "thinking" | "speaking"

const STATUS_TEXT: Record<Status, string> = {
  idle: "시작 버튼을 눌러 주세요",
  listening: "말씀을 듣고 있어요",
  thinking: "잠시만 기다려주세요",
  speaking: "안내를 시작할게요"
}

const WS_BASE =
  process.env.NEXT_PUBLIC_WS_BASE || "ws://localhost:8000/ws/voice"

export default function Home() {
  const [status, setStatus] = useState<Status>("idle")
  const [userText, setUserText] = useState("")
  const [botText, setBotText] = useState("")
  const [active, setActive] = useState(false)

  const wsRef = useRef<WebSocket | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  // ==================================================
  // WebSocket + 마이크 시작
  // ==================================================
  const startVoice = async () => {
    if (active) return

    setActive(true)
    setStatus("listening")
    setUserText("")
    setBotText("")

    // WebSocket 연결
    const ws = new WebSocket(WS_BASE)
    ws.binaryType = "arraybuffer"
    wsRef.current = ws

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.type === "bot_text") {
          setBotText(data.text)
          setStatus("speaking")
          // TTS는 서버 붙인 뒤 여기서 재생
          setTimeout(() => setStatus("listening"), 800)
        }
      } catch {
        // ignore
      }
    }

    ws.onclose = () => {
      stopVoice()
    }

    // 마이크 스트림
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    streamRef.current = stream

    const audioCtx = new AudioContext({ sampleRate: 16000 })
    audioCtxRef.current = audioCtx

    const source = audioCtx.createMediaStreamSource(stream)
    const processor = audioCtx.createScriptProcessor(4096, 1, 1)
    processorRef.current = processor

    processor.onaudioprocess = (e) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return

      const input = e.inputBuffer.getChannelData(0)
      wsRef.current.send(input.buffer)
    }

    source.connect(processor)
    processor.connect(audioCtx.destination)
  }

  // ==================================================
  // 종료 처리
  // ==================================================
  const stopVoice = () => {
    setActive(false)
    setStatus("idle")

    wsRef.current?.close()
    wsRef.current = null

    processorRef.current?.disconnect()
    processorRef.current = null

    audioCtxRef.current?.close()
    audioCtxRef.current = null

    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
  }

  const toggle = () => {
    active ? stopVoice() : startVoice()
  }

  const ringStyle = {
    idle: "from-emerald-300 to-sky-400",
    listening: "from-sky-400 to-blue-500 animate-pulse",
    thinking: "from-amber-300 to-orange-400",
    speaking: "from-purple-400 to-pink-400 animate-pulse"
  }[status]

  return (
    <main className="min-h-screen bg-gradient-to-br from-emerald-50 via-sky-50 to-white text-neutral-800 flex flex-col items-center justify-center px-8">

      {/* 브랜드 헤더 */}
      <header className="absolute top-14 text-center select-none">
        <h1 className="text-4xl font-semibold tracking-[0.28em] text-neutral-800/80">
          PARKING
        </h1>
        <p className="mt-2 text-xs tracking-[0.35em] text-neutral-400 uppercase">
          voice assistant
        </p>
      </header>

      {/* 상태 문구 */}
      <p className="mb-10 text-lg text-neutral-500">
        {STATUS_TEXT[status]}
      </p>

      {/* 중앙 버튼 */}
      <button
        onClick={toggle}
        className={`
          relative w-44 h-44 rounded-full
          bg-gradient-to-br ${ringStyle}
          flex items-center justify-center
          shadow-xl
          transition-all duration-300
        `}
      >
        <div className="w-32 h-32 bg-white rounded-full flex items-center justify-center shadow-inner">
          <span className="text-3xl font-semibold text-neutral-700">
            {active ? "STOP" : "START"}
          </span>
        </div>
      </button>

      {/* 대화 영역 */}
      <section className="mt-14 w-full max-w-xl space-y-5">
        {userText && (
          <div className="bg-white/70 backdrop-blur p-5 rounded-2xl shadow-sm">
            <p className="text-xs tracking-wide text-neutral-400 mb-2">
              VOICE INPUT
            </p>
            <p className="text-lg">{userText}</p>
          </div>
        )}

        {botText && (
          <div className="bg-emerald-100/70 backdrop-blur p-5 rounded-2xl shadow-sm">
            <p className="text-xs tracking-wide text-emerald-600 mb-2">
              SYSTEM RESPONSE
            </p>
            <p className="text-lg font-semibold">{botText}</p>
          </div>
        )}
      </section>

      {/* 하단 가이드 */}
      <footer className="absolute bottom-10 text-center text-sm text-neutral-500">
        시작을 누른 뒤 자유롭게 말씀해 주세요<br />
        출차 · 요금 · 정산 문제를 도와드립니다
      </footer>
    </main>
  )
}
