"use client"

import { useRef, useState } from "react"

type Status = "idle" | "listening" | "thinking" | "speaking"

const WS_BASE = "ws://127.0.0.1:8000/ws/voice"
const API_BASE = "http://127.0.0.1:8000"

export default function Home() {
  /* ===============================
     상태
  =============================== */
  const [status, _setStatus] = useState<Status>("idle")
  const statusRef = useRef<Status>("idle")
  const setStatus = (s: Status) => {
    statusRef.current = s
    _setStatus(s)
  }

  const [bubbleText, setBubbleText] = useState(
    "화면을 터치하거나 말씀해 주세요"
  )
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
     마이크 제어
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

  const muteMicHard = () => {
    streamRef.current?.getAudioTracks().forEach(t => (t.enabled = false))
  }
  const unmuteMicHard = () => {
    streamRef.current?.getAudioTracks().forEach(t => (t.enabled = true))
  }

  /* ===============================
     음성 시작
  =============================== */
  const startVoice = async () => {
    if (active) return

    setActive(true)
    setStatus("listening")
    setBubbleText("말씀을 듣고 있어요")

    const ws = new WebSocket(WS_BASE)
    ws.binaryType = "arraybuffer"
    wsRef.current = ws

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.type === "bot_text") {
          stopMicGraph()
          muteMicHard()
          setStatus("speaking")
          setBubbleText(data.text)

          if (data.tts_url) {
            const audio = new Audio(
              data.tts_url.startsWith("http")
                ? data.tts_url
                : `${API_BASE}${data.tts_url}`
            )

            audio.onended = () => {
              setTimeout(() => {
                setStatus("listening")
                setBubbleText("다른 도움이 필요하시면 저를 눌러주세요.")
                unmuteMicHard()
                startMicGraph()
              }, 400)
            }

            audio.play().catch(() => {
              setStatus("listening")
              unmuteMicHard()
              startMicGraph()
            })
          }
        }
      } catch {}
    }

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
      if (statusRef.current !== "listening") return
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
      wsRef.current.send(e.inputBuffer.getChannelData(0).buffer)
    }

    startMicGraph()
  }

  if (!active && typeof window !== "undefined") {
    startVoice()
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-emerald-50 via-sky-50 to-white flex flex-col items-center justify-center px-6 text-neutral-800">

      {/* 로고 (위 여백 축소) */}
      <header className="absolute top-8 text-center select-none">
        <h1 className="text-4xl font-semibold tracking-[0.35em]">PARKMATE</h1>
        <p className="mt-1 text-xs tracking-[0.35em] text-neutral-400 uppercase">
          Parking Guidance Kiosk
        </p>
      </header>

      {/* 캐릭터 + 말풍선 */}
      <div className="flex items-start gap-10 max-w-5xl">

        {/* 캐릭터 (🔥 크게) */}
        <div className="flex flex-col items-center">
          <div
            className={`
              relative w-56 h-40 rounded-[2.5rem] bg-white shadow-2xl
              flex items-center justify-center
              ${status === "speaking" ? "animate-pulse" : ""}
            `}
          >
            {/* 얼굴 */}
            <div className="w-44 h-28 rounded-2xl bg-gradient-to-br from-emerald-300 to-sky-400 flex items-center justify-center gap-6">
              <span className="w-4 h-4 bg-white rounded-full" />
              <span className="w-4 h-4 bg-white rounded-full" />
            </div>

            {/* 마이크 */}
            <div className="absolute -bottom-4 w-9 h-9 rounded-full bg-emerald-400 shadow-md" />
          </div>

          <p className="mt-4 text-base text-neutral-500">
            지미 · 주차 안내 파트너
          </p>
        </div>

        {/* 말풍선 (자동 확장) */}
        <div className="relative max-w-2xl bg-white/90 backdrop-blur-xl px-10 py-8 rounded-[2.5rem] shadow-xl border border-emerald-200/40">
          <div className="absolute left-0 top-16 -translate-x-1/2 w-6 h-6 bg-white rotate-45 border-l border-b border-emerald-200/40" />
          <p className="text-2xl font-semibold leading-relaxed whitespace-pre-line">
            {bubbleText}
          </p>
        </div>
      </div>

      {/* 하단 문구 (위로 끌어올림) */}
      <footer className="absolute bottom-8 text-center text-sm text-neutral-600 leading-relaxed">
        주차장 이용 중 불편한 점이 있으신가요?<br />
        <span className="font-semibold text-neutral-700">
          PARKMATE가 더 나은 주차 경험을 도와드립니다.
        </span>
      </footer>
    </main>
  )
}