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
    "문의하실 내용이 있으시면\n저를 눌러주세요."
  )
  const [active, setActive] = useState(false)

  /* ===============================
     Refs
  =============================== */
  const wsRef = useRef<WebSocket | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  /* ===============================
     마이크 하드 차단 / 복구
  =============================== */
  const muteMicHard = () => {
    streamRef.current?.getAudioTracks().forEach(t => (t.enabled = false))
  }

  const unmuteMicHard = () => {
    streamRef.current?.getAudioTracks().forEach(t => (t.enabled = true))
  }

  /* ===============================
     음성 시작 (최초 1회)
  =============================== */
  const startVoice = async () => {
    if (active) return

    setActive(true)
    setStatus("listening")

    /* ---------- WebSocket ---------- */
    const ws = new WebSocket(WS_BASE)
    ws.binaryType = "arraybuffer"
    wsRef.current = ws

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)

        /* ===============================
           🧠 THINKING (서버 신호 ONLY)
        =============================== */
        if (data.type === "assistant_state" && data.state === "THINKING") {
          setStatus("thinking")
          setBubbleText("잠시만요…\n생각 중이에요.")
          return
        }

        /* ===============================
           🤖 실제 응답
        =============================== */
        if (data.type === "assistant_message") {
          const { text, tts_url, end_session } = data

          if (text) {
            setBubbleText(text)
          }

          if (tts_url) {
            muteMicHard()
            setStatus("speaking")

            const audio = new Audio(
              tts_url.startsWith("http")
                ? tts_url
                : `${API_BASE}${tts_url}`
            )

            audio.onended = () => {
              setStatus("listening")
              unmuteMicHard()

              wsRef.current?.send(
                JSON.stringify({ type: "tts_end" })
              )
            }

            audio.play()
          }

          if (end_session) {
            setStatus("idle")
            setActive(false)
            setBubbleText(
              "문의하실 내용이 있으시면\n저를 눌러주세요."
            )
          }
        }
      } catch (e) {
        console.error("[WS] parse error", e)
      }
    }

    /* ---------- Microphone ---------- */
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    })
    streamRef.current = stream

    const audioCtx = new AudioContext({ sampleRate: 16000 })
    audioCtxRef.current = audioCtx

    const source = audioCtx.createMediaStreamSource(stream)
    const processor = audioCtx.createScriptProcessor(4096, 1, 1)

    source.connect(processor)
    processor.connect(audioCtx.destination)

    processor.onaudioprocess = (e) => {
      if (statusRef.current !== "listening") return
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return

      wsRef.current.send(e.inputBuffer.getChannelData(0).buffer)
    }
  }

  /* ===============================
     UI
  =============================== */
  return (
    <main className="min-h-screen bg-gradient-to-br from-emerald-50 via-sky-50 to-white flex items-center justify-center px-6 text-neutral-800">

      <header className="absolute top-8 text-center select-none">
        <h1 className="text-4xl font-semibold tracking-[0.35em]">
          PARKMATE
        </h1>
        <p className="mt-1 text-xs tracking-[0.35em] text-neutral-400 uppercase">
          Parking Guidance Kiosk
        </p>
      </header>

      <div className="relative flex items-center">
        {/* 🤖 루미 */}
        <div
          onClick={startVoice}
          className="relative z-10 cursor-pointer select-none"
        >
          <div
            className={`
              relative w-56 h-40 rounded-[2.5rem] bg-white shadow-2xl
              flex items-center justify-center
              ${
                status === "speaking"
                  ? "animate-pulse"
                  : status === "thinking"
                  ? "animate-bounce"
                  : ""
              }
            `}
          >
            <div className="w-44 h-28 rounded-2xl bg-gradient-to-br from-emerald-300 to-sky-400 flex items-center justify-center gap-6">
              <span className="w-4 h-4 bg-white rounded-full" />
              <span className="w-4 h-4 bg-white rounded-full" />
            </div>
          </div>

          <p className="mt-4 text-center text-base text-neutral-500">
            지미 · 주차 안내 파트너
          </p>
        </div>

        {/* 💬 말풍선 */}
        <div
          className={`
            relative ml-6 -translate-y-12
            max-w-[520px]
            bg-white
            px-10 py-8
            rounded-[2.2rem]
            shadow-[0_20px_40px_rgba(0,0,0,0.12)]
            transition-all duration-300
            ${status === "thinking" ? "animate-pulse" : ""}
          `}
        >
          <div
            className="
              absolute
              left-[-14px]
              bottom-[28px]
              w-0 h-0
              border-t-[10px] border-t-transparent
              border-b-[10px] border-b-transparent
              border-r-[16px] border-r-white
            "
          />

          <p className="text-2xl font-semibold leading-relaxed whitespace-pre-line break-words">
            {bubbleText}
          </p>
        </div>
      </div>
    </main>
  )
}
