"use client"

import { useRef, useState } from "react"

type Status = "idle" | "listening" | "processing" | "speaking"

const STATUS_TEXT: Record<Status, string> = {
  idle: "무엇을 도와드릴까요?",
  listening: "말씀을 듣고 있어요",
  processing: "잠시만 기다려주세요",
  speaking: "안내를 시작할게요"
}

export default function Home() {
  const [status, setStatus] = useState<Status>("idle")
  const [userText, setUserText] = useState("")
  const [botText, setBotText] = useState("")
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  const toggleRecording = async () => {
    if (status !== "listening") {
      setStatus("listening")

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      mediaRecorderRef.current = recorder

      recorder.ondataavailable = (e) => {
        chunksRef.current.push(e.data)
      }

      recorder.onstop = async () => {
        setStatus("processing")

        const audioBlob = new Blob(chunksRef.current, { type: "audio/webm" })
        chunksRef.current = []

        const formData = new FormData()
        formData.append("audio", audioBlob)

        const res = await fetch("http://localhost:8000/voice", {
          method: "POST",
          body: formData
        })

        const result = await res.json()

        setUserText(result.user_text)
        setBotText(result.bot_text)

        if (result.tts_url) {
          setStatus("speaking")
          const audio = new Audio(result.tts_url)
          audio.onended = () => setStatus("idle")
          audio.play()
        } else {
          setStatus("idle")
        }
      }

      recorder.start()
    } else {
      mediaRecorderRef.current?.stop()
    }
  }

  const ringStyle = {
    idle: "from-emerald-300 to-sky-400",
    listening: "from-sky-400 to-blue-500 animate-pulse",
    processing: "from-amber-300 to-orange-400",
    speaking: "from-purple-400 to-pink-400"
  }[status]

  return (
    <main className="min-h-screen bg-gradient-to-br from-emerald-50 via-sky-50 to-white text-neutral-800 flex flex-col items-center justify-center px-8">

      {/* 브랜드 헤더 */}
      <header className="absolute top-14 text-center select-none">
        <h1 className="text-4xl font-semibold tracking-[0.28em] text-neutral-800/80">
          PARKING
        </h1>
        <p className="mt-2 text-xs tracking-[0.35em] text-neutral-400 uppercase">
          voice support
        </p>
      </header>

      {/* 상태 문구 */}
      <p className="mb-10 text-lg text-neutral-500">
        {STATUS_TEXT[status]}
      </p>

      {/* 마이크 버튼 */}
      <button
        onClick={toggleRecording}
        className={`
          relative w-44 h-44 rounded-full
          bg-gradient-to-br ${ringStyle}
          flex items-center justify-center
          shadow-xl
          transition-all duration-300
        `}
      >
        <div className="w-32 h-32 bg-white rounded-full flex items-center justify-center shadow-inner">
          <span className="text-4xl font-medium text-neutral-700">
            MIC
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
        버튼을 누르고 자연스럽게 말씀해 주세요<br />
        출차 · 요금 · 정산 문제를 도와드립니다
      </footer>
    </main>
  )
}
