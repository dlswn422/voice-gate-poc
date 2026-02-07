"use client"

import { useRef, useState } from "react"

/* ===============================
   Types
=============================== */
type Status = "idle" | "listening" | "thinking" | "speaking"
type Intent =
  | "EXIT"
  | "ENTRY"
  | "PAYMENT"
  | "REGISTRATION"
  | "TIME_PRICE"
  | "FACILITY"
  | "NONE"

/* ===============================
   Constants
=============================== */
const WS_BASE = "ws://127.0.0.1:8000/ws/voice"
const API_BASE = "http://127.0.0.1:8000"

const INTENT_UI_KEYWORDS: Record<Intent, string[]> = {
  EXIT: ["차단기 안 열림", "출차 안 됨", "차량 인식 안 됨", "출구에서 멈춤", "기타", "관리실 호출"],
  ENTRY: ["입차 안 됨", "차단기 안 열림", "차량 인식 안 됨", "만차로 표시됨", "기타", "관리실 호출"],
  PAYMENT: ["결제 안 됨", "카드 오류", "요금 이상", "결제 방법 문의", "기타", "관리실 호출"],
  REGISTRATION: ["차량 등록", "방문자 등록", "등록 방법 문의", "등록했는데 안 됨", "기타", "관리실 호출"],
  TIME_PRICE: ["주차 시간 문의", "요금 문의", "할인 적용 문의", "요금 기준", "기타", "관리실 호출"],
  FACILITY: ["기기 멈춤", "화면 안 보임", "버튼 안 됨", "차단기 이상", "기타", "관리실 호출"],
  NONE: ["출차 관련", "입차 관련", "결제 관련", "방문등록 관련", "기타 문의", "관리실 호출"],
}

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
    "문의하실 내용이 있으시면\n저를 누르고 말씀해주세요."
  )

  const [active, setActive] = useState(false)
  const [showKeywords, setShowKeywords] = useState(false)
  const [currentIntent, setCurrentIntent] = useState<Intent | null>(null)

  /* ===============================
     Refs
  =============================== */
  const wsRef = useRef<WebSocket | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

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
    setBubbleText("말씀해 주세요.")
    setShowKeywords(false)
    setCurrentIntent(null)

    const ws = new WebSocket(WS_BASE)
    ws.binaryType = "arraybuffer"
    wsRef.current = ws

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)

        if (data.type === "assistant_state" && data.state === "THINKING") {
          setStatus("thinking")
          setBubbleText("잠시만요…\n생각 중이에요.")
          setShowKeywords(false)
          return
        }

        if (data.type === "assistant_message") {
          const { text, tts_url, end_session, one_turn, intent } = data

          if (text) setBubbleText(text)

          if (one_turn && intent) {
            setShowKeywords(true)
            setCurrentIntent(intent)
          } else {
            setShowKeywords(false)
            setCurrentIntent(null)
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
              wsRef.current?.send(JSON.stringify({ type: "tts_end" }))
            }

            audio.play()
          }

          if (end_session) {
            setStatus("idle")
            setActive(false)
            setBubbleText("문의하실 내용이 있으시면\n저를 누르고 말씀해주세요.")
            setShowKeywords(false)
            setCurrentIntent(null)
          }
        }
      } catch (e) {
        console.error("[WS] parse error", e)
      }
    }

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

  const showIdleKeywords = status === "idle" && !active

  /* ===============================
     UI
  =============================== */
  return (
    <main className="min-h-screen bg-gradient-to-br from-emerald-50 via-sky-50 to-white flex items-center justify-center px-6 text-neutral-800 font-[Pretendard]">

      <header className="absolute top-8 text-center select-none">
        <h1 className="text-4xl font-semibold tracking-[0.35em]">
          PARKMATE
        </h1>
        <p className="mt-1 text-xs tracking-[0.35em] text-neutral-400 uppercase">
          Parking Guidance Kiosk
        </p>
      </header>

      <div className="relative flex items-center">
        {/* 🤖 지미 */}
        <div
          onClick={startVoice}
          className={`
            relative z-10 cursor-pointer select-none
            ${status === "thinking" ? "animate-bounce" : ""}
          `}
        >
          <div className="w-56 h-40 rounded-[2.5rem] bg-white shadow-2xl flex items-center justify-center">
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
        <div className="relative ml-6 -translate-y-12 max-w-[520px] bg-white px-10 py-8 rounded-[2.2rem] shadow-[0_20px_40px_rgba(0,0,0,0.12)]">
          {/* 말풍선 꼬리 */}
          <div
            className="
              absolute
              left-[-14px]
              bottom-1/2
              -translate-y-1/2
              w-0 h-0
              border-t-[10px] border-t-transparent
              border-b-[10px] border-b-transparent
              border-r-[16px] border-r-white
            "
          />

          <p className="text-[22px] font-medium leading-relaxed whitespace-pre-line">
            {bubbleText}
          </p>

          {(showKeywords && currentIntent) || showIdleKeywords ? (
            <>
              <p className="mt-6 text-sm text-neutral-500">
                어떤 문의를 도와드릴까요?
              </p>

              <div className="mt-4 grid grid-cols-2 gap-3">
                {(showKeywords && currentIntent
                  ? INTENT_UI_KEYWORDS[currentIntent]
                  : INTENT_UI_KEYWORDS.NONE
                ).map((kw) => (
                  <button
                    key={kw}
                    onClick={() => {
                      wsRef.current?.send(
                        JSON.stringify({ type: "ui_keyword", text: kw })
                      )
                      setShowKeywords(false)
                    }}
                    className="
                      py-3 px-4
                      rounded-full
                      border border-neutral-300
                      bg-white
                      text-[16px]
                      font-semibold
                      text-neutral-800
                      hover:bg-neutral-100
                      active:scale-[0.97]
                      transition
                    "
                  >
                    {kw}
                  </button>
                ))}
              </div>
            </>
          ) : null}
        </div>
      </div>
    </main>
  )
}
