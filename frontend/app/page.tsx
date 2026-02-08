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
  ENTRY: ["입차 안 됨", "차단기 안 열림", "차량 인식 안 됨", "만차로 표시됨", "방문자 등록", "관리실 호출"],
  PAYMENT: ["결제 안 됨", "카드 오류", "요금 이상", "결제 방법 문의", "기타", "관리실 호출"],
  REGISTRATION: ["차량 등록", "방문자 등록", "등록 방법 문의", "등록했는데 안 됨", "기타", "관리실 호출"],
  TIME_PRICE: ["주차 시간 문의", "요금 문의", "할인 적용 문의", "요금 기준", "기타", "관리실 호출"],
  FACILITY: ["기기 멈춤", "화면 안 보임", "버튼 안 됨", "차단기 이상", "기타", "관리실 호출"],
  NONE: ["출차 관련", "입차 관련", "결제 관련", "방문등록 관련", "기타 문의", "관리실 호출"],
}

export default function Home() {
  /* ===============================
     State
  =============================== */
  const [status, _setStatus] = useState<Status>("idle")
  const statusRef = useRef<Status>("idle")
  const setStatus = (s: Status) => {
    statusRef.current = s
    _setStatus(s)
  }

  const [bubbleText, setBubbleText] = useState("어떤 문의가 있으신가요?")
  const [active, setActive] = useState(false)
  const [showAdminPopup, setShowAdminPopup] = useState(false)

  /* ===============================
     Refs
  =============================== */
  const wsRef = useRef<WebSocket | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const plateInputRef = useRef<HTMLInputElement | null>(null)

  /* ===============================
     Mic control
=============================== */
  const muteMicHard = () => {
    streamRef.current?.getAudioTracks().forEach(t => (t.enabled = false))
  }

  const unmuteMicHard = () => {
    streamRef.current?.getAudioTracks().forEach(t => (t.enabled = true))
  }

  /* ===============================
     Voice WS Start
=============================== */
  const startVoice = async () => {
    if (active) return

    setActive(true)
    setStatus("listening")

    const ws = new WebSocket(WS_BASE)
    ws.binaryType = "arraybuffer"
    wsRef.current = ws

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)

        /* =================================================
           FIX 1️⃣ assistant_state 대칭 처리 (핵심 수정)
           - THINKING만 처리하던 기존 버그 수정
           - LISTENING / SPEAKING도 동일하게 반영
        ================================================= */
        if (data.type === "assistant_state") {
          if (data.state === "THINKING") {
            setStatus("thinking")
            setBubbleText("잠시만요…\n확인 중이에요.")
          }

          if (data.state === "LISTENING") {
            setStatus("listening")
            // bubbleText는 유지 (서버가 새 메시지를 주지 않았기 때문)
          }

          if (data.state === "SPEAKING") {
            setStatus("speaking")
          }

          return
        }

        /* ===============================
           기존 assistant_message 로직
           (변경 없음)
        =============================== */
        if (data.type === "assistant_message") {
          const { text, tts_url, end_session, system_action } = data

          if (system_action === "CALL_ADMIN") {
            muteMicHard()
            setShowAdminPopup(true)

            setTimeout(() => {
              setShowAdminPopup(false)
              setActive(false)
              setStatus("idle")
              setBubbleText("어떤 문의가 있으신가요?")
            }, 1800)
            return
          }

          if (text) setBubbleText(text)

          if (tts_url) {
            muteMicHard()
            setStatus("speaking")

            const audio = new Audio(
              tts_url.startsWith("http") ? tts_url : `${API_BASE}${tts_url}`
            )

            audio.onended = () => {
              unmuteMicHard()
              setStatus("listening")
              wsRef.current?.send(JSON.stringify({ type: "tts_end" }))
            }

            audio.play()
          }

          if (end_session) {
            setActive(false)
            setStatus("idle")
            setBubbleText("어떤 문의가 있으신가요?")
          }
        }
      } catch (e) {
        console.error(e)
      }
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    })
    streamRef.current = stream

    const audioCtx = new AudioContext({ sampleRate: 16000 })
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
     Plate Upload
=============================== */
  const handlePlateUpload = async (file: File) => {
    if (active) return

    setStatus("thinking")
    setBubbleText("차량 번호판을 확인 중이에요…")

    const formData = new FormData()
    formData.append("image", file)

    try {
      const res = await fetch(`${API_BASE}/api/plate/recognize`, {
        method: "POST",
        body: formData,
      })
      const data = await res.json()

      if (!data.success) {
        setBubbleText("번호판을 인식하지 못했어요.\n다시 시도해 주세요.")
        setStatus("idle")
        return
      }

      setBubbleText(data.message)
      setStatus("speaking")

      const audio = new Audio(`${API_BASE}${data.tts_url}`)
      muteMicHard()
      audio.onended = () => {
        unmuteMicHard()
        startVoice()
      }
      audio.play()

    } catch (e) {
      console.error(e)
      setBubbleText("시스템 오류가 발생했어요.")
      setStatus("idle")
    }
  }

  const onPlateFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    handlePlateUpload(file)
    e.target.value = ""
  }

  /* ===============================
     UI
=============================== */
  return (
    <main className="min-h-screen bg-gradient-to-br from-emerald-50 via-sky-50 to-white flex items-center justify-center px-6 font-[Pretendard]">

      {showAdminPopup && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm">
          <div className="bg-white rounded-2xl px-10 py-8 shadow-2xl text-center">
            <p className="text-2xl font-semibold">🔔 관리실에 연락했습니다</p>
            <p className="mt-2 text-neutral-600">직원이 곧 도와드릴 예정입니다.</p>
          </div>
        </div>
      )}

      <header className="absolute top-8 text-center">
        <h1 className="text-4xl font-semibold tracking-[0.35em]">PARKMATE</h1>
        <p className="mt-1 text-xs tracking-[0.35em] text-neutral-400 uppercase">
          Parking Guidance Kiosk
        </p>
      </header>

      <div className="relative flex items-center">
        <div className={`${status === "thinking" ? "animate-bounce" : ""}`}>
          <div className="w-56 h-40 rounded-[2.5rem] bg-white shadow-2xl flex items-center justify-center">
            <div className="w-44 h-28 rounded-2xl bg-gradient-to-br from-emerald-300 to-sky-400 flex items-center justify-center gap-6">
              <span className="w-4 h-4 bg-white rounded-full" />
              <span className="w-4 h-4 bg-white rounded-full" />
            </div>
          </div>
          <p className="mt-4 text-center text-neutral-500">지미 · 주차 안내 파트너</p>
        </div>

        <div className="relative ml-6 -translate-y-12 max-w-[520px] bg-white px-10 py-8 rounded-[2.2rem] shadow-xl">
          <div className="absolute left-[-14px] bottom-1/2 -translate-y-1/2 w-0 h-0
            border-t-[10px] border-b-[10px] border-r-[16px]
            border-transparent border-r-white" />

          <p className="text-[22px] leading-relaxed whitespace-pre-line">
            {bubbleText}
          </p>

          <div className="mt-4 grid grid-cols-2 gap-3">
            {INTENT_UI_KEYWORDS.NONE.map((kw) => (
              <button
                key={kw}
                onClick={() =>
                  wsRef.current?.send(JSON.stringify({ type: "ui_keyword", text: kw }))
                }
                className="py-3 px-4 rounded-full border font-semibold hover:bg-neutral-100 transition"
              >
                {kw}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="absolute bottom-12 flex flex-col items-center gap-2">
        <input
          ref={plateInputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={onPlateFileChange}
        />
        <button
          onClick={() => plateInputRef.current?.click()}
          className="px-6 py-3 rounded-full bg-neutral-900 text-white font-semibold shadow-lg hover:bg-neutral-800 transition"
        >
          🚗 차량 번호판 업로드
        </button>
        <p className="text-xs text-neutral-400">
          ※ 실제 환경에서는 차량 정차 시 자동 인식됩니다
        </p>
      </div>
    </main>
  )
}
