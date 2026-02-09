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

type PaymentResult = "SUCCESS" | "FAIL"
type PaymentFailReason =
  | "LIMIT_EXCEEDED"
  | "NETWORK_ERROR"
  | "INSUFFICIENT_FUNDS"
  | "USER_CANCEL"
  | "ETC"

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
  const [direction, setDirection] = useState<"ENTRY" | "EXIT" | null>(null)
  const [status, _setStatus] = useState<Status>("idle")
  const statusRef = useRef<Status>("idle")
  const setStatus = (s: Status) => {
    statusRef.current = s
    _setStatus(s)
  }

  const [bubbleText, setBubbleText] = useState("어떤 문의가 있으신가요?")
  const [active, setActive] = useState(false)
  const [showAdminPopup, setShowAdminPopup] = useState(false)
  const [intent, setIntent] = useState<Intent>("NONE")

  const [showPaymentPopup, setShowPaymentPopup] = useState(false)
  const [paymentResult, setPaymentResult] = useState<PaymentResult | null>(null)
  const [paymentReason, setPaymentReason] = useState<PaymentFailReason | null>(null)

  // ✅ 추가된 상태
  const [paymentSubmitting, setPaymentSubmitting] = useState(false)
  const [paymentFeedback, setPaymentFeedback] = useState<PaymentResult | null>(null)

  // ✅ 핵심: 현재 주차 세션 ID
  const [parkingSessionId, setParkingSessionId] = useState<string | null>(null)

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
    setIntent("NONE")

    const ws = new WebSocket(WS_BASE)
    ws.binaryType = "arraybuffer"
    wsRef.current = ws

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)

        if (data.type === "assistant_state") {
          if (data.state === "THINKING") {
            setStatus("thinking")
            setBubbleText("잠시만요…\n확인 중이에요.")
          }
          if (data.state === "LISTENING") setStatus("listening")
          if (data.state === "SPEAKING") setStatus("speaking")
          return
        }

        if (data.type === "assistant_message") {
          const { text, tts_url, end_session, system_action, intent: serverIntent } = data

          if (serverIntent) setIntent(serverIntent)

          if (system_action === "CALL_ADMIN") {
            muteMicHard()
            setShowAdminPopup(true)

            setTimeout(() => {
              setShowAdminPopup(false)
              setActive(false)
              setStatus("idle")
              setIntent("NONE")
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
            setIntent("NONE")
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
      /* 입출차 구분 */
      setDirection(data.direction)
      setParkingSessionId(data.parking_session_id ?? null)

      setParkingSessionId(data.parking_session_id ?? null)

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

  /* ===============================
     Payment
  =============================== */
  const confirmPayment = async () => {
    if (!paymentResult || !parkingSessionId) return

    setPaymentSubmitting(true)
    setPaymentFeedback(null)

    try {
      const res = await fetch(`${API_BASE}/api/payment/demo`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          parking_session_id: parkingSessionId,
          result: paymentResult,
          reason: paymentResult === "FAIL" ? paymentReason : null,
        }),
      })

      if (res.ok) {
        setPaymentFeedback(paymentResult)
      } else {
        setPaymentFeedback("FAIL")
      }
    } catch {
      setPaymentFeedback("FAIL")
    } finally {
      setPaymentSubmitting(false)
    }
  }

  /* ===============================
     UI
  =============================== */
  return (
    <main className="min-h-screen bg-gradient-to-br from-emerald-50 via-sky-50 to-white flex items-center justify-center px-6 font-[Pretendard]">

      {/* 상단 헤더 */}
      <header className="absolute top-8 text-center">
        <h1 className="text-4xl font-semibold tracking-[0.35em]">PARKMATE</h1>
        <p className="mt-1 text-xs tracking-[0.35em] text-neutral-400 uppercase">
          Parking Guidance Kiosk
        </p>
      </header>

      {/* 지미 + 말풍선 */}
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
            {INTENT_UI_KEYWORDS[intent].map((kw) => (
              <button
                key={kw}
                className="py-3 px-4 rounded-full border font-semibold hover:bg-neutral-100 transition"
              >
                {kw}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 하단 버튼 */}
      <div className="absolute bottom-12 flex flex-col items-center gap-2">
        <input
          ref={plateInputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (!file) return
            handlePlateUpload(file)
            e.target.value = ""
          }}
        />

        <div className="flex items-center gap-4">
          <button
            onClick={() => plateInputRef.current?.click()}
            className="px-6 py-3 rounded-full bg-neutral-900 text-white font-semibold shadow-lg hover:bg-neutral-800 transition"
          >
            🚗 차량 번호판 업로드
          </button>
          {direction === "EXIT" && (
            <button
              onClick={() => setShowPaymentPopup(true)}
              className="px-6 py-3 rounded-full bg-emerald-600 text-white font-semibold shadow-lg hover:bg-emerald-700 transition"
            >
              💳 결제하기
            </button>
          )}
        </div>

        <p className="text-xs text-neutral-400">
          ※ 현재는 데모 환경으로, 차량 번호판 업로드 방식으로 입·출차를 확인합니다
        </p>
      </div>

      {/* ===============================
         결제 팝업 (UI 개선, 나머지 전부 동일)
      =============================== */}
      {showPaymentPopup && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm">
          <div className="bg-white rounded-2xl px-10 py-8 shadow-2xl w-[420px]">
            <p className="text-xl font-semibold text-center">💳 결제 처리</p>

            {/* 결과 선택 */}
            <div className="mt-6 grid grid-cols-2 gap-4">
              <button
                onClick={() => {
                  setPaymentResult("SUCCESS")
                  setPaymentReason(null)
                }}
                className={`p-4 rounded-xl border text-center font-semibold transition
                  ${paymentResult === "SUCCESS"
                    ? "bg-emerald-600 text-white border-emerald-600"
                    : "hover:bg-neutral-100"
                  }`}
              >
                ✅ 결제 성공
              </button>

              <button
                onClick={() => setPaymentResult("FAIL")}
                className={`p-4 rounded-xl border text-center font-semibold transition
                  ${paymentResult === "FAIL"
                    ? "bg-rose-500 text-white border-rose-500"
                    : "hover:bg-neutral-100"
                  }`}
              >
                ❌ 결제 실패
              </button>
            </div>

            {/* 실패 사유 */}
            {paymentResult === "FAIL" && (
              <div className="mt-6">
                <p className="mb-2 text-sm text-neutral-500">실패 사유 선택</p>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    ["LIMIT_EXCEEDED", "한도 초과"],
                    ["INSUFFICIENT_FUNDS", "잔액 부족"],
                    ["NETWORK_ERROR", "통신 오류"],
                    ["USER_CANCEL", "사용자 취소"],
                    ["ETC", "기타"],
                  ].map(([code, label]) => (
                    <button
                      key={code}
                      onClick={() => setPaymentReason(code as PaymentFailReason)}
                      className={`px-3 py-2 rounded-lg border text-sm transition
                        ${paymentReason === code
                          ? "bg-neutral-900 text-white border-neutral-900"
                          : "hover:bg-neutral-100"
                        }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* 경고 문구 */}
            {(!paymentResult || (paymentResult === "FAIL" && !paymentReason)) && (
              <p className="mt-4 text-sm text-rose-500 text-center">
                결제 결과와 필요한 정보를 모두 선택해 주세요.
              </p>
            )}
            {/* 결제 결과 피드백 */}
            {paymentFeedback && (
              <div
                className={`mt-4 p-3 rounded-xl text-center font-semibold
                  ${paymentFeedback === "SUCCESS"
                    ? "bg-emerald-100 text-emerald-700"
                    : "bg-rose-100 text-rose-700"
                  }`}
              >
                {paymentFeedback === "SUCCESS"
                  ? "결제가 성공적으로 완료되었습니다."
                  : "결제에 실패했습니다. 다시 시도해 주세요."}
              </div>
            )}
            <div className="mt-6 flex justify-between">
              <button
                onClick={() => setShowPaymentPopup(false)}
                className="px-4 py-2 rounded-full border"
              >
                취소
              </button>
              <button
                onClick={confirmPayment}
                disabled={!paymentResult || (paymentResult === "FAIL" && !paymentReason)}
                className="px-4 py-2 rounded-full bg-emerald-600 text-white disabled:opacity-40"
              >
                확인
              </button>
            </div>
          </div>
        </div>
      )}

    </main>
  )
}
