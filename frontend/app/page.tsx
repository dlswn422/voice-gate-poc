"use client"

import { useRef, useState, useEffect } from "react"

/* ===============================
   Types
=============================== */
type Status = "idle" | "listening" | "thinking" | "speaking"
type Intent =
  | "EXIT"
  | "ENTRY"
  | "PAYMENT"
  | "FEE"
  | "FACILITY"
  | "ADMIN"
  | "REGISTRATION"
  | "TIME_PRICE"
  | "NONE"

type PaymentResult = "SUCCESS" | "FAIL"
type PaymentFailReason =
  | "LIMIT_EXCEEDED"
  | "NETWORK_ERROR"
  | "INSUFFICIENT_FUNDS"
  | "USER_CANCEL"
  | "ETC"

/* ===============================
   Constants — 로컬 네트워크 동적 대응
=============================== */
const API_BASE =
  typeof window !== "undefined"
    ? `http://${window.location.hostname}:8000`
    : "http://127.0.0.1:8000"

const WS_BASE =
  typeof window !== "undefined"
    ? `ws://${window.location.hostname}:8000/ws/voice`
    : "ws://127.0.0.1:8000/ws/voice"

const INTENT_UI_KEYWORDS: Record<string, string[]> = {
  EXIT: ["차단기 안 열림", "출차 안 됨", "차량 인식 안 됨", "출구에서 멈춤", "기타", "관리실 호출"],
  ENTRY: ["입차 안 됨", "차단기 안 열림", "차량 인식 안 됨", "만차로 표시됨", "방문자 등록", "관리실 호출"],
  PAYMENT: ["결제 안 됨", "카드 오류", "요금 이상", "결제 방법 문의", "기타", "관리실 호출"],
  FEE: ["주차 요금 문의", "할인 적용 문의", "요금 기준 문의", "시간별 요금", "기타", "관리실 호출"],
  FACILITY: ["문 안 열림", "차단기 이상", "기기 멈춤", "화면 안 보임", "기타", "관리실 호출"],
  ADMIN: ["관리자 호출", "직원 연결", "민원 접수", "긴급 상황", "기타", "관리실 호출"],
  REGISTRATION: ["차량 등록", "방문자 등록", "등록 방법 문의", "등록했는데 안 됨", "기타", "관리실 호출"],
  TIME_PRICE: ["주차 시간 문의", "요금 문의", "할인 적용 문의", "요금 기준", "기타", "관리실 호출"],
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
  const [intent, setIntent] = useState<Intent>("NONE")

  const [showPaymentPopup, setShowPaymentPopup] = useState(false)
  const [paymentResult, setPaymentResult] = useState<PaymentResult | null>(null)
  const [paymentReason, setPaymentReason] = useState<PaymentFailReason | null>(null)
  const [paymentSubmitting, setPaymentSubmitting] = useState(false)
  const [paymentFeedback, setPaymentFeedback] = useState<PaymentResult | null>(null)

  const [parkingSessionId, setParkingSessionId] = useState<string | null>(null)

  // 🔒 음성 완전 제어용
  const [voiceLocked, setVoiceLocked] = useState(false)

  // 🚗 차량번호 입력 (CLI 메뉴 대응)
  const [plateInput, setPlateInput] = useState("")
  const [currentPlate, setCurrentPlate] = useState<string | null>(null)

  /* ===============================
     Refs
  =============================== */
  const wsRef = useRef<WebSocket | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const ttsQueueRef = useRef<string[]>([])           // TTS URL 큐
  const ttsPlayingRef = useRef<boolean>(false)        // 현재 재생 중 여부
  const currentAudioRef = useRef<HTMLAudioElement | null>(null)

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
  const startVoice = async (plate?: string) => {
    if (active || voiceLocked || showPaymentPopup) return

    setActive(true)
    setStatus("listening")
    setIntent("NONE")

    const ws = new WebSocket(WS_BASE)
    ws.binaryType = "arraybuffer"
    wsRef.current = ws

    ws.onopen = () => {
      // 서버에 현재 차량번호 전달
      if (plate || currentPlate) {
        ws.send(JSON.stringify({ type: "set_plate", plate: plate || currentPlate }))
      }
    }

    // ── TTS 큐 순차 재생 헬퍼 ──
    const playNextTts = () => {
      if (ttsPlayingRef.current) return
      const nextUrl = ttsQueueRef.current.shift()
      if (!nextUrl) {
        // 큐 비었음 → listening 복귀
        setStatus("listening")
        wsRef.current?.send(JSON.stringify({ type: "tts_end" }))
        return
      }
      ttsPlayingRef.current = true
      // barge-in: 마이크는 음소거하지 않음 (서버에서 VAD로 barge-in 감지)
      setStatus("speaking")
      const audio = new Audio(`${API_BASE}${nextUrl}`)
      currentAudioRef.current = audio
      audio.onended = () => {
        ttsPlayingRef.current = false
        currentAudioRef.current = null
        playNextTts()   // 다음 문장 자동 재생
      }
      audio.play()
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)

      if (voiceLocked) return

      // ── Barge-in: 서버에서 사용자 음성 감지 → TTS 즉시 중단 ──
      if (data.type === "barge_in") {
        // 현재 재생 중인 오디오 중단
        if (currentAudioRef.current) {
          currentAudioRef.current.pause()
          currentAudioRef.current.currentTime = 0
          currentAudioRef.current = null
        }
        // TTS 큐 비우기
        ttsQueueRef.current = []
        ttsPlayingRef.current = false
        setStatus("listening")
        return
      }

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
        if (data.intent) setIntent(data.intent)
        if (data.text) setBubbleText(data.text)

        // 문장 단위 TTS → 큐에 추가 후 순차 재생
        if (data.tts_url) {
          ttsQueueRef.current.push(data.tts_url)
          playNextTts()
        }
      }
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    })
    streamRef.current = stream

    // 이전 AudioContext 정리 (리소스 누수 방지)
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => { })
      audioCtxRef.current = null
    }

    const audioCtx = new AudioContext({ sampleRate: 16000 })
    await audioCtx.resume()  // onended 등 비-사용자 제스처에서 호출 시 suspended 해소
    audioCtxRef.current = audioCtx

    const source = audioCtx.createMediaStreamSource(stream)
    const processor = audioCtx.createScriptProcessor(4096, 1, 1)

    source.connect(processor)
    processor.connect(audioCtx.destination)

    processor.onaudioprocess = (e) => {
      // barge-in: listening + speaking 둘 다 오디오 전송 (서버에서 VAD 판단)
      if (statusRef.current !== "listening" && statusRef.current !== "speaking") return
      if (voiceLocked) return
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
      wsRef.current.send(e.inputBuffer.getChannelData(0).buffer)
    }
  }

  /* ===============================
     Voice Session Stop (수동 세션 종료)
  =============================== */
  const stopVoice = () => {
    // TTS 재생 중단
    if (currentAudioRef.current) {
      currentAudioRef.current.pause()
      currentAudioRef.current.currentTime = 0
      currentAudioRef.current = null
    }
    ttsQueueRef.current = []
    ttsPlayingRef.current = false

    // AudioContext 종료
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => { })
      audioCtxRef.current = null
    }

    // WebSocket 종료
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    // 마이크 트랙 해제
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null

    // 상태 리셋
    setActive(false)
    setStatus("idle")
    setIntent("NONE")
    setCurrentPlate(null)
    setDirection(null)
    setParkingSessionId(null)
    setVoiceLocked(false)
    setBubbleText("어떤 문의가 있으신가요?")
  }

  /* ===============================
     입/출차 처리 (CLI 메뉴 1, 2 대응)
  =============================== */
  const handlePlateAction = async (actionDirection: "ENTRY" | "EXIT") => {
    const plate = plateInput.trim()
    if (!plate) return

    // 기존 음성 세션 정리
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => { })
      audioCtxRef.current = null
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop())
      streamRef.current = null
    }
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setActive(false)
    setVoiceLocked(false)

    setStatus("thinking")
    setBubbleText("차량 정보를 확인 중이에요…")
    setDirection(actionDirection)
    setCurrentPlate(plate)
    setParkingSessionId(null)

    const formData = new FormData()
    formData.append("plate_number", plate)
    formData.append("direction", actionDirection)

    try {
      const res = await fetch(`${API_BASE}/api/plate/recognize`, {
        method: "POST",
        body: formData,
      })
      const data = await res.json()

      if (!data.success) {
        setBubbleText(data.message || "처리에 실패했어요.")
        setStatus("idle")
        return
      }

      setBubbleText(data.message)
      setParkingSessionId(data.parking_session_id ?? null)

      if (data.tts_url) {
        setStatus("speaking")
        const audio = new Audio(`${API_BASE}${data.tts_url}`)
        audio.onended = () => {
          setStatus("idle")
          // 입/출차 완료 후 음성 세션 시작
          startVoice(plate)
        }
        audio.play()
      } else {
        setStatus("idle")
        startVoice(plate)
      }
    } catch {
      setBubbleText("서버와 통신할 수 없어요.")
      setStatus("idle")
    }
  }

  /* ===============================
     Payment
  =============================== */
  const confirmPayment = async () => {
    if (paymentSubmitting) return
    if (!paymentResult || !parkingSessionId) return

    setPaymentSubmitting(true)
    setPaymentFeedback(null)

    setVoiceLocked(true)
    muteMicHard()

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

      const data = await res.json()
      if (!res.ok || !data.success) {
        throw new Error(data?.detail || "PAYMENT_FAILED")
      }

      setPaymentFeedback(paymentResult)

      if (paymentResult === "SUCCESS") {
        setBubbleText(
          "결제가 완료되었습니다.\n차량 번호를 다시 입력해 주세요."
        )
      }

      setTimeout(() => {
        setShowPaymentPopup(false)

        wsRef.current?.send(JSON.stringify({
          type: "voice_mode",
          value: "NORMAL",
        }))

        wsRef.current?.send(JSON.stringify({
          type: "payment_result",
          value: paymentResult,
        }))

        setVoiceLocked(false)
        unmuteMicHard()
      }, 300)

    } catch (e) {
      console.error("[PAYMENT ERROR]", e)

      setPaymentFeedback("FAIL")
      setVoiceLocked(false)
      unmuteMicHard()

      setBubbleText("결제 처리 중 오류가 발생했어요.")
    } finally {
      setPaymentSubmitting(false)
    }
  }

  /* ===============================
     🔥 결제 팝업 ↔ 음성 세션 동기화
  =============================== */
  useEffect(() => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return

    if (showPaymentPopup) {
      muteMicHard()
      ws.send(JSON.stringify({ type: "voice_mode", value: "PAYMENT" }))
    } else {
      ws.send(JSON.stringify({ type: "voice_mode", value: "NORMAL" }))
      unmuteMicHard()
    }
  }, [showPaymentPopup])

  /* ===============================
     UI
  =============================== */
  return (
    <main className="min-h-screen bg-gradient-to-br from-emerald-50 via-sky-50 to-white flex items-center justify-center px-6 font-[Pretendard]">
      {/* 상단 헤더 */}
      <header className="absolute top-8 text-center z-10">
        <h1 className="text-4xl font-semibold tracking-[0.35em]">PARKMATE</h1>
        <p className="mt-1 text-xs tracking-[0.35em] text-neutral-400 uppercase">
          Parking Guidance Kiosk
        </p>
      </header>

      {/* 메인 컨텐츠 */}
      <div className="relative flex items-center gap-12">

        {/* 🔥 현재 차량 정보 (입/출차 후 표시) */}
        {currentPlate && (
          <div className="relative w-[320px] overflow-visible">
            <div
              className="absolute -right-4 top-20 w-0 h-0 z-10
              border-t-[14px] border-b-[14px] border-l-[18px]
              border-transparent border-l-white"
            />
            <div className="bg-white rounded-3xl shadow-2xl overflow-hidden p-7 space-y-3">
              <div className="flex items-center justify-between">
                <p className="text-2xl font-bold tracking-wider">
                  🚗 {currentPlate}
                </p>
                <span className={`inline-block px-3 py-1 rounded-full text-sm font-semibold
                  ${direction === "ENTRY"
                    ? "bg-emerald-100 text-emerald-700"
                    : "bg-amber-100 text-amber-700"
                  }`}>
                  {direction === "ENTRY" ? "입차" : "출차"}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* 지미 + 말풍선 */}
        <div className="flex items-center">
          <div className={`${status === "thinking" ? "animate-bounce" : ""}`}>
            <div className="w-56 h-40 rounded-[2.5rem] bg-white shadow-2xl flex items-center justify-center">
              <div className="w-44 h-28 rounded-2xl bg-gradient-to-br from-emerald-300 to-sky-400 flex items-center justify-center gap-6">
                <span className="w-4 h-4 bg-white rounded-full" />
                <span className="w-4 h-4 bg-white rounded-full" />
              </div>
            </div>
            <p className="mt-4 text-center text-neutral-500">
              지미 · 주차 안내 파트너
            </p>
          </div>

          {/* 말풍선 */}
          <div className="relative ml-6 -translate-y-10 max-w-[520px] bg-white px-10 py-8 rounded-[2.2rem] shadow-xl">
            <div
              className="absolute left-[-14px] top-1/2 -translate-y-1/2 w-0 h-0
              border-t-[10px] border-b-[10px] border-r-[16px]
              border-transparent border-r-white"
            />

            <p className="text-[22px] leading-relaxed whitespace-pre-line">
              {bubbleText}
            </p>

            <div className="mt-4 grid grid-cols-2 gap-3">
              {(INTENT_UI_KEYWORDS[intent] || INTENT_UI_KEYWORDS["NONE"]).map((kw) => (
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
      </div>

      {/* 하단 — 차량번호 입력 + 입/출차 버튼 (CLI 대응) */}
      <div className="absolute bottom-12 flex flex-col items-center gap-4 z-10">
        <div className="flex items-center gap-3">
          <input
            type="text"
            value={plateInput}
            onChange={(e) => setPlateInput(e.target.value)}
            placeholder="차량번호 입력 (예: 12가3456)"
            className="px-5 py-3 rounded-full border-2 border-neutral-200 focus:border-emerald-500 focus:outline-none text-lg w-[280px] text-center"
            onKeyDown={(e) => {
              if (e.key === "Enter" && plateInput.trim()) {
                handlePlateAction("ENTRY")
              }
            }}
          />

          <button
            onClick={() => handlePlateAction("ENTRY")}
            disabled={!plateInput.trim() || status === "thinking"}
            className="px-6 py-3 rounded-full bg-emerald-600 text-white font-semibold shadow-lg hover:bg-emerald-700 transition disabled:opacity-40"
          >
            🅿️ 입차
          </button>

          <button
            onClick={() => handlePlateAction("EXIT")}
            disabled={!plateInput.trim() || status === "thinking"}
            className="px-6 py-3 rounded-full bg-amber-600 text-white font-semibold shadow-lg hover:bg-amber-700 transition disabled:opacity-40"
          >
            🚗 출차
          </button>

          {direction === "EXIT" && parkingSessionId && (
            <button
              onClick={() => {
                setShowPaymentPopup(true)
                setVoiceLocked(true)
                muteMicHard()

                wsRef.current?.send(JSON.stringify({
                  type: "voice_mode",
                  value: "PAYMENT",
                }))
              }}
              className="px-6 py-3 rounded-full bg-neutral-900 text-white font-semibold shadow-lg hover:bg-neutral-800 transition"
            >
              💳 결제
            </button>
          )}

          {active && (
            <button
              onClick={stopVoice}
              className="px-6 py-3 rounded-full bg-rose-600 text-white font-semibold shadow-lg hover:bg-rose-700 transition"
            >
              ⏹ 세션 종료
            </button>
          )}
        </div>

        <p className="text-xs text-neutral-400">
          ※ 차량번호를 입력하고 입차/출차 버튼을 누르세요. 이후 마이크로 음성 대화가 가능합니다.
        </p>
      </div>

      {/* ===============================
          결제 팝업
      =============================== */}
      {showPaymentPopup && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-white rounded-2xl px-10 py-8 shadow-2xl w-[420px]">
            <p className="text-xl font-semibold text-center">💳 결제 처리</p>

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

            {(!paymentResult || (paymentResult === "FAIL" && !paymentReason)) && (
              <p className="mt-4 text-sm text-rose-500 text-center">
                결제 결과와 필요한 정보를 모두 선택해 주세요.
              </p>
            )}

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
            <div className="mt-6 flex justify-end gap-3">
              <button
                onClick={() => setShowPaymentPopup(false)}
                className="px-5 py-2 rounded-full border border-neutral-300 text-neutral-600 hover:bg-neutral-100"
              >
                취소
              </button>

              <button
                onClick={confirmPayment}
                disabled={
                  paymentSubmitting ||
                  !paymentResult ||
                  (paymentResult === "FAIL" && !paymentReason)
                }
                className="px-4 py-2 rounded-full bg-emerald-600 text-white disabled:opacity-40"
              >
                {paymentSubmitting ? "처리 중..." : "확인"}
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  )
}
