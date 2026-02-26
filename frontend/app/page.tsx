"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import MicCards from "./MicCard"
import RealtimeLogCard from "./RealtimeLogCard"
import GuideCards from "./GuideCards"

/**
 * ✅ [추가] Avatar 제어 훅
 * - start(): WebRTC + Azure Avatar 세션 열기(1회)
 * - speak(text): bot_text를 Azure에 보내 "이거 말해줘" 호출
 * - stop(): 세션/PeerConnection 종료
 */
import { useAvatar } from "./useAvatar"

type Status = "OFF" | "LISTENING" | "THINKING" | "SPEAKING"

// ===============================
// WS URL (Production은 env 필수)
// ===============================
const ENV_WS = process.env.NEXT_PUBLIC_BACKEND_WS?.trim()
const WS_URL = ENV_WS || "ws://localhost:8000/ws/voice"

// ===============================
// ✅ [추가] 아바타 모드 토글
// - true면: bot_text → Azure Avatar(Speech SDK)로 말시키고,
//           서버가 보내는 WAV bytes는 재생하지 않음(이중 음성 방지)
// ===============================
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
  const workletRef = useRef<AudioWorkletNode | null>(null)

  // ======================================================
  // ✅ [추가] Avatar 관련 refs/hooks
  // - avatarVideoRef: Azure가 보내는 WebRTC video+audio stream을 붙일 <video>
  // - avatar: start/speak/stop 제공
  // ======================================================
 const avatarVideoRef = useRef<HTMLVideoElement>(null!)
  const avatar = useAvatar()

  // (참고) useMemo에 ref.current 넣어봤자 렌더 트리거가 아니라 의미가 약함.
  // 그래도 디버깅 용도로 남겨둠.
  const isWsOpen = useMemo(
    () => wsRef.current?.readyState === WebSocket.OPEN,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [isRunning]
  )

  /**
   * ======================================================
       STT→final까지는 기존과 동일
   *   바뀌는 건 bot_text 이후:
   *   (기존) bot_text → 서버TTS(wav) → ws bytes → 브라우저 오디오 재생
   *   (변경) bot_text → 브라우저가 Azure Avatar에 "말해줘" → WebRTC 스트림을 <video>로 재생
   * ======================================================
   */

  // ======================================================
  // WAV 재생용(서버가 send_bytes로 보냄)
  // ✅ [변경] USE_AVATAR=true면 WAV는 재생하지 않음(이중 음성 방지)
  // ======================================================
  const playWavBytes = async (wavBytes: ArrayBuffer) => {
    // ✅ [추가] 아바타 모드면 WAV 출력은 무시
    if (USE_AVATAR) return

    try {
      setStatus("SPEAKING")
      const blob = new Blob([wavBytes], { type: "audio/wav" })
      const url = URL.createObjectURL(blob)
      const audio = new Audio(url)
      await audio.play()
      audio.onended = () => {
        URL.revokeObjectURL(url)
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          setStatus("LISTENING")
        } else {
          setStatus("OFF")
        }
      }
    } catch {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) setStatus("LISTENING")
      else setStatus("OFF")
    }
  }

  const stopAll = async () => {
    // UI 먼저 OFF
    setIsRunning(false)
    setStatus("OFF")

    // WS stop signal
    try {
      const ws = wsRef.current
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "stop" }))
      }
    } catch {}

    // close ws (OPEN/CONNECTING에서만)
    try {
      const ws = wsRef.current
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        ws.close()
      }
    } catch {}
    wsRef.current = null

    // stop mic
    try {
      mediaStreamRef.current?.getTracks().forEach((t) => t.stop())
    } catch {}
    mediaStreamRef.current = null

    // close audio graph
    try {
        workletRef.current?.disconnect()
    } catch {}
     workletRef.current = null

    try {
      await audioCtxRef.current?.close()
    } catch {}
    audioCtxRef.current = null

    // ✅ [추가] 아바타 세션 정리
    // - 데모에서는 stopAll 때 같이 닫아도 되고,
    // - “아바타는 계속 띄워두고 싶다”면 여기 avatar.stop()을 빼도 됨.
    try {
      await avatar.stop()
    } catch {}
  }

  const startAll = async () => {
    // ✅ env가 없으면 바로 티나게 (배포에서 특히 중요)
    if (!ENV_WS && typeof window !== "undefined" && window.location.protocol === "https:") {
      console.error("❌ NEXT_PUBLIC_BACKEND_WS is missing on HTTPS (production).")
    }

    /**
     * ✅ [추가] 아바타 세션은 '유저 클릭 이후'에 여는 게 안전함(autoplay 정책)
     * - 마이크 버튼(토글)을 눌렀을 때 startAll이 호출되므로 UX도 자연스러움
     * - 성공하면 Azure Avatar가 보내는 WebRTC stream이 <video>에 붙고 "대기 화면"이 뜸
     */
    if (USE_AVATAR) {
      try {
        await avatar.start(avatarVideoRef)
      } catch (e) {
        console.error("[AVATAR] start failed:", e)
        // 아바타 실패해도 STT/LLM 흐름 자체는 돌릴 수 있게 그냥 진행(원하면 stopAll로 중단해도 됨)
      }
    }

    // 1) WS 연결 (기존 그대로)
    const ws = new WebSocket(WS_URL)
    wsRef.current = ws // ✅ 아주 중요: onopen 전에 ref 먼저 세팅 (audio loop에서 ref 참조함)
    ws.binaryType = "arraybuffer"

    ws.onopen = async () => {
      setIsRunning(true)
      setStatus("LISTENING")

      // 2) 마이크 캡처 + PCM16(16k)로 변환해서 ws.send(bytes) (기존 그대로)
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        })
        mediaStreamRef.current = stream

        const AudioCtx = window.AudioContext || (window as any).webkitAudioContext
        const audioCtx: AudioContext = new AudioCtx()
        audioCtxRef.current = audioCtx

        const source = audioCtx.createMediaStreamSource(stream)

        // ScriptProcessor는 구식이지만 “빠르게 붙이기”엔 제일 간단함
        await audioCtx.audioWorklet.addModule("/worklets/pcm16-processor.js")

        const node = new AudioWorkletNode(audioCtx, "pcm16-processor")
        workletRef.current = node

        node.port.onmessage = (e: MessageEvent) => {
          const sock = wsRef.current
          if (!sock || sock.readyState !== WebSocket.OPEN) return
          if (e.data instanceof ArrayBuffer) sock.send(e.data)
        }

        source.connect(node)
      } catch (err) {
        // 마이크 권한/장치 오류
        await stopAll()
      }
    }

    ws.onmessage = async (ev) => {
      // 3) 서버 응답 처리: JSON(text) or WAV(bytes)
      if (typeof ev.data === "string") {
        try {
          const msg = JSON.parse(ev.data)

          if (msg.type === "partial") {
            setPartialText(msg.text ?? "")
          }

          if (msg.type === "final") {
            setFinalText(msg.text ?? "")
            setPartialText("")
            setStatus("THINKING")
          }

          /**
           * ✅ [핵심 변경]
           * - 예전: bot_text는 화면 표시만 하고, 실제 발화는 server→WAV bytes를 받아 재생
           * - 지금: bot_text를 받는 즉시, 브라우저가 Azure Avatar에 "이거 말해줘" 호출
           */
          if (msg.type === "bot_text") {
            const text = msg.text ?? ""
            setBotText(text)

            if (USE_AVATAR) {
              try {
                setStatus("SPEAKING")
                await avatar.speak(text)
              } catch (e) {
                console.error("[AVATAR] speak failed:", e)
              } finally {
                // 말하기 끝나면 다시 듣기 상태로
                if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) setStatus("LISTENING")
                else setStatus("OFF")
              }
            }
          }

          /**
           * (옵션) server.py에서 barge_in 이벤트를 보내는 구조가 이미 있다면,
           * 여기서 처리해서 UI 상태만 빨리 LISTENING으로 되돌릴 수 있음.
           */
          if (msg.type === "barge_in") {
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) setStatus("LISTENING")
          }
        } catch {
          // ignore
        }
        return
      }

      /**
       * ✅ [변경/유지]
       * - 서버가 WAV bytes를 보내는 기존 로직은 유지되어도 됨(백엔드가 아직 분기 전이라면)
       * - 하지만 USE_AVATAR=true면 playWavBytes가 early return 해서 실제로는 재생 안 됨
       */
      if (ev.data instanceof ArrayBuffer) {
        await playWavBytes(ev.data)
        return
      }

      // 일부 브라우저는 Blob으로 올 수 있음
      if (ev.data instanceof Blob) {
        const ab = await ev.data.arrayBuffer()
        await playWavBytes(ab)
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

  // 페이지 이탈 시 정리
  useEffect(() => {
    return () => {
      stopAll()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <main className="min-h-screen px-6 flex justify-center">
      <div className="w-full max-w-[1200px]">
        <header className="pt-10 text-center select-none">
          <h1 className="text-4xl font-semibold tracking-[0.35em]">PARKMATE</h1>
          <p className="mt-1 text-xs tracking-[0.35em] text-neutral-400 uppercase">
            Parking Guidance Kiosk
          </p>
        </header>

        <section className="mt-12 flex flex-col items-center">
          {/* ======================================================
              ✅ [추가] Azure Avatar 비디오 자리
              - Azure Avatar가 WebRTC로 보내는 video+audio stream을 여기서 재생
              - "아바타가 그려진다" = 실제로는 Azure가 만든 영상을 스트리밍으로 받아 video가 보여주는 것
             ====================================================== */}
          <div className="w-full mb-6">
            <div className="rounded-3xl border border-white/70 bg-white/50 shadow-xl backdrop-blur overflow-hidden">
              <div className="px-6 py-4 flex items-center justify-between">
                <div className="text-sm font-semibold text-neutral-900">Avatar</div>
                <div className="text-xs text-neutral-400">{USE_AVATAR ? "ON" : "OFF"}</div>
              </div>

              <div className="px-6 pb-6">
                <video
                  ref={avatarVideoRef}
                  id="avatarVideo"
                  autoPlay
                  playsInline
                  // controls를 켜면 디버깅은 편한데, 키오스크 느낌은 깨져서 일단 OFF
                  // controls
                  className="w-full aspect-video rounded-2xl bg-black/10"
                />
                <p className="mt-2 text-xs text-neutral-400">
                  {USE_AVATAR
                    ? "bot_text → Speech SDK → Azure Avatar → (WebRTC) → <video>"
                    : "USE_AVATAR=false (기존 WAV 재생 방식)"}
                </p>
              </div>
            </div>
          </div>

          {/* ✅ 기존 UI: 마이크 카드 */}
          <div className="w-full">
            <MicCards isRunning={isRunning} status={status} onToggle={onToggle} />
          </div>

          {/* ✅ 기존 UI: 로그 카드 */}
          <RealtimeLogCard
            partialText={partialText}
            finalText={finalText}
            botText={botText}
            wsUrl={WS_URL}
            wsRef={wsRef}
            isWsOpen={isWsOpen}
          />

          {/* ✅ 기존 UI: 가이드 카드 */}
          <GuideCards />

          <p className="mt-8 text-center text-xs text-neutral-400"></p>
        </section>
      </div>
    </main>
  )
}