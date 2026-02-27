"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import MicCards from "./MicCard"
import RealtimeLogCard from "./RealtimeLogCard"
import GuideCards from "./GuideCards"
import { useAvatar } from "./useAvatar"

type Status = "OFF" | "LISTENING" | "THINKING" | "SPEAKING"

// ===============================
// WS URL (Production은 env 필수)
// ===============================
const ENV_WS = process.env.NEXT_PUBLIC_BACKEND_WS?.trim()
const WS_URL = ENV_WS || "ws://localhost:8000/ws/voice"

// ===============================
// ✅ [추가] 아바타 모드 토글
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

// ===============================
// Expression helpers (blendshapes -> stable label)
// ===============================
type ExprLabel = "neutral" | "positive" | "frustrated" | "angry" | "confused"

type ExprSignals = {
  smile: number
  frown: number
  browDown: number
  browUp: number
  eyeWide: number
  eyeSquint: number
  mouthPress: number
  jawOpen: number
}

// ✅ [해결1] scores 타입 + infer 결과 타입 통일
type ExprScores = {
  angryScore: number
  frustratedScore: number
  confusedScore: number
  positiveScore: number
  negStrength: number
  intensity: number
  polarity: number
}

type ExprInferResult = {
  label: ExprLabel
  valence: number
  arousal: number
  confidence: number
  deltaSignals: ExprSignals
  scores: ExprScores
}

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x))
}
function round3(x: number) {
  return Math.round(x * 1000) / 1000
}
function bsScore(map: Map<string, number>, name: string) {
  return map.get(name) ?? 0
}

// ✅ [해결1] 항상 존재하는 기본값
const ZERO_SIGNALS: ExprSignals = {
  smile: 0,
  frown: 0,
  browDown: 0,
  browUp: 0,
  eyeWide: 0,
  eyeSquint: 0,
  mouthPress: 0,
  jawOpen: 0,
}

const ZERO_SCORES: ExprScores = {
  angryScore: 0,
  frustratedScore: 0,
  confusedScore: 0,
  positiveScore: 0,
  negStrength: 0,
  intensity: 0,
  polarity: 0,
}

// EMA smoother to stabilize signals
class EmaSmoother {
  private alpha: number
  private state: ExprSignals
  constructor(alpha = 0.35) {
    this.alpha = alpha
    this.state = { ...ZERO_SIGNALS }
  }
  update(next: ExprSignals) {
    const a = this.alpha
    const s = this.state
    this.state = {
      smile: s.smile + a * (next.smile - s.smile),
      frown: s.frown + a * (next.frown - s.frown),
      browDown: s.browDown + a * (next.browDown - s.browDown),
      browUp: s.browUp + a * (next.browUp - s.browUp),
      eyeWide: s.eyeWide + a * (next.eyeWide - s.eyeWide),
      eyeSquint: s.eyeSquint + a * (next.eyeSquint - s.eyeSquint),
      mouthPress: s.mouthPress + a * (next.mouthPress - s.mouthPress),
      jawOpen: s.jawOpen + a * (next.jawOpen - s.jawOpen),
    }
    return this.state
  }
  get() {
    return this.state
  }
}

// Hysteresis labeler to reduce flicker
class HysteresisLabeler {
  private current: ExprLabel = "neutral"
  private candidate: ExprLabel = "neutral"
  private hits = 0
  private hold = 3
  constructor(holdFrames = 3) {
    this.hold = holdFrames
  }
  step(next: ExprLabel) {
    if (next === this.current) {
      this.candidate = next
      this.hits = 0
      return this.current
    }
    if (next === this.candidate) this.hits += 1
    else {
      this.candidate = next
      this.hits = 1
    }
    if (this.hits >= this.hold) {
      this.current = this.candidate
      this.hits = 0
    }
    return this.current
  }
  get() {
    return this.current
  }
}

// ✅ NEW: Baseline calibrator (1~2초 기본 표정 평균)
class BaselineCalibrator {
  private n = 0
  private maxN: number
  private sum: ExprSignals
  private baseline: ExprSignals | null = null

  constructor(maxSamples = 8) {
    this.maxN = maxSamples // 250ms 간격이면 8개=약 2초
    this.sum = { ...ZERO_SIGNALS }
  }

  reset() {
    this.n = 0
    this.baseline = null
    this.sum = { ...ZERO_SIGNALS }
  }

  update(x: ExprSignals) {
    if (this.baseline) return this.baseline
    this.n += 1
    this.sum = {
      smile: this.sum.smile + x.smile,
      frown: this.sum.frown + x.frown,
      browDown: this.sum.browDown + x.browDown,
      browUp: this.sum.browUp + x.browUp,
      eyeWide: this.sum.eyeWide + x.eyeWide,
      eyeSquint: this.sum.eyeSquint + x.eyeSquint,
      mouthPress: this.sum.mouthPress + x.mouthPress,
      jawOpen: this.sum.jawOpen + x.jawOpen,
    }
    if (this.n >= this.maxN) {
      this.baseline = {
        smile: this.sum.smile / this.n,
        frown: this.sum.frown / this.n,
        browDown: this.sum.browDown / this.n,
        browUp: this.sum.browUp / this.n,
        eyeWide: this.sum.eyeWide / this.n,
        eyeSquint: this.sum.eyeSquint / this.n,
        mouthPress: this.sum.mouthPress / this.n,
        jawOpen: this.sum.jawOpen / this.n,
      }
    }
    return this.baseline
  }

  get() {
    return this.baseline
  }
}

// ✅ Face bbox area from landmarks (normalized 0..1 coords)
const bboxAreaFromLandmarks = (lm: any[]) => {
  let minX = 1, minY = 1, maxX = 0, maxY = 0
  for (const p of lm) {
    if (p.x < minX) minX = p.x
    if (p.y < minY) minY = p.y
    if (p.x > maxX) maxX = p.x
    if (p.y > maxY) maxY = p.y
  }
  const w = Math.max(0, maxX - minX)
  const h = Math.max(0, maxY - minY)
  return w * h
}

// ✅ Face quality score from bbox area (tunable)
const faceQualityFromArea = (area: number) => {
  if (area <= 0.03) return 0
  if (area >= 0.10) return 1
  return clamp01((area - 0.03) / (0.10 - 0.03))
}

// ✅ NEW: improved inference using baseline + quality
function inferExpressionImproved(
  smoothed: ExprSignals,
  baseline: ExprSignals | null,
  quality: number
): ExprInferResult {
  const base = baseline ?? { ...ZERO_SIGNALS }

  // baseline 대비 변화량(delta) — 개인 기본 얼굴 보정
  const d: ExprSignals = {
    smile: Math.max(0, smoothed.smile - base.smile),
    frown: Math.max(0, smoothed.frown - base.frown),
    browDown: Math.max(0, smoothed.browDown - base.browDown),
    browUp: Math.max(0, smoothed.browUp - base.browUp),
    eyeWide: Math.max(0, smoothed.eyeWide - base.eyeWide),
    eyeSquint: Math.max(0, smoothed.eyeSquint - base.eyeSquint),
    mouthPress: Math.max(0, smoothed.mouthPress - base.mouthPress),
    jawOpen: Math.max(0, smoothed.jawOpen - base.jawOpen),
  }

  const smile = d.smile
  const frown = d.frown
  const browDown = d.browDown
  const browUp = d.browUp
  const eyeWide = d.eyeWide
  const eyeSquint = d.eyeSquint
  const mouthPress = d.mouthPress
  const jawOpen = d.jawOpen

  // valence: positive vs negative (baseline-corrected)
  let valence = 1.4 * smile - 1.0 * frown - 0.85 * browDown - 0.65 * mouthPress - 0.15 * eyeSquint
  valence = Math.max(-1, Math.min(1, valence))

  // arousal: activation
  let arousal = 0.85 * jawOpen + 0.75 * eyeWide + 0.35 * browUp + 0.35 * browDown + 0.2 * eyeSquint
  arousal = clamp01(arousal)

  const negStrength = Math.max(frown, browDown, mouthPress)
  const posStrength = smile
  const intensity = clamp01(Math.max(posStrength, negStrength, arousal))
  const polarity = clamp01(Math.abs(valence))

  // confidence: intensity + polarity + quality
  const confidence = clamp01((0.12 + 0.75 * intensity + 0.35 * polarity) * (0.35 + 0.65 * quality))

  // label scoring
  const angryScore = 0.55 * negStrength + 0.45 * browDown + 0.25 * mouthPress + 0.20 * eyeSquint + 0.20 * arousal
  const frustratedScore = 0.60 * negStrength + 0.30 * mouthPress + 0.25 * eyeSquint + 0.10 * arousal
  const confusedScore = 0.60 * browUp + 0.45 * eyeWide + 0.15 * jawOpen + 0.15 * (1 - polarity)
  const positiveScore = 0.80 * smile + 0.10 * (1 - negStrength) + 0.10 * polarity

  let label: ExprLabel = "neutral"

  if (quality < 0.20 || confidence < 0.20) {
    label = "neutral"
  } else if (angryScore >= 0.62 && valence <= -0.18) {
    label = "angry"
  } else if (frustratedScore >= 0.48 && valence <= -0.08) {
    label = "frustrated"
  } else if (confusedScore >= 0.52 && polarity <= 0.50) {
    label = "confused"
  } else if (positiveScore >= 0.46 && valence >= 0.12) {
    label = "positive"
  } else {
    label = "neutral"
  }

  return {
    label,
    valence,
    arousal,
    confidence,
    deltaSignals: d,
    scores: {
      angryScore,
      frustratedScore,
      confusedScore,
      positiveScore,
      negStrength,
      intensity,
      polarity,
    },
  }
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

  // Avatar
  const avatarVideoRef = useRef<HTMLVideoElement>(null!)
  const avatar = useAvatar()

  // Camera
  const [camOn, setCamOn] = useState(false)
  const faceVideoRef = useRef<HTMLVideoElement | null>(null)
  const camStreamRef = useRef<MediaStream | null>(null)

  const faceLandmarkerRef = useRef<any>(null)
  const visionTimerRef = useRef<number | null>(null)
  const exprSmootherRef = useRef<EmaSmoother | null>(null)
  const exprLabelerRef = useRef<HysteresisLabeler | null>(null)

  // baseline + debug refs
  const baselineRef = useRef<BaselineCalibrator | null>(null)
  const lastQualityRef = useRef<number>(0)

  const stopFaceCamera = async () => {
    try {
      if (visionTimerRef.current) {
        window.clearInterval(visionTimerRef.current)
        visionTimerRef.current = null
      }
      if (camStreamRef.current) {
        camStreamRef.current.getTracks().forEach((t) => t.stop())
        camStreamRef.current = null
      }
      if (faceVideoRef.current) {
        // @ts-ignore
        faceVideoRef.current.srcObject = null
      }
      faceLandmarkerRef.current = null
      baselineRef.current?.reset()
    } catch {}
  }

  useEffect(() => {
    let cancelled = false

    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        })
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop())
          return
        }
        camStreamRef.current = stream
        if (faceVideoRef.current) {
          // @ts-ignore
          faceVideoRef.current.srcObject = stream
          await faceVideoRef.current.play().catch(() => {})
        }

        const vision = await import("@mediapipe/tasks-vision")
        const { FaceLandmarker, FilesetResolver } = vision
        const fileset = await FilesetResolver.forVisionTasks("/mediapipe/wasm")

        const landmarker = await FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: "/mediapipe/models/face_landmarker.task" },
          runningMode: "VIDEO",
          numFaces: 1,
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: false,
        })

        if (cancelled) return
        faceLandmarkerRef.current = landmarker

        exprSmootherRef.current = exprSmootherRef.current ?? new EmaSmoother(0.35)
        exprLabelerRef.current = exprLabelerRef.current ?? new HysteresisLabeler(2)
        baselineRef.current = baselineRef.current ?? new BaselineCalibrator(8)

        visionTimerRef.current = window.setInterval(() => {
          const ws = wsRef.current
          const video = faceVideoRef.current
          const lm = faceLandmarkerRef.current

          if (!ws || ws.readyState !== WebSocket.OPEN) return
          if (!video || !lm) return
          if (video.readyState < 2) return
          if (!video.videoWidth || !video.videoHeight) return

          try {
            const res = lm.detectForVideo(video, performance.now())

            const blend = res?.faceBlendshapes?.[0]
            const categories = blend?.categories || []

            const hasFace = categories.length > 0
            const landmarks = res?.faceLandmarks?.[0] || null

            let quality = 0
            let area = 0
            if (landmarks && Array.isArray(landmarks) && landmarks.length > 0) {
              area = bboxAreaFromLandmarks(landmarks)
              quality = faceQualityFromArea(area)
            }
            lastQualityRef.current = quality

            const map = new Map<string, number>()
            for (const c of categories as any[]) {
              if (c?.categoryName) map.set(c.categoryName, typeof c.score === "number" ? c.score : 0)
            }

            const rawSignals: ExprSignals = {
              smile: Math.max(bsScore(map, "mouthSmileLeft"), bsScore(map, "mouthSmileRight")),
              frown: Math.max(bsScore(map, "mouthFrownLeft"), bsScore(map, "mouthFrownRight")),
              browDown: Math.max(bsScore(map, "browDownLeft"), bsScore(map, "browDownRight")),
              browUp: Math.max(
                bsScore(map, "browInnerUp"),
                Math.max(bsScore(map, "browOuterUpLeft"), bsScore(map, "browOuterUpRight"))
              ),
              eyeWide: Math.max(bsScore(map, "eyeWideLeft"), bsScore(map, "eyeWideRight")),
              eyeSquint: Math.max(bsScore(map, "eyeSquintLeft"), bsScore(map, "eyeSquintRight")),
              mouthPress: Math.max(bsScore(map, "mouthPressLeft"), bsScore(map, "mouthPressRight")),
              jawOpen: bsScore(map, "jawOpen"),
            }

            const smoother = exprSmootherRef.current!
            const smoothed = hasFace ? smoother.update(rawSignals) : smoother.update(ZERO_SIGNALS)

            const calibrator = baselineRef.current!
            const baseline = hasFace && quality >= 0.25 ? calibrator.update(smoothed) : calibrator.get()

            // ✅ [해결1] inferred는 항상 ExprInferResult
            const inferred: ExprInferResult = hasFace
              ? inferExpressionImproved(smoothed, baseline, quality)
              : {
                  label: "neutral",
                  valence: 0,
                  arousal: 0,
                  confidence: 0,
                  deltaSignals: ZERO_SIGNALS,
                  scores: ZERO_SCORES,
                }

            const labeler = exprLabelerRef.current!
            const stableLabel = labeler.step(inferred.label)

            const payload = {
              type: "vision_expression",
              ts: Date.now(),
              expression: {
                label: stableLabel,
                valence: round3(inferred.valence),
                arousal: round3(inferred.arousal),
                confidence: round3(inferred.confidence),
                signals: {
                  smile: round3(inferred.deltaSignals.smile),
                  frown: round3(inferred.deltaSignals.frown),
                  browDown: round3(inferred.deltaSignals.browDown),
                  browUp: round3(inferred.deltaSignals.browUp),
                  eyeWide: round3(inferred.deltaSignals.eyeWide),
                  eyeSquint: round3(inferred.deltaSignals.eyeSquint),
                  mouthPress: round3(inferred.deltaSignals.mouthPress),
                  jawOpen: round3(inferred.deltaSignals.jawOpen),
                },
                _debug: {
                  quality: round3(quality),
                  area: round3(area),
                  baselineReady: !!baseline,
                  angryScore: round3(inferred.scores.angryScore),
                  frustratedScore: round3(inferred.scores.frustratedScore),
                  confusedScore: round3(inferred.scores.confusedScore),
                  positiveScore: round3(inferred.scores.positiveScore),
                  intensity: round3(inferred.scores.intensity),
                  polarity: round3(inferred.scores.polarity),
                },
              },
            }

            const now = Date.now()
            // @ts-ignore
            if (!(window as any).__lastVisionLog || now - (window as any).__lastVisionLog > 2000) {
              // @ts-ignore
              ;(window as any).__lastVisionLog = now
              console.log("[VISION_EXPR] send", payload)
            }

            ws.send(JSON.stringify(payload))
          } catch (e) {
            console.warn("detectForVideo failed:", e)
          }
        }, 250)
      } catch (e) {
        console.warn("camera init failed:", e)
        setCamOn(false)
      }
    }

    if (camOn) startCamera()
    else stopFaceCamera()

    return () => {
      cancelled = true
      stopFaceCamera()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camOn])

  const isWsOpen = useMemo(
    () => wsRef.current?.readyState === WebSocket.OPEN,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [isRunning]
  )

  const playWavBytes = async (wavBytes: ArrayBuffer) => {
    if (USE_AVATAR) return
    try {
      setStatus("SPEAKING")
      const blob = new Blob([wavBytes], { type: "audio/wav" })
      const url = URL.createObjectURL(blob)
      const audio = new Audio(url)
      await audio.play()
      audio.onended = () => {
        URL.revokeObjectURL(url)
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) setStatus("LISTENING")
        else setStatus("OFF")
      }
    } catch {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) setStatus("LISTENING")
      else setStatus("OFF")
    }
  }

  const stopAll = async () => {
    setIsRunning(false)
    setStatus("OFF")

    try {
      const ws = wsRef.current
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "stop" }))
    } catch {}

    try {
      const ws = wsRef.current
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) ws.close()
    } catch {}
    wsRef.current = null

    try {
      mediaStreamRef.current?.getTracks().forEach((t) => t.stop())
    } catch {}
    mediaStreamRef.current = null

    try {
      workletRef.current?.disconnect()
    } catch {}
    workletRef.current = null

    try {
      await audioCtxRef.current?.close()
    } catch {}
    audioCtxRef.current = null

    try {
      await avatar.stop()
    } catch {}

    try {
      setCamOn(false)
    } catch {}
  }

  const startAll = async () => {
    if (!ENV_WS && typeof window !== "undefined" && window.location.protocol === "https:") {
      console.error("❌ NEXT_PUBLIC_BACKEND_WS is missing on HTTPS (production).")
    }

    if (USE_AVATAR) {
      try {
        await avatar.start(avatarVideoRef)
      } catch (e) {
        console.error("[AVATAR] start failed:", e)
      }
    }

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws
    ws.binaryType = "arraybuffer"

    ws.onopen = async () => {
      setIsRunning(true)
      setStatus("LISTENING")

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        })
        mediaStreamRef.current = stream

        const AudioCtx = window.AudioContext || (window as any).webkitAudioContext
        const audioCtx: AudioContext = new AudioCtx()
        audioCtxRef.current = audioCtx

        const source = audioCtx.createMediaStreamSource(stream)

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
        await stopAll()
      }
    }

    ws.onmessage = async (ev) => {
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
                if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) setStatus("LISTENING")
                else setStatus("OFF")
              }
            }
          }

          if (msg.type === "barge_in") {
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) setStatus("LISTENING")
          }
        } catch {
          // ignore
        }
        return
      }

      if (ev.data instanceof ArrayBuffer) {
        await playWavBytes(ev.data)
        return
      }

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
          <p className="mt-1 text-xs tracking-[0.35em] text-neutral-400 uppercase">Parking Guidance Kiosk</p>
        </header>

        <section className="mt-12 flex flex-col items-center">
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
                  className="w-full aspect-video rounded-2xl bg-black/10"
                />

                {/* Expression Camera (표정 신호 전송) */}
                <div className="mt-6">
                  <div className="flex items-center justify-between">
                    <div className="text-sm font-semibold text-neutral-900">Expression Camera</div>
                    <div className="flex gap-2">
                      <button
                        className="rounded-lg px-3 py-2 text-xs font-semibold shadow-sm border border-neutral-200 bg-white hover:bg-neutral-50 disabled:opacity-50"
                        onClick={() => setCamOn(true)}
                        disabled={camOn}
                      >
                        카메라 시작
                      </button>
                      <button
                        className="rounded-lg px-3 py-2 text-xs font-semibold shadow-sm border border-neutral-200 bg-white hover:bg-neutral-50 disabled:opacity-50"
                        onClick={() => setCamOn(false)}
                        disabled={!camOn}
                      >
                        카메라 종료
                      </button>
                    </div>
                  </div>

                  <video
                    ref={faceVideoRef}
                    autoPlay
                    playsInline
                    muted
                    className="mt-3 w-full aspect-video rounded-2xl bg-black/10"
                  />

                  <p className="mt-2 text-xs text-neutral-400">
                    표정(vision_expression)만 서버로 전송합니다. (보강: 얼굴품질게이트 + baseline 보정 + conf 재설계 + score 디버그)
                  </p>
                </div>

                <p className="mt-2 text-xs text-neutral-400">
                  {USE_AVATAR ? "bot_text → Speech SDK → Azure Avatar → (WebRTC) → <video>" : "USE_AVATAR=false (기존 WAV 재생 방식)"}
                </p>
              </div>
            </div>
          </div>

          <div className="w-full">
            <MicCards isRunning={isRunning} status={status} onToggle={onToggle} />
          </div>

          <RealtimeLogCard partialText={partialText} finalText={finalText} botText={botText} wsUrl={WS_URL} wsRef={wsRef} isWsOpen={isWsOpen} />

          <GuideCards />
          <p className="mt-8 text-center text-xs text-neutral-400"></p>
        </section>
      </div>
    </main>
  )
}
