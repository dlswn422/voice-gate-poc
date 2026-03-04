"use client"

import { RefObject, useCallback, useRef } from "react"

const USE_AVATAR = (process.env.NEXT_PUBLIC_USE_AVATAR || "").trim().toLowerCase() === "true"

// ✅ viewer가 받는 메시지 타입 (Demo에 message listener 패치 필요)
const MSG_TYPE = "L2D_MOUTH"

// 첫 응답 앞부분 잘림 방지용: 아주 짧은 무음 WAV
const SILENT_WAV_DATA_URI =
  "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQAAAAA="

async function warmupBrowserAudio() {
  try {
    const a = new Audio(SILENT_WAV_DATA_URI)
    a.volume = 0
    await a.play()
    a.pause()
  } catch {
    // ignore
  }
}

async function decodeWithLeadingSilence(ctx: AudioContext, wavBytes: ArrayBuffer, silenceMs = 120) {
  const audio = await ctx.decodeAudioData(wavBytes.slice(0))
  const silenceSamples = Math.floor((audio.sampleRate * silenceMs) / 1000)

  const out = ctx.createBuffer(audio.numberOfChannels, audio.length + silenceSamples, audio.sampleRate)
  for (let ch = 0; ch < audio.numberOfChannels; ch++) {
    const outData = out.getChannelData(ch)
    outData.set(audio.getChannelData(ch), silenceSamples)
  }
  return out
}

export function useAvatar() {
  const startedRef = useRef(false)
  const iframeRef = useRef<RefObject<HTMLIFrameElement | null> | null>(null)

  // Audio
  const audioCtxRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const rafRef = useRef<number | null>(null)
  const speakingRef = useRef(false)

  const postMouth = (v: number) => {
    const win = iframeRef.current?.current?.contentWindow
    if (!win) return
    win.postMessage({ type: MSG_TYPE, v }, "*")
  }

  const start = useCallback(async (viewerRef: RefObject<HTMLIFrameElement | null>) => {
    if (startedRef.current) return
    if (!USE_AVATAR) return

    iframeRef.current = viewerRef

    // iframe DOM 준비 대기
    if (!viewerRef.current) {
      // React 렌더 타이밍
      await new Promise((r) => setTimeout(r, 50))
    }

    await warmupBrowserAudio()

    const AudioCtx = window.AudioContext || (window as any).webkitAudioContext
    const ctx: AudioContext = audioCtxRef.current ?? new AudioCtx()
    audioCtxRef.current = ctx

    const analyser = analyserRef.current ?? ctx.createAnalyser()
    analyser.fftSize = 2048
    analyserRef.current = analyser

    startedRef.current = true
    console.log("[AVATAR_VIEWER] started OK")
  }, [])

  const playWav = useCallback(async (wavBytes: ArrayBuffer) => {
    if (!USE_AVATAR) return
    if (!startedRef.current) return
    if (!wavBytes || wavBytes.byteLength === 0) return
    if (speakingRef.current) return
    speakingRef.current = true

    try {
      const ctx = audioCtxRef.current
      const analyser = analyserRef.current
      if (!ctx || !analyser) return

      try {
        await ctx.resume()
      } catch {}

      // ✅ 첫 응답 끊김 방지: 앞에 무음 패딩 추가
      const buffer = await decodeWithLeadingSilence(ctx, wavBytes, 140)

      const src = ctx.createBufferSource()
      src.buffer = buffer

      // analyser 연결
      src.connect(analyser)
      analyser.connect(ctx.destination)

      // mouth 루프 시작
      const data = new Uint8Array(analyser.fftSize)
      let smooth = 0

      const tick = () => {
        analyser.getByteTimeDomainData(data)
        let sum = 0
        for (let i = 0; i < data.length; i++) {
          const v = (data[i] - 128) / 128
          sum += v * v
        }
        const rms = Math.sqrt(sum / data.length)

        // 감도(3.0~4.0 사이에서 취향대로)
        const raw = Math.max(0, Math.min(1, rms * 3.2))
        // 약간 스무딩 (입 떨림 방지)
        smooth = smooth + 0.35 * (raw - smooth)

        postMouth(smooth)
        rafRef.current = requestAnimationFrame(tick)
      }

      rafRef.current = requestAnimationFrame(tick)
      src.start(0)

      await new Promise<void>((resolve) => {
        src.onended = () => resolve()
      })
    } finally {
      // 정리
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      rafRef.current = null
      postMouth(0)
      speakingRef.current = false
    }
  }, [])

  const stop = useCallback(async () => {
    startedRef.current = false
    iframeRef.current = null

    try {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    } catch {}
    rafRef.current = null

    try {
      analyserRef.current?.disconnect()
    } catch {}
    analyserRef.current = null

    try {
      await audioCtxRef.current?.close()
    } catch {}
    audioCtxRef.current = null

    speakingRef.current = false
  }, [])

  return { start, playWav, stop }
}