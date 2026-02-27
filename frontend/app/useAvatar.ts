"use client"

import { RefObject, useCallback, useRef } from "react"

export function useAvatar() {
  const pcRef = useRef<RTCPeerConnection | null>(null)
  const synthesizerRef = useRef<any>(null)
  const startedRef = useRef(false)

  const start = useCallback(async (videoRef: RefObject<HTMLVideoElement>) => {
    if (startedRef.current) return
    if (process.env.NEXT_PUBLIC_USE_AVATAR !== "true") return

    const key = process.env.NEXT_PUBLIC_AZURE_SPEECH_KEY?.trim()
    const region = process.env.NEXT_PUBLIC_AZURE_SPEECH_REGION?.trim()
    const character = process.env.NEXT_PUBLIC_AZURE_AVATAR_CHARACTER?.trim() || "lisa"
    const style = process.env.NEXT_PUBLIC_AZURE_AVATAR_STYLE?.trim() || "casual-sitting"
    if (!key || !region) throw new Error("Missing NEXT_PUBLIC_AZURE_SPEECH_KEY / REGION")

    const speechsdk: any = await import("microsoft-cognitiveservices-speech-sdk")

    const speechConfig = speechsdk.SpeechConfig.fromSubscription(key, region)
    const avatarConfig = new speechsdk.AvatarConfig(character, style)
    const avatarSynthesizer = new speechsdk.AvatarSynthesizer(speechConfig, avatarConfig)
    synthesizerRef.current = avatarSynthesizer

    const pc = new RTCPeerConnection({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    })
    pcRef.current = pc

    pc.addTransceiver("video", { direction: "recvonly" })
    pc.addTransceiver("audio", { direction: "recvonly" })

    pc.ontrack = (ev) => {
      const videoEl = videoRef.current
      const stream = ev.streams?.[0]
      if (!videoEl || !stream) return
      if (videoEl.srcObject !== stream) {
        videoEl.srcObject = stream
        videoEl.play().catch(() => {})
      }
    }

    await avatarSynthesizer.startAvatarAsync(pc)
    startedRef.current = true
  }, [])

  const speak = useCallback(async (text: string) => {
    if (process.env.NEXT_PUBLIC_USE_AVATAR !== "true") return
    const s = synthesizerRef.current
    if (!s || !startedRef.current) return
    if (!text?.trim()) return
    await s.speakTextAsync(text)
  }, [])

  const stop = useCallback(async () => {
    startedRef.current = false
    try {
      synthesizerRef.current?.close?.()
    } catch {}
    synthesizerRef.current = null

    try {
      pcRef.current?.close()
    } catch {}
    pcRef.current = null
  }, [])

  return { start, speak, stop }
}