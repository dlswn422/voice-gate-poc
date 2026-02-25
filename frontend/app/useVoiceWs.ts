"use client";

import { useCallback, useRef, useState } from "react";

type WsStatus = "OFF" | "CONNECTING" | "RUNNING";

type ServerMsg =
  | { type: "partial"; text?: string }
  | { type: "final"; text?: string }
  | { type: "bot_text"; text?: string }
  | { type: "error"; message?: string }
  | { type: "barge_in" }; // (서버에서 보내면 잡음)

export function useVoiceWs() {
  const [status, setStatus] = useState<WsStatus>("OFF");
  const [partial, setPartial] = useState("");
  const [finalText, setFinalText] = useState("");
  const [botText, setBotText] = useState("");

  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // ✅ 오디오 1개만 유지 + URL도 추적해서 정리
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioUrlRef = useRef<string | null>(null);

  const stopAudio = useCallback(() => {
    const a = audioRef.current;
    if (a) {
      try {
        a.pause();
        a.currentTime = 0;
        a.src = "";
        a.load();
      } catch {}
    }
    if (audioUrlRef.current) {
      try {
        URL.revokeObjectURL(audioUrlRef.current);
      } catch {}
      audioUrlRef.current = null;
    }
  }, []);

  const stop = useCallback(async () => {
    try {
      // ✅ 현재 재생 중인 TTS도 즉시 중단
      stopAudio();

      // audio stop
      workletNodeRef.current?.disconnect();
      sourceRef.current?.disconnect();
      await audioCtxRef.current?.close().catch(() => {});
      streamRef.current?.getTracks().forEach((t) => t.stop());

      // ws stop
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "stop" }));
      }
      ws?.close();
    } finally {
      wsRef.current = null;
      audioCtxRef.current = null;
      workletNodeRef.current = null;
      sourceRef.current = null;
      streamRef.current = null;
      setStatus("OFF");
      setPartial("");
    }
  }, [stopAudio]);

  const start = useCallback(async () => {
    if (status !== "OFF") return;
    setStatus("CONNECTING");

    const wsUrl = process.env.NEXT_PUBLIC_BACKEND_WS?.trim() || "ws://localhost:8000/ws/voice";
    console.log("WS URL:", wsUrl);

    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = async () => {
      ws.send(JSON.stringify({ type: "start", sample_rate: 16000, format: "pcm16" }));

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;

      const audioCtx = new AudioContext();
      audioCtxRef.current = audioCtx;

      await audioCtx.audioWorklet.addModule("/worklets/pcm16-processor.js");

      const source = audioCtx.createMediaStreamSource(stream);
      sourceRef.current = source;

      const node = new AudioWorkletNode(audioCtx, "pcm16-processor");
      workletNodeRef.current = node;

      node.port.onmessage = (e: MessageEvent) => {
        const data = e.data;
        if (!(data instanceof ArrayBuffer)) return;
        if (ws.readyState === WebSocket.OPEN) ws.send(data);
      };

      source.connect(node);
      setStatus("RUNNING");
    };

    ws.onmessage = (evt: MessageEvent) => {
      // JSON 텍스트 메시지
      if (typeof evt.data === "string") {
        let msg: ServerMsg | null = null;
        try {
          msg = JSON.parse(evt.data) as ServerMsg;
        } catch {
          return;
        }
        if (!msg) return;

        if (msg.type === "partial") {
          // ✅ 사용자가 말하기 시작하면 기존 TTS 즉시 중단 (겹침 방지 핵심)
          stopAudio();
          setPartial(msg.text || "");
        }

        if (msg.type === "final") {
          setFinalText(msg.text || "");
          setPartial("");
        }

        if (msg.type === "bot_text") {
          setBotText(msg.text || "");
        }

        if (msg.type === "error") {
          console.error(msg.message);
        }

        if (msg.type === "barge_in") {
          // 서버가 barge-in을 보내면 이것도 즉시 stop
          stopAudio();
        }

        return;
      }

      // binary 오디오 (WAV bytes)
      if (evt.data instanceof ArrayBuffer) {
        // ✅ 새 TTS가 오면 이전 TTS 즉시 중단 (겹침 방지 핵심)
        stopAudio();

        if (!audioRef.current) audioRef.current = new Audio();

        const blob = new Blob([evt.data], { type: "audio/wav" });
        const url = URL.createObjectURL(blob);
        audioUrlRef.current = url;

        const a = audioRef.current;
        a.src = url;

        a.onended = () => {
          if (audioUrlRef.current === url) {
            try {
              URL.revokeObjectURL(url);
            } catch {}
            audioUrlRef.current = null;
          }
        };

        a.play().catch(() => {});
      }
    };

    ws.onerror = () => {
      stop();
    };

    ws.onclose = () => {
      setStatus("OFF");
    };
  }, [status, stop, stopAudio]);

  return { status, partial, finalText, botText, start, stop };
}
