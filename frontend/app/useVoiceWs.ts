"use client";

import { useCallback, useRef, useState } from "react";

type WsStatus = "OFF" | "CONNECTING" | "RUNNING";

type ServerMsg =
  | { type: "partial"; text?: string }
  | { type: "final"; text?: string }
  | { type: "bot_text"; text?: string }
  | { type: "error"; message?: string };

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

  const stop = useCallback(async () => {
    try {
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
  }, []);

  const start = useCallback(async () => {
    if (status !== "OFF") return;
    setStatus("CONNECTING");

    const wsUrl = process.env.NEXT_PUBLIC_BACKEND_WS || "ws://localhost:8000/ws/voice";
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

      // 녹음만 (스피커 출력 연결 X)
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

        if (msg.type === "partial") setPartial(msg.text || "");
        if (msg.type === "final") {
          setFinalText(msg.text || "");
          setPartial("");
        }
        if (msg.type === "bot_text") setBotText(msg.text || "");
        return;
      }

      // binary 오디오 (WAV bytes)
      if (evt.data instanceof ArrayBuffer) {
        const blob = new Blob([evt.data], { type: "audio/wav" });
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.play().catch(() => {});
        audio.onended = () => URL.revokeObjectURL(url);
      }
    };

    ws.onerror = () => {
      stop();
    };

    ws.onclose = () => {
      setStatus("OFF");
    };
  }, [status, stop]);

  return { status, partial, finalText, botText, start, stop };
}
