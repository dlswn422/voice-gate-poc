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

  // ✅ TTS 재생 겹침 방지용: 현재 재생 중인 오디오 1개만 유지
  const ttsAudioRef = useRef<HTMLAudioElement | null>(null);
  const ttsUrlRef = useRef<string | null>(null);

  const stopTts = useCallback(() => {
    try {
      if (ttsAudioRef.current) {
        ttsAudioRef.current.pause();
        ttsAudioRef.current.currentTime = 0;
      }
    } catch {}
    ttsAudioRef.current = null;

    try {
      if (ttsUrlRef.current) URL.revokeObjectURL(ttsUrlRef.current);
    } catch {}
    ttsUrlRef.current = null;
  }, []);

  const stop = useCallback(async () => {
    try {
      // ✅ TTS stop 먼저
      stopTts();

      // audio capture stop
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
  }, [stopTts]);

  const start = useCallback(async () => {
    if (status !== "OFF") return;
    setStatus("CONNECTING");

    const wsUrl = process.env.NEXT_PUBLIC_BACKEND_WS?.trim();
    if (!wsUrl) {
      console.error("❌ NEXT_PUBLIC_BACKEND_WS is undefined");
      setStatus("OFF");
      return;
    }

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

        if (msg.type === "error") {
          // 서버가 "이미 상담 중" 같은 에러를 보내면 UI에 보여주고 종료
          setBotText(msg.message || "서버 오류가 발생했습니다.");
          stop();
          return;
        }

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
        // ✅ 새 TTS가 오면 기존 TTS 즉시 중단 (겹침 방지)
        stopTts();

        const blob = new Blob([evt.data], { type: "audio/wav" });
        const url = URL.createObjectURL(blob);
        ttsUrlRef.current = url;

        const audio = new Audio(url);
        ttsAudioRef.current = audio;

        audio.play().catch(() => {});
        audio.onended = () => {
          try {
            if (ttsUrlRef.current === url) {
              URL.revokeObjectURL(url);
              ttsUrlRef.current = null;
            }
          } catch {}
          if (ttsAudioRef.current === audio) {
            ttsAudioRef.current = null;
          }
        };
      }
    };

    ws.onerror = () => {
      stop();
    };

    ws.onclose = () => {
      setStatus("OFF");
    };
  }, [status, stop, stopTts]);

  return { status, partial, finalText, botText, start, stop };
}
