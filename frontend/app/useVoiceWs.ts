"use client";

import { useCallback, useRef, useState } from "react";

type WsStatus = "OFF" | "CONNECTING" | "RUNNING";

type ServerMsg =
  | { type: "partial"; text?: string; playback_id?: number }
  | { type: "final"; text?: string; playback_id?: number }
  | { type: "bot_text"; text?: string; playback_id?: number }
  | { type: "stopPlayback"; playback_id?: number } // ✅ server.py에서 오는 타입
  | { type: "barge_in" } // ✅ 구버전 호환
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

  // ✅ 현재 재생 오디오 추적
  const playingAudioRef = useRef<HTMLAudioElement | null>(null);
  const playingUrlRef = useRef<string | null>(null);

  const stopPlayback = useCallback(() => {
    const audio = playingAudioRef.current;
    if (audio) {
      try {
        audio.pause();
        audio.currentTime = 0;
      } catch {}
    }
    playingAudioRef.current = null;

    const url = playingUrlRef.current;
    if (url) {
      try {
        URL.revokeObjectURL(url);
      } catch {}
    }
    playingUrlRef.current = null;
  }, []);

  const stop = useCallback(async () => {
    try {
      stopPlayback();

      // audio capture stop
      try {
        workletNodeRef.current?.disconnect();
      } catch {}
      try {
        sourceRef.current?.disconnect();
      } catch {}
      try {
        await audioCtxRef.current?.close().catch(() => {});
      } catch {}
      try {
        streamRef.current?.getTracks().forEach((t) => t.stop());
      } catch {}

      // ws stop
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(JSON.stringify({ type: "stop" }));
        } catch {}
      }
      try {
        ws?.close();
      } catch {}
    } finally {
      wsRef.current = null;
      audioCtxRef.current = null;
      workletNodeRef.current = null;
      sourceRef.current = null;
      streamRef.current = null;

      setStatus("OFF");
      setPartial("");
    }
  }, [stopPlayback]);

  const start = useCallback(async () => {
    if (status !== "OFF") return;
    setStatus("CONNECTING");

    const wsUrl =
      process.env.NEXT_PUBLIC_BACKEND_WS?.trim() || "ws://localhost:8000/ws/voice";
    console.log("WS URL:", wsUrl);

    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = async () => {
      try {
        ws.send(JSON.stringify({ type: "start", sample_rate: 16000, format: "pcm16" }));

        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });
        streamRef.current = stream;

        // ✅ sampleRate 힌트(브라우저는 무시할 수도 있지만 시도)
        const audioCtx = new AudioContext({ sampleRate: 16000 });
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
      } catch (err) {
        console.error(err);
        stop();
      }
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

        // ✅ 서버가 "새 답변 시작"을 알리면 즉시 기존 재생 중단
        if (msg.type === "stopPlayback" || msg.type === "barge_in") {
          stopPlayback();
          return;
        }

        // ✅ 사용자가 말하기 시작하면(Partial) 재생 중단 (바지인)
        if (msg.type === "partial") {
          stopPlayback();
          setPartial(msg.text || "");
          return;
        }

        if (msg.type === "final") {
          setFinalText(msg.text || "");
          setPartial("");
          return;
        }

        if (msg.type === "bot_text") {
          setBotText(msg.text || "");
          return;
        }

        if (msg.type === "error") {
          console.error("Server error:", msg.message);
          stop();
          return;
        }

        return;
      }

      // binary 오디오 (WAV bytes)
      if (evt.data instanceof ArrayBuffer) {
        // ✅ 새 오디오가 오면 무조건 이전 재생 멈추고 새로 재생
        stopPlayback();

        const blob = new Blob([evt.data], { type: "audio/wav" });
        const url = URL.createObjectURL(blob);
        playingUrlRef.current = url;

        const audio = new Audio(url);
        playingAudioRef.current = audio;

        audio.play().catch(() => {});
        audio.onended = () => {
          // 현재 재생 중인 url이 이 url이면 정리
          if (playingUrlRef.current === url) {
            try {
              URL.revokeObjectURL(url);
            } catch {}
            playingUrlRef.current = null;
            playingAudioRef.current = null;
          } else {
            // 이미 다른 오디오로 교체됐으면 이것만 정리
            try {
              URL.revokeObjectURL(url);
            } catch {}
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
  }, [status, stop, stopPlayback]);

  return { status, partial, finalText, botText, start, stop };
}
