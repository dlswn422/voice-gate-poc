"use client";

import { useCallback, useRef, useState } from "react";

type WsStatus = "OFF" | "CONNECTING" | "RUNNING";

type ServerMsg =
  | { type: "partial"; text?: string; playback_id?: number }
  | { type: "final"; text?: string; playback_id?: number }
  | { type: "bot_text"; text?: string; playback_id?: number }
  | { type: "stopPlayback"; playback_id?: number }
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

  const playingAudioRef = useRef<HTMLAudioElement | null>(null);
  const playingUrlRef = useRef<string | null>(null);

  // ✅ 최신 답변 세대(pid)
  const currentPidRef = useRef<number>(0);

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

  const playWavBytes = useCallback(
    (wavBytes: ArrayBuffer) => {
      stopPlayback();

      const blob = new Blob([wavBytes], { type: "audio/wav" });
      const url = URL.createObjectURL(blob);
      playingUrlRef.current = url;

      const audio = new Audio(url);
      playingAudioRef.current = audio;

      audio.play().catch((e) => {
        console.warn("audio.play() failed:", e);
      });

      audio.onended = () => {
        if (playingUrlRef.current === url) {
          try {
            URL.revokeObjectURL(url);
          } catch {}
          playingUrlRef.current = null;
          playingAudioRef.current = null;
        } else {
          try {
            URL.revokeObjectURL(url);
          } catch {}
        }
      };
    },
    [stopPlayback]
  );

  const stop = useCallback(async () => {
    try {
      stopPlayback();

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
    setPartial("");
    setFinalText("");
    setBotText("");

    currentPidRef.current = 0;

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
          if (ws.readyState === WebSocket.OPEN) {
            try {
              ws.send(data);
            } catch {}
          }
        };

        source.connect(node);
        setStatus("RUNNING");
      } catch (err) {
        console.error(err);
        await stop();
      }
    };

    ws.onmessage = async (evt: MessageEvent) => {
      // 1) JSON
      if (typeof evt.data === "string") {
        let msg: ServerMsg | null = null;
        try {
          msg = JSON.parse(evt.data) as ServerMsg;
        } catch {
          return;
        }
        if (!msg) return;

        if (msg.type === "stopPlayback") {
          // ✅ 서버가 알려준 pid가 있으면 반영, 없으면 +1
          if (typeof msg.playback_id === "number") {
            currentPidRef.current = msg.playback_id;
          } else {
            currentPidRef.current += 1;
          }
          stopPlayback();
          return;
        }

        if (msg.type === "partial") {
          // 말 시작하면 재생 중단 (barge-in)
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

      // 2) Binary (ArrayBuffer or Blob)
      let buf: ArrayBuffer;
      if (evt.data instanceof ArrayBuffer) {
        buf = evt.data;
      } else if (evt.data instanceof Blob) {
        buf = await evt.data.arrayBuffer();
      } else {
        return;
      }

      const u8 = new Uint8Array(buf);

      // "WAV0" + uint32 pid + wav bytes
      if (u8.length >= 8) {
        const magic =
          String.fromCharCode(u8[0]) +
          String.fromCharCode(u8[1]) +
          String.fromCharCode(u8[2]) +
          String.fromCharCode(u8[3]);

        if (magic === "WAV0") {
          const dv = new DataView(buf);
          const pid = dv.getUint32(4, true);
          const wavBytes = buf.slice(8);

          // ✅ 여기 핵심: pid를 “오디오 수신 시점”에도 최신으로 갱신
          // - currentPid가 0이면 최초 pid로 세팅
          // - pid가 더 크면 최신 pid로 갱신 (이전 오디오는 자동 드랍)
          if (currentPidRef.current === 0 || pid > currentPidRef.current) {
            currentPidRef.current = pid;
          }

          // ✅ 최신 pid만 재생
          if (pid !== currentPidRef.current) {
            return;
          }

          playWavBytes(wavBytes);
          return;
        }
      }

      // fallback: 헤더 없는 wav
      playWavBytes(buf);
    };

    ws.onerror = () => {
      stop();
    };

    ws.onclose = () => {
      setStatus("OFF");
    };
  }, [status, stop, stopPlayback, playWavBytes]);

  return { status, partial, finalText, botText, start, stop };
}
