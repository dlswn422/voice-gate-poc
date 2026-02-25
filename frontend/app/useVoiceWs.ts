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

  // ✅ 현재 재생 오디오 추적
  const playingAudioRef = useRef<HTMLAudioElement | null>(null);
  const playingUrlRef = useRef<string | null>(null);

  // ✅ “최신 답변 세대(playback_id)” 추적: 이 값과 pid가 같을 때만 재생
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

  const cleanupAudioCapture = useCallback(async () => {
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

    audioCtxRef.current = null;
    workletNodeRef.current = null;
    sourceRef.current = null;
    streamRef.current = null;
  }, []);

  const stop = useCallback(async () => {
    try {
      // ✅ 재생 중단
      stopPlayback();

      // audio capture stop
      await cleanupAudioCapture();

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

      setStatus("OFF");
      setPartial("");
    }
  }, [cleanupAudioCapture, stopPlayback]);

  const start = useCallback(async () => {
    if (status !== "OFF") return;

    setStatus("CONNECTING");
    setPartial("");
    setFinalText("");
    setBotText("");

    // 최신 pid 초기화
    currentPidRef.current = 0;

    const wsUrl = process.env.NEXT_PUBLIC_BACKEND_WS?.trim() || "ws://localhost:8000/ws/voice";
    console.log("WS URL:", wsUrl);

    let ws: WebSocket;
    try {
      ws = new WebSocket(wsUrl);
    } catch (e) {
      console.error("WebSocket init failed:", e);
      setStatus("OFF");
      return;
    }

    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = async () => {
      try {
        ws.send(JSON.stringify({ type: "start", sample_rate: 16000, format: "pcm16" }));

        // 마이크 권한
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });
        streamRef.current = stream;

        // iOS/Safari 호환: sampleRate 등은 브라우저가 결정
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

        // 녹음만 (스피커 출력 연결 X)
        source.connect(node);

        setStatus("RUNNING");
      } catch (err) {
        console.error(err);
        await stop();
      }
    };

    ws.onmessage = (evt: MessageEvent) => {
      // -----------------------
      // 1) JSON 텍스트 메시지
      // -----------------------
      if (typeof evt.data === "string") {
        let msg: ServerMsg | null = null;
        try {
          msg = JSON.parse(evt.data) as ServerMsg;
        } catch {
          return;
        }
        if (!msg) return;

        if (msg.type === "stopPlayback") {
          // ✅ 서버가 새로운 세대를 시작했음을 알림 -> 최신 pid 갱신 + 즉시 재생 중단
          if (typeof msg.playback_id === "number") {
            currentPidRef.current = msg.playback_id;
          } else {
            // pid가 없으면 그냥 증가(보수적)
            currentPidRef.current += 1;
          }
          stopPlayback();
          return;
        }

        if (msg.type === "partial") {
          // ✅ 사용자가 말 시작(부분인식) -> 즉시 재생 중단
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

      // -----------------------
      // 2) Binary 오디오 프레임
      //    "WAV0" + uint32 pid + wav bytes
      // -----------------------
      if (evt.data instanceof ArrayBuffer) {
        const buf = evt.data;
        const u8 = new Uint8Array(buf);

        if (u8.length >= 8) {
          const magic =
            String.fromCharCode(u8[0]) +
            String.fromCharCode(u8[1]) +
            String.fromCharCode(u8[2]) +
            String.fromCharCode(u8[3]);

          if (magic === "WAV0") {
            const dv = new DataView(buf);
            const pid = dv.getUint32(4, true); // little-endian
            const wavBytes = buf.slice(8);

            // ✅ 최신 pid만 재생 (늦게 온 이전 오디오 완전 차단)
            if (pid !== currentPidRef.current) {
              // console.log("drop audio pid", pid, "current", currentPidRef.current);
              return;
            }

            // ✅ 새 오디오가 오면 이전 재생 무조건 중단 후 재생
            stopPlayback();

            const blob = new Blob([wavBytes], { type: "audio/wav" });
            const url = URL.createObjectURL(blob);
            playingUrlRef.current = url;

            const audio = new Audio(url);
            playingAudioRef.current = audio;

            audio.play().catch(() => {});
            audio.onended = () => {
              // 끝났는데 이미 다른 url로 교체됐을 수도 있으니 체크
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
            return;
          }
        }

        // -----------------------
        // (fallback) 구버전: 헤더 없는 wav bytes
        // -----------------------
        stopPlayback();

        const blob = new Blob([buf], { type: "audio/wav" });
        const url = URL.createObjectURL(blob);
        playingUrlRef.current = url;

        const audio = new Audio(url);
        playingAudioRef.current = audio;

        audio.play().catch(() => {});
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
