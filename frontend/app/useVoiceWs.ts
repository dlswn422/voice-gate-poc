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

  // ✅ TTS 재생 중 마이크 프레임 전송 차단(안정적으로 TTS 들리게)
  const captureMutedRef = useRef<boolean>(false);

  // ✅ Barge-in(끼어들기) 감지용(로컬 마이크 RMS)
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafIdRef = useRef<number | null>(null);

  // ✅ 현재 재생 오디오 추적
  const playingAudioRef = useRef<HTMLAudioElement | null>(null);
  const playingUrlRef = useRef<string | null>(null);
  const bufferSourceRef = useRef<AudioBufferSourceNode | null>(null);

  // ✅ “최신 답변 세대(playback_id)” 추적: 이 값과 pid가 같을 때만 재생
  const currentPidRef = useRef<number>(0);

  const stopPlayback = useCallback(() => {
    // ✅ WebAudio 재생 중단
    const bs = bufferSourceRef.current;
    if (bs) {
      try {
        bs.stop();
      } catch {}
    }
    bufferSourceRef.current = null;

    // ✅ HTMLAudio 재생 중단(구버전 fallback)
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
  }, []);;

  const cleanupAudioCapture = useCallback(async () => {
    // ✅ barge-in 모니터 중단
    try {
      if (rafIdRef.current != null) cancelAnimationFrame(rafIdRef.current);
    } catch {}
    rafIdRef.current = null;

    try {
      analyserRef.current?.disconnect();
    } catch {}
    analyserRef.current = null;
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
      captureMutedRef.current = false;

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
        // ✅ 사용자 클릭 직후 오디오 재생 가능하도록 unlock
        try { await audioCtx.resume(); } catch {}

        await audioCtx.audioWorklet.addModule("/worklets/pcm16-processor.js");

        const source = audioCtx.createMediaStreamSource(stream);
        sourceRef.current = source;
// ✅ 로컬 barge-in 감지(마이크 레벨)용 analyser
const analyser = audioCtx.createAnalyser();
analyser.fftSize = 2048;
analyserRef.current = analyser;
try {
  source.connect(analyser);
} catch {}

// ✅ barge-in 모니터 시작 (TTS 재생 중 사용자가 말하면 즉시 끊고 전송 재개)
const buf = new Float32Array(analyser.fftSize);
const BAR_GE_IN_RMS = 0.03; // 환경에 따라 조절
const HOLD_FRAMES = 3;
let hit = 0;

const tick = () => {
  try {
    analyser.getFloatTimeDomainData(buf);
    let sum = 0;
    for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    const rms = Math.sqrt(sum / buf.length);

    const audio = playingAudioRef.current;
    const isTtsPlaying = !!audio && !audio.paused;

    // TTS 재생 중 + 입력 차단 상태에서만 끼어들기 감지
    if (isTtsPlaying && captureMutedRef.current) {
      if (rms > BAR_GE_IN_RMS) hit++;
      else hit = 0;

      if (hit >= HOLD_FRAMES) {
        hit = 0;

        // ✅ 1) TTS 즉시 끊기
        stopPlayback();

        // ✅ 2) 입력 차단 해제(사용자 발화를 서버로 전송)
        captureMutedRef.current = false;

        // ✅ 3) 서버에도 현재 턴 중단 요청(LLM/TTS task 취소)
        try {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "barge_in" }));
          }
        } catch {}
      }
    } else {
      hit = 0;
    }
  } catch {
    // ignore
  }
  rafIdRef.current = requestAnimationFrame(tick);
};

rafIdRef.current = requestAnimationFrame(tick);


        const node = new AudioWorkletNode(audioCtx, "pcm16-processor");
        workletNodeRef.current = node;

        node.port.onmessage = (e: MessageEvent) => {
          const data = e.data;
          if (!(data instanceof ArrayBuffer)) return;

          // ✅ TTS 재생 중에는 마이크 프레임 전송 차단
          if (captureMutedRef.current) return;

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
          // ✅ 사용자가 말 시작(부분인식) -> 즉시 재생 중단 + 입력 허용
          stopPlayback();
          captureMutedRef.current = false;
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

            // ✅ TTS 재생 중에는 입력 차단(스피커->마이크 루프 방지)
            captureMutedRef.current = true;

            // ✅ WebAudio로 재생(브라우저 정책/코덱 이슈에 더 안정적)
            const audioCtx = audioCtxRef.current;
            if (audioCtx) {
              try {
                const decoded = await audioCtx.decodeAudioData(wavBytes.slice(0));
                const src = audioCtx.createBufferSource();
                src.buffer = decoded;
                src.connect(audioCtx.destination);
                bufferSourceRef.current = src;

                src.onended = () => {
                  // ✅ TTS 종료 -> 입력 허용
                  captureMutedRef.current = false;
                  if (bufferSourceRef.current === src) bufferSourceRef.current = null;
                };

                src.start(0);
                return;
              } catch (e) {
                console.warn("WebAudio decode/play failed, fallback to HTMLAudio:", e);
              }
            }

            // (fallback) HTMLAudio
            const blob = new Blob([wavBytes], { type: "audio/wav" });
            const url = URL.createObjectURL(blob);
            playingUrlRef.current = url;

            const audio = new Audio(url);
            playingAudioRef.current = audio;

            audio.play().catch((e) => console.warn("audio.play blocked/failed:", e));
            audio.onended = () => {
              // ✅ TTS 종료 -> 입력 허용
              captureMutedRef.current = false;
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

        // ✅ TTS 재생 중에는 입력 차단
        captureMutedRef.current = true;

        const audioCtx = audioCtxRef.current;
        if (audioCtx) {
          try {
            const decoded = await audioCtx.decodeAudioData(buf.slice(0));
            const src = audioCtx.createBufferSource();
            src.buffer = decoded;
            src.connect(audioCtx.destination);
            bufferSourceRef.current = src;

            src.onended = () => {
              captureMutedRef.current = false;
              if (bufferSourceRef.current === src) bufferSourceRef.current = null;
            };

            src.start(0);
            return;
          } catch (e) {
            console.warn("WebAudio decode/play failed (fallback), using HTMLAudio:", e);
          }
        }

        const blob = new Blob([buf], { type: "audio/wav" });
        const url = URL.createObjectURL(blob);
        playingUrlRef.current = url;

        const audio = new Audio(url);
        playingAudioRef.current = audio;

        audio.play().catch((e) => console.warn("audio.play blocked/failed:", e));
        audio.onended = () => {
          captureMutedRef.current = false;
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
        };      }
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
