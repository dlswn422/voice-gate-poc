"use client";

import { useMemo } from "react";
import MicCards from "./MicCards";
import { useVoiceWs } from "./useVoiceWs";

export default function Home() {
  const { status, partial, finalText, botText, start, stop } = useVoiceWs();

  const isRunning = useMemo(
    () => status === "RUNNING" || status === "CONNECTING",
    [status]
  );

  const onToggle = async () => {
    if (isRunning) await stop();
    else await start();
  };

  return (
    <main className="min-h-screen px-6">
      <header className="pt-10 text-center select-none">
        <h1 className="text-4xl font-semibold tracking-[0.35em]">PARKMATE</h1>
        <p className="mt-1 text-xs tracking-[0.35em] text-neutral-400 uppercase">
          Parking Guidance Kiosk
        </p>
      </header>

      <div className="mt-44 flex justify-center">
        <MicCards isRunning={isRunning} onToggle={onToggle} />
      </div>

      <div className="mt-8 mx-auto max-w-[720px] text-sm text-neutral-700">
        <div className="rounded-xl border bg-white/70 p-4">
          <div className="text-xs text-neutral-500">STT (partial)</div>
          <div className="mt-1">{partial || "-"}</div>

          <div className="mt-4 text-xs text-neutral-500">STT (final)</div>
          <div className="mt-1 font-medium">{finalText || "-"}</div>

          <div className="mt-4 text-xs text-neutral-500">LLM 답변</div>
          <div className="mt-1 font-medium">{botText || "-"}</div>
        </div>
      </div>
    </main>
  );
}
