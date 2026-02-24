"use client"

import { useState } from "react"
import MicCards, { MicKey } from "./MicCards"

export default function Home() {
  const [activeMic, setActiveMic] = useState<MicKey | null>(null)

  const stopAll = async () => {
 
    setActiveMic(null)
  }

  const startMic = async (key: MicKey) => {
 
    setActiveMic(key)
  }

  const onClickCard = async (key: MicKey) => {
    if (activeMic === key) {
      await stopAll()
      return
    }
    await stopAll()
    await startMic(key)
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-emerald-50 via-sky-50 to-white px-6">
      <header className="pt-10 text-center">
        <h1 className="text-4xl font-semibold tracking-[0.35em]">PARKMATE</h1>
        <p className="mt-1 text-xs tracking-[0.35em] text-neutral-400 uppercase">
          Parking Guidance Kiosk
        </p>
      </header>

      <div className="h-24" />

      <MicCards activeMic={activeMic} onClickCard={onClickCard} />

      <div className="fixed bottom-8 left-0 right-0 flex justify-center">
        <button
          onClick={stopAll}
          disabled={!activeMic}
          className={[
            "rounded-xl px-6 py-3 text-sm transition shadow-lg",
            activeMic
              ? "bg-neutral-900 text-white hover:bg-neutral-800"
              : "bg-neutral-200 text-neutral-500 cursor-not-allowed",
          ].join(" ")}
        >
          전체 종료
        </button>
      </div>
    </main>
  )
}