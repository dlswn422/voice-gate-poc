"use client"

import { useState } from "react"
import MicCards from "./MicCards"

export default function Home() {
  const [isRunning, setIsRunning] = useState(false)

  const stopAll = async () => {
    // TODO: 백엔드 stop 호출 붙이면 여기서 처리
    setIsRunning(false)
  }

  const startMic = async () => {
    // TODO: 백엔드 start 호출 붙이면 여기서 처리
    setIsRunning(true)
  }

  const onToggle = async () => {
    if (isRunning) await stopAll()
    else await startMic()
  }

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


    </main>
  )
}