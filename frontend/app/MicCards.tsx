"use client"

import { useMemo } from "react"

export type MicKey = "A" | "B" | "C" | "D"

export type MicCard = {
  key: MicKey
  title: string
  desc: string
}

type Props = {
  cards?: MicCard[]
  activeMic: MicKey | null
  onClickCard: (key: MicKey) => void
}

export default function MicCards({ cards, activeMic, onClickCard }: Props) {
  const defaultCards = useMemo<MicCard[]>(
    () => [
      { key: "A", title: "A안", desc: "Google Cloud + GPT + ElevenLabs" },
      { key: "B", title: "B안", desc: "Azure + GPT + Azure SpeechServices" },
      { key: "C", title: "C안", desc: "Faster-Whisper + Llama + Melo" },
      { key: "D", title: "D안", desc: "Naver Clova + Ollama + Microsoft Edge" },
    ],
    []
  )

  const list = cards ?? defaultCards

  return (
    <section className="mt-24 mx-auto max-w-6xl pb-28">
      <div className="grid grid-cols-4 gap-6">
        {list.map((c) => {
          const isActive = activeMic === c.key
          return (
            <div
              key={c.key}
              className={[
                "rounded-3xl bg-white/90 shadow-xl border transition p-6",
                isActive ? "border-emerald-400 ring-2 ring-emerald-200" : "border-neutral-200",
              ].join(" ")}
            >
              <div className="flex items-start justify-between">
                <div>
                  <div className="text-lg font-semibold text-neutral-900">{c.title}</div>
                  <div className="mt-1 text-sm text-neutral-500">{c.desc}</div>
                </div>

                <span
                  className={[
                    "px-3 py-1 rounded-full text-xs font-semibold",
                    isActive ? "bg-emerald-100 text-emerald-700" : "bg-neutral-100 text-neutral-500",
                  ].join(" ")}
                >
                  {isActive ? "RUNNING" : "OFF"}
                </span>
              </div>

              <button
                onClick={() => onClickCard(c.key)}
                className={[
                  "mt-8 w-full rounded-2xl py-6 flex items-center justify-center gap-3 transition",
                  isActive
                    ? "bg-emerald-500 text-white hover:bg-emerald-600"
                    : "bg-neutral-900 text-white hover:bg-neutral-800",
                ].join(" ")}
              >
                <MicIcon />
                <span className="text-base font-semibold">{isActive ? "중지" : "마이크 시작"}</span>
              </button>

              <p className="mt-3 text-xs text-neutral-400">
                {isActive ? "이 안만 실행 중 (다른 안은 자동 종료됨)" : "클릭 시 다른 안을 종료하고 이 안을 실행"}
              </p>
            </div>
          )
        })}
      </div>
    </section>
  )
}

/** 마이크 아이콘 (외부 라이브러리 없이 SVG) */
function MicIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M19 11a7 7 0 0 1-14 0"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path d="M12 18v3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <path d="M8 21h8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  )
}