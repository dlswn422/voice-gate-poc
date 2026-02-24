"use client"

import { useState } from "react"
import MicCards from "./MicCards"

type Status = "OFF" | "LISTENING" | "THINKING" | "SPEAKING"

export default function Home() {
  const [isRunning, setIsRunning] = useState(false)

  // 데모용 상태 (백엔드 붙이면 여기만 바꾸면 됨)
  const status: Status = isRunning ? "LISTENING" : "OFF"

  const onToggle = () => {
    setIsRunning((v) => !v)
  }

  return (
    <main className="min-h-screen px-6 flex justify-center">
      {/* ✅ 전체 콘텐츠를 화면 중앙에 고정 */}
      <div className="w-full max-w-[1200px]">
        <header className="pt-10 text-center select-none">
          <h1 className="text-4xl font-semibold tracking-[0.35em]">PARKMATE</h1>
          <p className="mt-1 text-xs tracking-[0.35em] text-neutral-400 uppercase">
            Parking Guidance Kiosk
          </p>
        </header>

        {/* ✅ 가운데 메인 콘텐츠 */}
        <section className="mt-24 flex flex-col items-center">
          {/* MicCards는 넓게(그대로 두면 내부 w-full 기준으로 꽉 참) */}
          <div className="w-full">
            <MicCards isRunning={isRunning} status={status} onToggle={onToggle} />
          </div>

          {/* MicCards와 3개 카드 간격 */}
          <div className="mt-20 w-full grid grid-cols-1 gap-8 sm:grid-cols-3">
            <GuideChip
              title="사용 방법"
              items={["마이크 시작 누르기", "문의하기", "안내 듣기"]}
              icon="🧭"
            />
            <GuideChip
              title="지원 항목"
              items={["요금/정산", "출차/입차", "등록/오류 안내"]}
              icon="🧩"
            />
            <GuideChip
              title="안내"
              items={[
                "음성 인식 후 자동으로 안내 시작",
                "결제 오류 시 사유 안내 가능",
                "필요 시 직원 호출이 가능",
              ]}
              icon="ℹ️"
            />
          </div>

          <p className="mt-12 text-center text-xs text-neutral-400">
            * 데모 화면입니다. 버튼을 누르면 음성 안내가 시작됩니다.
          </p>
        </section>
      </div>
    </main>
  )
}

/* ===============================
   GuideChip
=============================== */

function GuideChip({
  title,
  items,
  icon,
}: {
  title: string
  items: string[]
  icon: string
}) {
  return (
    <div className="min-h-[180px] rounded-2xl border border-white/60 bg-white/70 p-7 shadow-sm backdrop-blur">
      <div className="flex items-center gap-3">
        <span className="text-xl" aria-hidden="true">
          {icon}
        </span>
        <div className="text-base font-semibold text-neutral-900">{title}</div>
      </div>

      <ul className="mt-5 space-y-3 text-sm text-neutral-600">
        {items.map((t) => (
          <li key={t} className="flex items-start gap-3">
            <span className="mt-[7px] inline-block size-2 rounded-full bg-neutral-300" />
            <span>{t}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}