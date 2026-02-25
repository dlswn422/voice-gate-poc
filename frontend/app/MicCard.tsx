"use client"

type Props = {
  isRunning: boolean
  status: "OFF" | "LISTENING" | "THINKING" | "SPEAKING"
  onToggle: () => void
}

export default function MicCards({ isRunning, status, onToggle }: Props) {
  const statusLabel = status === "OFF" ? "OFF" : status

  return (
    <div
      className={[
        "w-full rounded-3xl border shadow-xl backdrop-blur transition-colors duration-300",
        isRunning
          ? "border-emerald-200/80 bg-emerald-50/80 shadow-emerald-200/40"
          : "border-white/70 bg-white/70",
      ].join(" ")}
    >
      <div className="px-12 pt-10">
        <div className="flex items-start justify-between">
          {/* 왼쪽 작은 아이콘/배지 */}
          <div className="flex items-center gap-3">
            <div
              className={[
                "size-11 rounded-full grid place-items-center shadow transition-colors",
                isRunning ? "bg-emerald-600 text-white" : "bg-neutral-900 text-white",
              ].join(" ")}
            >
              <span className="text-sm font-bold">A</span>
            </div>

            <div className="leading-tight">
              <div
                className={[
                  "text-sm font-semibold transition-colors",
                  isRunning ? "text-emerald-700" : "text-neutral-900",
                ].join(" ")}
              >
                음성 안내
              </div>
              <div className="text-xs text-neutral-500">버튼을 눌러 시작/중지</div>
            </div>
          </div>

          {/* 오른쪽 상태 + 툴팁 */}
          <div className="flex items-center gap-2">
            <div
              className={[
                "px-3 py-1 rounded-full text-xs font-semibold border",
                statusLabel === "OFF"
                  ? "bg-neutral-100 text-neutral-500 border-neutral-200"
                  : "bg-emerald-50 text-emerald-700 border-emerald-200",
              ].join(" ")}
            >
              {statusLabel}
            </div>

                    <button
            type="button"
            className={[
              "size-8 rounded-full border transition-colors duration-200 grid place-items-center",
              isRunning
                ? "border-emerald-200 bg-emerald-50 text-emerald-700 hover:bg-emerald-100"
                : "border-neutral-200 bg-white/70 text-neutral-500 hover:text-neutral-800 hover:bg-white",
            ].join(" ")}
            title="소음이 있으면 가까이 말씀해 주세요. 짧게 핵심만 말하면 더 빠르게 안내됩니다."
            aria-label="도움말"
          >
            i
          </button>
          </div>
        </div>

        <button
          onClick={onToggle}
          className={[
          "mt-6 w-full rounded-2xl py-6 flex items-center justify-center gap-3",
          "transition-all duration-200 ease-out",
          "cursor-pointer select-none",
          "shadow-lg hover:shadow-xl active:shadow-md",
            "hover:-translate-y-[2px] hover:scale-[1.01]",
            "active:translate-y-[1px] active:scale-[0.99]",

          isRunning
            ? "bg-emerald-600 text-white hover:bg-emerald-500 active:bg-emerald-700"
            : "bg-neutral-900 text-white hover:bg-neutral-800 active:bg-neutral-950",
        ].join(" ")}
        >
          <MicIcon />
          <span className="text-base font-semibold">{isRunning ? "중지" : "마이크 시작"}</span>
        </button>

        <p className="mt-3 text-xs text-neutral-400">
          {isRunning
            ? "음성 안내가 진행 중입니다. 필요하면 다시 눌러 중지하세요."
            : "버튼을 누르면 음성 안내가 시작됩니다."}
        </p>
      </div>

      <div className="px-12 pb-9">
        <div className="mt-4 h-px bg-neutral-200/70" />
        <div className="mt-3 text-[11px] text-neutral-400"></div>
      </div>
    </div>
  )
}

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