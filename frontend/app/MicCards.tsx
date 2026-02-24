"use client"

type MicCardsProps = {
  isRunning: boolean;
  onToggle: () => void | Promise<void>;
};

export default function MicCards({ isRunning, onToggle }: MicCardsProps) {
  return (
    <section className="w-full max-w-[520px] pb-28">
      <div
        className={[
          "relative overflow-hidden rounded-[28px] border",
          "bg-white/65 backdrop-blur-xl",
          "shadow-[0_24px_60px_-30px_rgba(0,0,0,0.35)]",
          isRunning
            ? "border-emerald-300/60 ring-1 ring-emerald-200/70"
            : "border-white/40 ring-1 ring-black/5",
        ].join(" ")}
      >
        {/* top glow */}
        <div className="pointer-events-none absolute -top-24 left-1/2 h-48 w-[520px] -translate-x-1/2 rounded-full bg-gradient-to-r from-emerald-200/50 via-sky-200/40 to-indigo-200/40 blur-3xl" />

        <div className="relative p-7">
          {/* Header row */}
          <div className="flex items-start justify-between gap-6">
            <div>
              <div className="flex items-center gap-2">
                <span className="inline-flex h-9 w-9 items-center justify-center rounded-2xl bg-neutral-900/90 text-white shadow-sm">
                  <AzureGlyph />
                </span>
                <div className="text-lg font-semibold text-neutral-900">Azure</div>
              </div>

              <div className="mt-2 text-sm text-neutral-600">
                Azure Speech Services + GPT
              </div>

             
            </div>

            <StatusPill isRunning={isRunning} />
          </div>

          {/* Divider */}
          <div className="mt-6 h-px w-full bg-gradient-to-r from-transparent via-black/10 to-transparent" />

          {/* Main CTA */}
          <button
            onClick={onToggle}
            className={[
              "mt-6 w-full rounded-2xl px-6 py-5",
              "flex items-center justify-center gap-3",
              "transition active:scale-[0.99]",
              "shadow-[0_18px_40px_-24px_rgba(0,0,0,0.55)]",
              isRunning
                ? "bg-emerald-500 text-white hover:bg-emerald-600"
                : "bg-neutral-900 text-white hover:bg-neutral-800",
            ].join(" ")}
          >
            <MicIcon />
            <span className="text-base font-semibold tracking-tight">
              {isRunning ? "중지" : "마이크 시작"}
            </span>
          </button>

          {/* Helper text */}
          <p className="mt-3 text-xs leading-relaxed text-neutral-500">
            {isRunning
              ? "현재 Azure 세션이 실행 중입니다. (다시 누르면 즉시 중지)"
              : "버튼을 누르면 Azure 기반 음성 세션을 시작합니다."}
          </p>
        </div>

        {/* bottom edge highlight */}
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-white/40 to-transparent" />
      </div>
    </section>
  )
}

function StatusPill({ isRunning }: { isRunning: boolean }) {
  return (
    <span
      className={[
        "inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-semibold",
        "border backdrop-blur-md",
        isRunning
          ? "bg-emerald-50/70 text-emerald-700 border-emerald-200/70"
          : "bg-white/40 text-neutral-500 border-white/40",
      ].join(" ")}
    >
      <span
        className={[
          "h-2 w-2 rounded-full",
          isRunning ? "bg-emerald-500" : "bg-neutral-400",
        ].join(" ")}
      />
      {isRunning ? "RUNNING" : "OFF"}
    </span>
  )
}

function Chip({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center rounded-full border border-black/5 bg-white/50 px-3 py-1 text-xs text-neutral-600 shadow-sm">
      {children}
    </span>
  )
}

/** 마이크 아이콘 (SVG) */
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

/** Azure 느낌의 간단한 글리프 */
function AzureGlyph() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12.2 3 4 20.5h6.3l1.9-4.4h7.3L12.2 3Z"
        fill="currentColor"
        opacity="0.95"
      />
      <path
        d="M13.3 10.8 10.7 16h6.1l-3.5-5.2Z"
        fill="currentColor"
        opacity="0.75"
      />
    </svg>
  )
}