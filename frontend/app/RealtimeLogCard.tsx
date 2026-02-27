"use client"

type Props = {
  partialText: string
  finalText: string
  botText: string
  wsUrl: string
  wsRef: React.MutableRefObject<WebSocket | null>
  isWsOpen: boolean

  // ✅ 추가: 우측 사이드 같은 “작은 영역”에 넣기 위해
  className?: string
  scroll?: boolean
}

export default function RealtimeLogCard({
  partialText,
  finalText,
  botText,
  wsUrl,
  wsRef,
  isWsOpen,
  className = "",
  scroll = false,
}: Props) {
  return (
    <div
      className={[
        "w-full rounded-2xl border border-white/60 bg-white/70 p-5 shadow-sm backdrop-blur",
        className,
      ].join(" ")}
    >
      <div className="text-sm font-semibold text-neutral-900">실시간 로그</div>

      <div
        className={[
          "mt-4 space-y-3 text-sm",
          scroll ? "overflow-auto pr-2" : "",
        ].join(" ")}
        style={scroll ? { maxHeight: "calc(100% - 28px)" } : undefined}
      >
        <div className="flex gap-3">
          <div className="w-20 shrink-0 text-neutral-500">PARTIAL</div>
          <div className="text-neutral-800">
            {partialText || <span className="text-neutral-400">-</span>}
          </div>
        </div>

        <div className="flex gap-3">
          <div className="w-20 shrink-0 text-neutral-500">FINAL</div>
          <div className="text-neutral-800">
            {finalText || <span className="text-neutral-400">-</span>}
          </div>
        </div>

        <div className="flex gap-3">
          <div className="w-20 shrink-0 text-neutral-500">BOT</div>
          <div className="text-neutral-800">
            {botText || <span className="text-neutral-400">-</span>}
          </div>
        </div>

        <div className="pt-2 text-xs text-neutral-400">
          WS: {wsUrl} · 연결상태:{" "}
          {wsRef.current
            ? ["CONNECTING", "OPEN", "CLOSING", "CLOSED"][wsRef.current.readyState] ?? "UNKNOWN"
            : "NONE"}{" "}
          {isWsOpen ? "(open)" : ""}
        </div>
      </div>
    </div>
  )
}