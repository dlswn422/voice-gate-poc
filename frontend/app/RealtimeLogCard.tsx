"use client"

type Props = {
  partialText: string
  finalText: string
  botText: string
  wsUrl: string
  wsRef: React.MutableRefObject<WebSocket | null>
  isWsOpen: boolean
}

export default function RealtimeLogCard({
  partialText,
  finalText,
  botText,
  wsUrl,
  wsRef,
  isWsOpen,
}: Props) {
  return (
    <div className="mt-8 w-full rounded-2xl border border-white/60 bg-white/70 p-6 shadow-sm backdrop-blur">
      <div className="text-sm font-semibold text-neutral-900">실시간 로그</div>

      <div className="mt-4 space-y-3 text-sm">
        <div className="flex gap-3">
          <div className="w-24 shrink-0 text-neutral-500">PARTIAL</div>
          <div className="text-neutral-800">
            {partialText || <span className="text-neutral-400">-</span>}
          </div>
        </div>

        <div className="flex gap-3">
          <div className="w-24 shrink-0 text-neutral-500">FINAL</div>
          <div className="text-neutral-800">
            {finalText || <span className="text-neutral-400">-</span>}
          </div>
        </div>

        <div className="flex gap-3">
          <div className="w-24 shrink-0 text-neutral-500">BOT</div>
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