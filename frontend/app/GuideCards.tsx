"use client"

type Props = {
  className?: string
}

export default function GuideCards({ className = "" }: Props) {
  return (
    <div className={`mt-10 w-full grid grid-cols-1 gap-8 sm:grid-cols-3 ${className}`}>
      <GuideChip title="사용 방법" items={["마이크 시작 누르기", "문의하기", "안내 듣기"]} icon="🧭" />
      <GuideChip title="지원 항목" items={["요금/정산", "출차/입차", "등록/오류 안내"]} icon="🧩" />
      <GuideChip
        title="안내"
        items={[
          "음성 인식 후 자동으로 안내 시작",
          "결제 오류 시 사유 안내 가능",
          "필요 시 직원 호출 가능",
        ]}
        icon="ℹ️"
      />
    </div>
  )
}

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