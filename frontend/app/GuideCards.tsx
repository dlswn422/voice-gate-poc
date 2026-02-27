"use client"

type Props = {
  className?: string
}

export default function GuideCards({ className = "" }: Props) {
  return (
    <div className={`w-full flex flex-col gap-6 ${className}`}>
      <GuideChip title="사용 방법" items={["마이크 시작 누르기", "문의하기", "안내 듣기"]} icon="🧭" />
      <GuideChip title="지원 항목" items={["인기 메뉴 및 오늘의 추천", "대기 현황 및 좌석 안내", "주문 · 포장 · 결제 문의"]} icon="🧩" />
      <GuideChip
        title="안내"
        items={[
          "음성으로 메뉴 상세 설명 제공",
          "재료 · 알레르기 정보 안내",
          "매장 이용 방법 안내",
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
    <div className="min-h-[150px] rounded-2xl border border-white/60 bg-white/70 p-5 shadow-sm backdrop-blur">
      <div className="flex items-center gap-3">
        <span className="text-xl" aria-hidden="true">
          {icon}
        </span>
        <div className="text-base font-semibold text-neutral-900">{title}</div>
      </div>

      <ul className="mt-4 space-y-2 text-sm text-neutral-600">
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