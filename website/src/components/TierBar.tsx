import type { Tier } from "../types";
import { TIER_COLOR } from "../util";

interface Props {
  tier_accuracy: Record<Tier, number>;
  showLabels?: boolean;
}

const TIERS: Tier[] = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"];

export default function TierBar({ tier_accuracy, showLabels = false }: Props) {
  return (
    <div className="flex gap-0.5 items-end h-8">
      {TIERS.map((t) => {
        const v = tier_accuracy?.[t] ?? 0;
        return (
          <div key={t} className="flex-1 flex flex-col items-center gap-0.5">
            <div
              className="w-full rounded-sm"
              style={{ backgroundColor: TIER_COLOR[t], height: `${Math.max(v * 100, 2)}%` }}
              title={`${t}: ${(v * 100).toFixed(0)}%`}
            />
            {showLabels && <div className="text-[10px] text-ink/50">{t}</div>}
          </div>
        );
      })}
    </div>
  );
}
