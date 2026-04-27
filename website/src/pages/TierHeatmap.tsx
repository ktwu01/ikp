import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useModels } from "../data";
import Loading from "../components/Loading";
import { TIER_COLOR } from "../util";
import type { ModelSummary, Tier } from "../types";

const TIERS: Tier[] = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"];

export default function TierHeatmap() {
  const { data, loading } = useModels();
  const [topN, setTopN] = useState(30);
  const [vendorFilter, setVendorFilter] = useState<string | null>(null);

  const { top, boxplots, vendors } = useMemo(() => {
    if (!data) return { top: [], boxplots: [], vendors: [] };
    const pool = vendorFilter ? data.filter((m) => m.vendor === vendorFilter) : data;
    const top = [...pool]
      .sort((a, b) => b.accuracy - a.accuracy)
      .slice(0, topN);
    const boxplots: { tier: Tier; q: number[]; median: number; mean: number; min: number; max: number }[] = TIERS.map((t) => {
      const xs = data.map((m) => m.tier_accuracy?.[t] ?? 0).sort((a, b) => a - b);
      const q = (p: number) => xs[Math.max(0, Math.min(xs.length - 1, Math.floor(p * (xs.length - 1))))];
      const median = q(0.5);
      return {
        tier: t,
        q: [q(0.1), q(0.25), q(0.5), q(0.75), q(0.9)],
        median,
        mean: xs.reduce((a, b) => a + b, 0) / xs.length,
        min: xs[0],
        max: xs[xs.length - 1],
      };
    });
    const vendors = [...new Set(data.map((m) => m.vendor).filter(Boolean) as string[])].sort();
    return { top, boxplots, vendors };
  }, [data, topN, vendorFilter]);

  if (loading || !data) return <Loading what="models" />;

  return (
    <div className="space-y-10">
      <header>
        <h1 className="text-4xl font-bold tracking-tight">Per-tier accuracy</h1>
        <p className="mt-3 text-ink/70 max-w-3xl">
          Every model's pattern across the 7 tiers of obscurity. T1–T2 saturate early; T3–T5 provide the
          discrimination; T6–T7 separate only the strongest frontier models.
        </p>
      </header>

      <section className="bg-white border border-ink/10 rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-ink/10 flex flex-wrap justify-between items-start gap-3">
          <div>
            <h2 className="text-lg font-semibold">Heatmap (top {topN} models)</h2>
            <p className="text-xs text-ink/50 mt-1">Color = tier accuracy. Hover a cell for count and hallucination rate.</p>
          </div>
          <div className="flex gap-2 flex-wrap">
            {[15, 25, 50, 100].map((n) => (
              <button
                key={n}
                onClick={() => setTopN(n)}
                className={`text-xs px-2.5 py-1 rounded-full border transition ${
                  topN === n ? "bg-ink text-paper border-ink" : "border-ink/20 text-ink/60 hover:bg-ink/5"
                }`}
              >
                Top {n}
              </button>
            ))}
          </div>
        </div>
        <div className="px-4 py-3 border-b border-ink/5 flex flex-wrap gap-1.5">
          <button
            onClick={() => setVendorFilter(null)}
            className={`text-xs px-2 py-0.5 rounded-full border transition ${
              vendorFilter === null ? "bg-ink text-paper border-ink" : "border-ink/20 text-ink/60 hover:bg-ink/5"
            }`}
          >
            All vendors
          </button>
          {vendors.map((v) => (
            <button
              key={v}
              onClick={() => setVendorFilter(v === vendorFilter ? null : v)}
              className={`text-xs px-2 py-0.5 rounded-full border transition ${
                vendorFilter === v ? "bg-ink text-paper border-ink" : "border-ink/20 text-ink/60 hover:bg-ink/5"
              }`}
            >
              {v}
            </button>
          ))}
        </div>
        <HeatmapTable models={top} />
      </section>

      <section className="bg-white border border-ink/10 rounded-lg p-6">
        <h2 className="text-lg font-semibold">Tier distribution across all {data.length} models</h2>
        <p className="text-xs text-ink/50 mt-1">Median / IQR / 10-90 range per tier.</p>
        <div className="mt-5 relative" style={{ height: 220 }}>
          <Boxplots stats={boxplots} />
        </div>
      </section>
    </div>
  );
}

function HeatmapTable({ models }: { models: ModelSummary[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="text-xs tabular-nums border-collapse">
        <thead>
          <tr>
            <th className="sticky left-0 bg-ink/5 z-10 px-3 py-2 text-left font-medium text-ink/60 uppercase text-[11px]">
              Model
            </th>
            <th className="bg-ink/5 px-3 py-2 text-left font-medium text-ink/60 uppercase text-[11px]">Vendor</th>
            <th className="bg-ink/5 px-3 py-2 text-right font-medium text-ink/60 uppercase text-[11px]">Acc.</th>
            {TIERS.map((t) => (
              <th
                key={t}
                className="px-3 py-2 text-center font-medium text-[11px]"
                style={{ background: TIER_COLOR[t] + "15", color: TIER_COLOR[t] }}
              >
                {t}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <tr key={m.model} className="border-t border-ink/5 hover:bg-ink/[0.03]">
              <td className="sticky left-0 bg-paper px-3 py-2">
                <Link to={`/models/${m.model}`} className="text-ink hover:text-accent font-medium">
                  {m.model}
                </Link>
                {m.thinking && <span className="ml-1.5 text-[10px] text-violet-600">think</span>}
              </td>
              <td className="px-3 py-2 text-ink/60">{m.vendor}</td>
              <td className="px-3 py-2 text-right font-medium">{(m.accuracy * 100).toFixed(1)}%</td>
              {TIERS.map((t) => {
                const acc = m.tier_accuracy?.[t] ?? 0;
                const s = m.tier_stats?.[t];
                const title = s
                  ? `${t}: ${(acc * 100).toFixed(1)}% correct · ${s.wrong} wrong · ${s.refusal} refused · ${s.total} total`
                  : `${t}: ${(acc * 100).toFixed(1)}%`;
                const intensity = Math.max(0.05, acc);
                return (
                  <td key={t} title={title} className="px-0 py-0.5">
                    <div
                      className="mx-1 rounded-sm flex items-center justify-center font-medium"
                      style={{
                        background: `rgba(37,99,235,${intensity})`,
                        color: acc > 0.55 ? "white" : "rgba(11,13,18,0.85)",
                        height: 24,
                        minWidth: 44,
                      }}
                    >
                      {(acc * 100).toFixed(0)}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Boxplots({ stats }: { stats: { tier: Tier; q: number[]; median: number; mean: number }[] }) {
  // Layout: horizontal tiers as columns
  const W = 900;
  const H = 220;
  const pad = { left: 40, right: 20, top: 20, bottom: 40 };
  const innerW = W - pad.left - pad.right;
  const innerH = H - pad.top - pad.bottom;
  const bandW = innerW / stats.length;
  const scaleY = (v: number) => pad.top + innerH - v * innerH;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {/* gridlines */}
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <g key={v}>
          <line x1={pad.left} x2={W - pad.right} y1={scaleY(v)} y2={scaleY(v)} stroke="#e5e7eb" strokeDasharray="3 3" />
          <text x={pad.left - 4} y={scaleY(v) + 3} textAnchor="end" fontSize={10} fill="#6b7280">
            {(v * 100).toFixed(0)}%
          </text>
        </g>
      ))}
      {stats.map((s, i) => {
        const cx = pad.left + bandW * i + bandW / 2;
        const color = TIER_COLOR[s.tier];
        const boxW = bandW * 0.4;
        const [p10, p25, p50, p75, p90] = s.q;
        return (
          <g key={s.tier}>
            {/* whiskers */}
            <line x1={cx} x2={cx} y1={scaleY(p10)} y2={scaleY(p90)} stroke={color} strokeWidth={1.5} />
            <line x1={cx - 10} x2={cx + 10} y1={scaleY(p10)} y2={scaleY(p10)} stroke={color} strokeWidth={1.5} />
            <line x1={cx - 10} x2={cx + 10} y1={scaleY(p90)} y2={scaleY(p90)} stroke={color} strokeWidth={1.5} />
            {/* IQR box */}
            <rect
              x={cx - boxW / 2}
              y={scaleY(p75)}
              width={boxW}
              height={scaleY(p25) - scaleY(p75)}
              fill={color}
              fillOpacity={0.35}
              stroke={color}
              strokeWidth={1.5}
            />
            {/* median */}
            <line x1={cx - boxW / 2} x2={cx + boxW / 2} y1={scaleY(p50)} y2={scaleY(p50)} stroke={color} strokeWidth={2.5} />
            {/* tier label */}
            <text x={cx} y={H - 20} textAnchor="middle" fontSize={12} fill="#6b7280" fontWeight={500}>
              {s.tier}
            </text>
            <text x={cx} y={H - 6} textAnchor="middle" fontSize={10} fill="#9ca3af">
              med {(p50 * 100).toFixed(0)}%
            </text>
          </g>
        );
      })}
    </svg>
  );
}
