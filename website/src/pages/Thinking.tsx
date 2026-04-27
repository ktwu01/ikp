import { useMemo } from "react";
import { Link } from "react-router-dom";
import { useThinkingPairs } from "../data";
import Loading from "../components/Loading";
import { VendorTag } from "../components/Tag";
import { formatPercent } from "../util";

export default function Thinking() {
  const { data: pairs, loading } = useThinkingPairs();

  const stats = useMemo(() => {
    if (!pairs || pairs.length === 0) return null;
    const deltas = pairs.map((p) => p.delta);
    const mean = deltas.reduce((a, b) => a + b, 0) / deltas.length;
    const positive = deltas.filter((d) => d > 0).length;
    const negative = deltas.filter((d) => d < 0).length;
    const max = Math.max(...deltas);
    const min = Math.min(...deltas);
    return { mean, positive, negative, n: pairs.length, max, min };
  }, [pairs]);

  if (loading || !pairs) return <Loading what="thinking pairs" />;

  const sorted = [...pairs].sort((a, b) => b.delta - a.delta);
  const maxAbs = Math.max(...pairs.map((p) => Math.abs(p.delta)));

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-4xl font-bold tracking-tight">Thinking mode</h1>
        <p className="mt-3 text-ink/70 max-w-3xl">
          {pairs.length} matched base/think model pairs from the same family. Chain-of-thought reasoning
          generally helps factual recall — but not always, and the effect is usually small.
        </p>
      </header>

      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <KV label="Pairs" value={String(stats.n)} />
          <KV label="Mean Δ (think − base)" value={`${stats.mean >= 0 ? "+" : ""}${(stats.mean * 100).toFixed(1)} pp`} />
          <KV label="Positive Δ" value={`${stats.positive} / ${stats.n}`} sub="thinking helps" />
          <KV label="Range" value={`${(stats.min * 100).toFixed(1)} … +${(stats.max * 100).toFixed(1)} pp`} />
        </div>
      )}

      <div className="bg-white border border-ink/10 rounded-lg overflow-hidden">
        <div className="px-6 py-3 border-b border-ink/10">
          <h2 className="text-sm font-semibold">Pairs sorted by Δ (largest gain → largest loss)</h2>
        </div>
        <table className="w-full text-sm">
          <thead className="bg-ink/5 text-left text-ink/60 uppercase text-xs tracking-wide">
            <tr>
              <th className="px-4 py-2">Pair</th>
              <th className="px-4 py-2">Vendor</th>
              <th className="px-4 py-2 text-right">Base</th>
              <th className="px-4 py-2 text-right">Think</th>
              <th className="px-4 py-2 text-right">Δ (pp)</th>
              <th className="px-4 py-2 w-[260px]">Visual</th>
            </tr>
          </thead>
          <tbody className="tabular-nums">
            {sorted.map((p) => {
              const positive = p.delta >= 0;
              const widthPct = (Math.abs(p.delta) / maxAbs) * 50;
              return (
                <tr key={p.base} className="border-t border-ink/5 hover:bg-ink/5">
                  <td className="px-4 py-3 leading-snug">
                    <Link to={`/models/${p.base}`} className="text-ink hover:text-accent font-medium">
                      {p.base}
                    </Link>
                    <div className="text-xs text-ink/50">
                      vs <Link to={`/models/${p.think}`} className="hover:text-accent text-violet-600">{p.think}</Link>
                    </div>
                  </td>
                  <td className="px-4 py-3"><VendorTag vendor={p.vendor} /></td>
                  <td className="px-4 py-3 text-right">{formatPercent(p.base_acc)}</td>
                  <td className="px-4 py-3 text-right">{formatPercent(p.think_acc)}</td>
                  <td className={`px-4 py-3 text-right font-medium ${positive ? "text-emerald-600" : "text-red-600"}`}>
                    {positive ? "+" : ""}{(p.delta * 100).toFixed(1)}
                  </td>
                  <td className="px-4 py-3">
                    <div className="relative h-3 bg-ink/[0.04] rounded">
                      <div className="absolute top-0 bottom-0 w-px bg-ink/30" style={{ left: "50%" }} />
                      <div
                        className={`absolute top-0 bottom-0 ${positive ? "bg-emerald-500" : "bg-red-500"} rounded-sm`}
                        style={{
                          left: positive ? "50%" : `${50 - widthPct}%`,
                          width: `${widthPct}%`,
                        }}
                      />
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="bg-accent/5 border border-accent/20 rounded-lg p-5 text-sm text-ink/80 leading-relaxed">
        <strong className="text-accent">Reading the table:</strong> A positive Δ means the thinking variant
        scored higher on IKP than its non-thinking sibling. Note that for pure factual recall (which IKP
        measures), reasoning often provides little benefit — and occasionally hurts when chain-of-thought
        introduces hallucinated intermediate steps.
      </div>
    </div>
  );
}

function KV({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-white border border-ink/10 rounded-lg p-4">
      <div className="text-[11px] uppercase tracking-wider text-ink/50 font-medium">{label}</div>
      <div className="mt-1 text-2xl font-bold tabular-nums">{value}</div>
      {sub && <div className="text-xs text-ink/50">{sub}</div>}
    </div>
  );
}
