import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useGenerations } from "../data";
import Loading from "../components/Loading";
import { TIER_COLOR } from "../util";

const VARIANT_COLOR: Record<string, string> = {
  base: "#1d4ed8",
  mini: "#f59e0b",
  nano: "#dc2626",
  pro: "#7c3aed",
  think: "#059669",
};

export default function Generations() {
  const { data, loading } = useGenerations();
  const [family, setFamily] = useState<string | null>(null);

  const active = useMemo(() => {
    if (!data) return null;
    return data.families.find((f) => f.family === family) ?? data.families[0];
  }, [data, family]);

  if (loading || !data) return <Loading what="generations data" />;

  return (
    <div className="space-y-10">
      <header>
        <h1 className="text-4xl font-bold tracking-tight">Generations &amp; families</h1>
        <p className="mt-3 text-ink/70 max-w-3xl">
          Within-family trajectories across model generations. Most families improve steadily, but several show
          regressions when vendors optimize for efficiency or safety at the expense of factual recall.
        </p>
      </header>

      <section className="bg-white border border-ink/10 rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-ink/10">
          <h2 className="text-lg font-semibold">Cross-generation evolution</h2>
          <p className="text-xs text-ink/50 mt-1">Accuracy on IKP across consecutive generations. Select a family.</p>
        </div>
        <div className="flex flex-wrap gap-1.5 px-4 py-3 border-b border-ink/5">
          {data.families.map((f) => (
            <button
              key={f.family}
              onClick={() => setFamily(f.family === family ? null : f.family)}
              className={`text-xs px-2.5 py-1 rounded-full border transition ${
                (active?.family ?? null) === f.family
                  ? "bg-ink text-paper border-ink"
                  : "border-ink/20 text-ink/60 hover:bg-ink/5"
              }`}
            >
              {f.family}
            </button>
          ))}
        </div>

        {active && (
          <div className="p-6">
            <h3 className="text-sm font-semibold text-ink/70">{active.family}</h3>
            <div className="h-[340px] -ml-5 mt-3">
              <ResponsiveContainer>
                <LineChart data={active.chain.map((m, i) => ({ i, ...m }))} margin={{ top: 10, right: 24, bottom: 30, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis
                    dataKey="model"
                    stroke="#6b7280"
                    fontSize={11}
                    tick={{ fontSize: 10 }}
                    angle={-15}
                    textAnchor="end"
                    interval={0}
                    height={50}
                  />
                  <YAxis
                    domain={[0, 1]}
                    stroke="#6b7280"
                    fontSize={12}
                    tickFormatter={(v) => `${Math.round(v * 100)}%`}
                    label={{ value: "IKP accuracy", angle: -90, position: "insideLeft", offset: 14, fill: "#6b7280", fontSize: 13 }}
                  />
                  <Tooltip content={<ModelTip />} />
                  <Line
                    type="monotone"
                    dataKey="accuracy"
                    stroke="#1d4ed8"
                    strokeWidth={2.5}
                    dot={{ r: 5 }}
                    name="Penalized"
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="raw_accuracy"
                    stroke="#9ca3af"
                    strokeWidth={1.5}
                    strokeDasharray="4 2"
                    dot={false}
                    name="Raw"
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <TierStack chain={active.chain} />
          </div>
        )}
      </section>

      <section className="bg-white border border-ink/10 rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-ink/10">
          <h2 className="text-lg font-semibold">GPT-5 family size stratification</h2>
          <p className="text-xs text-ink/50 mt-1">
            T5 accuracy (rare facts) is the cleanest size signal: nano {"<"} mini {"<"} base ≈ pro, revealing 20×
            knowledge gap across the size lineup.
          </p>
        </div>
        <GPT5Family data={data.gpt5_family} />
      </section>
    </div>
  );
}

function ModelTip({ active, payload }: any) {
  if (!active || !payload || !payload.length) return null;
  const p = payload[0].payload;
  return (
    <div className="bg-white border border-ink/15 shadow-lg rounded-lg p-3 text-xs max-w-xs">
      <div className="font-semibold text-sm">{p.model}</div>
      <div className="grid grid-cols-2 gap-x-3 mt-1.5 tabular-nums">
        <div className="text-ink/50">Penalized</div><div className="text-right">{(p.accuracy * 100).toFixed(1)}%</div>
        <div className="text-ink/50">Raw</div><div className="text-right">{(p.raw_accuracy * 100).toFixed(1)}%</div>
      </div>
    </div>
  );
}

function TierStack({ chain }: { chain: any[] }) {
  const TIERS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"];
  return (
    <div className="mt-6">
      <h4 className="text-xs uppercase tracking-wider text-ink/60 mb-2">Per-tier breakdown</h4>
      <table className="w-full text-xs tabular-nums">
        <thead>
          <tr>
            <th className="px-2 py-1 text-left text-ink/60">Model</th>
            {TIERS.map((t) => (
              <th key={t} className="px-2 py-1 text-right" style={{ color: TIER_COLOR[t] }}>
                {t}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {chain.map((m) => (
            <tr key={m.model} className="border-t border-ink/5">
              <td className="px-2 py-1.5">
                <Link to={`/models/${m.model}`} className="text-ink hover:text-accent">{m.model}</Link>
              </td>
              {TIERS.map((t) => {
                const acc = m.tier_accuracy?.[t] ?? 0;
                return (
                  <td
                    key={t}
                    className="px-2 py-1.5 text-right"
                    style={{
                      background: `rgba(37,99,235,${Math.max(0.05, acc)})`,
                      color: acc > 0.55 ? "white" : "rgba(11,13,18,0.85)",
                    }}
                  >
                    {(acc * 100).toFixed(0)}
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

function GPT5Family({ data }: { data: any[] }) {
  const groups: Record<string, any[]> = {};
  for (const m of data) {
    const k = m.variant;
    (groups[k] ??= []).push(m);
  }
  const order = ["nano", "mini", "base", "pro", "think"];
  return (
    <div className="p-6 space-y-5">
      {order.map((variant) => {
        const rows = groups[variant];
        if (!rows?.length) return null;
        return (
          <div key={variant}>
            <div className="flex items-baseline justify-between mb-2">
              <h3 className="text-sm font-semibold capitalize" style={{ color: VARIANT_COLOR[variant] }}>
                {variant} ({rows.length})
              </h3>
              <div className="text-xs text-ink/50">
                T5 range {(Math.min(...rows.map((r) => r.tier_accuracy.T5 ?? 0)) * 100).toFixed(0)}–
                {(Math.max(...rows.map((r) => r.tier_accuracy.T5 ?? 0)) * 100).toFixed(0)}%
              </div>
            </div>
            <table className="w-full text-sm tabular-nums">
              <thead>
                <tr className="text-xs text-ink/60 uppercase tracking-wide">
                  <th className="text-left px-2 py-1">Model</th>
                  <th className="text-right px-2 py-1">Accuracy</th>
                  <th className="text-right px-2 py-1">T5</th>
                  <th className="text-right px-2 py-1">T6</th>
                  <th className="px-2 py-1">Visual</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => {
                  const acc = r.accuracy * 100;
                  return (
                    <tr key={r.model} className="border-t border-ink/5">
                      <td className="px-2 py-1.5">
                        <Link to={`/models/${r.model}`} className="text-ink hover:text-accent">{r.model}</Link>
                      </td>
                      <td className="px-2 py-1.5 text-right font-medium">{acc.toFixed(1)}%</td>
                      <td className="px-2 py-1.5 text-right">{((r.tier_accuracy.T5 ?? 0) * 100).toFixed(0)}</td>
                      <td className="px-2 py-1.5 text-right">{((r.tier_accuracy.T6 ?? 0) * 100).toFixed(0)}</td>
                      <td className="px-2 py-1.5">
                        <div className="h-2 bg-ink/[0.06] rounded w-full max-w-[280px] relative">
                          <div
                            className="absolute inset-y-0 left-0 rounded"
                            style={{ width: `${acc}%`, background: VARIANT_COLOR[variant] + "cc" }}
                          />
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        );
      })}
    </div>
  );
}
