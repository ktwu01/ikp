import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useModelDetail } from "../data";
import { formatPercent, formatParams, TIER_COLOR, VERDICT_COLOR } from "../util";
import Loading from "../components/Loading";
import { VendorTag, VerdictTag } from "../components/Tag";
import type { Tier } from "../types";

const TIERS: Tier[] = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"];
const VERDICTS = ["CORRECT", "WRONG", "REFUSAL"] as const;
type Verdict = (typeof VERDICTS)[number];

export default function ModelDetail() {
  const { name } = useParams<{ name: string }>();
  const { data: m, loading, error } = useModelDetail(name || "");
  const [tier, setTier] = useState<Tier>("T4");
  const [verdict, setVerdict] = useState<Verdict>("CORRECT");

  if (loading || !m) {
    if (error) {
      return (
        <div className="text-center py-20">
          <p className="text-ink/60">Model not found: <code>{name}</code></p>
          <Link to="/models" className="text-accent underline">Back to models</Link>
        </div>
      );
    }
    return <Loading what={`model ${name}`} />;
  }

  const chartData = TIERS.map((t) => ({
    tier: t,
    accuracy: (m.tier_accuracy[t] ?? 0) * 100,
    fill: TIER_COLOR[t],
  }));

  const samples = m.samples[tier]?.[verdict] || [];

  return (
    <div className="space-y-8">
      <div>
        <Link to="/models" className="text-sm text-ink/50 hover:text-accent">← All models</Link>
        <h1 className="text-3xl font-bold tracking-tight mt-2">{m.model}</h1>
        <div className="flex flex-wrap items-center gap-3 mt-2 text-sm text-ink/60">
          <VendorTag vendor={m.vendor} />
          {m.family && <span>family: {m.family}</span>}
          {m.arch && <span>arch: {m.arch}</span>}
          {m.thinking && <span className="text-violet-600">thinking</span>}
          {m.type && <span>· {m.type}</span>}
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KV label="Penalized accuracy" value={formatPercent(m.accuracy)} />
        <KV label="Raw accuracy" value={formatPercent(m.raw_accuracy)} />
        <KV label="Total params" value={formatParams(m.params_B)} sub={m.arch === "moe" && m.active_B ? `${formatParams(m.active_B)} active` : undefined} />
        <KV label="Refusal rate" value={formatPercent(refusalRate(m.tier_stats))} />
      </div>

      <div className="bg-white border border-ink/10 rounded-lg p-6">
        <h2 className="text-lg font-semibold mb-3">Per-tier accuracy</h2>
        <div className="h-[260px] -ml-4">
          <ResponsiveContainer>
            <BarChart data={chartData} margin={{ top: 10, right: 16, bottom: 12, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="tier" stroke="#6b7280" fontSize={12} />
              <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} stroke="#6b7280" fontSize={12} />
              <Tooltip
                cursor={{ fill: "#f3f4f6" }}
                formatter={(v: number) => [`${v.toFixed(1)}%`, "accuracy"]}
              />
              <Bar dataKey="accuracy" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="overflow-x-auto mt-4">
          <table className="w-full text-xs tabular-nums">
            <thead className="text-ink/50 text-left">
              <tr>
                <th className="py-1 pr-4">Tier</th>
                {TIERS.map((t) => (
                  <th key={t} className="py-1 px-2 text-right" style={{ color: TIER_COLOR[t] }}>{t}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr className="border-t border-ink/5">
                <td className="py-1 pr-4 text-ink/60">Correct</td>
                {TIERS.map((t) => <td key={t} className="py-1 px-2 text-right">{m.tier_stats[t]?.correct ?? 0}</td>)}
              </tr>
              <tr className="border-t border-ink/5">
                <td className="py-1 pr-4 text-ink/60">Wrong</td>
                {TIERS.map((t) => <td key={t} className="py-1 px-2 text-right">{m.tier_stats[t]?.wrong ?? 0}</td>)}
              </tr>
              <tr className="border-t border-ink/5">
                <td className="py-1 pr-4 text-ink/60">Refusal</td>
                {TIERS.map((t) => <td key={t} className="py-1 px-2 text-right">{m.tier_stats[t]?.refusal ?? 0}</td>)}
              </tr>
              <tr className="border-t border-ink/5 font-medium">
                <td className="py-1 pr-4">Accuracy</td>
                {TIERS.map((t) => <td key={t} className="py-1 px-2 text-right">{formatPercent(m.tier_accuracy[t] ?? 0, 0)}</td>)}
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="bg-white border border-ink/10 rounded-lg p-6">
        <div className="flex items-center justify-between flex-wrap gap-3 mb-4">
          <h2 className="text-lg font-semibold">Sample responses</h2>
          <div className="flex gap-2">
            <div className="flex gap-1">
              {TIERS.map((t) => (
                <button
                  key={t}
                  onClick={() => setTier(t)}
                  className={`text-xs px-2 py-1 rounded border ${
                    tier === t ? "border-current font-medium" : "border-ink/15 text-ink/50 hover:bg-ink/5"
                  }`}
                  style={{ color: tier === t ? TIER_COLOR[t] : undefined }}
                >
                  {t}
                </button>
              ))}
            </div>
            <div className="flex gap-1">
              {VERDICTS.map((v) => (
                <button
                  key={v}
                  onClick={() => setVerdict(v)}
                  className={`text-xs px-2 py-1 rounded border ${
                    verdict === v ? "border-current font-medium" : "border-ink/15 text-ink/50 hover:bg-ink/5"
                  }`}
                  style={{ color: verdict === v ? VERDICT_COLOR[v] : undefined }}
                >
                  {v}
                </button>
              ))}
            </div>
          </div>
        </div>

        {samples.length === 0 ? (
          <div className="text-sm text-ink/50 py-8 text-center">No {verdict} samples cached for {tier}.</div>
        ) : (
          <div className="space-y-3">
            {samples.map((s) => (
              <div key={s.probe_id} className="border border-ink/10 rounded-md p-4 text-sm">
                <div className="flex items-start justify-between gap-3 mb-2">
                  <div className="font-medium">{s.question}</div>
                  <Link to={`/probes/${s.probe_id}`} className="text-xs text-ink/40 hover:text-accent shrink-0">
                    {s.probe_id} →
                  </Link>
                </div>
                <div className="text-xs text-ink/50 mb-2">
                  Gold: <span className="text-ink/80 font-mono">{s.gold_answer}</span>
                </div>
                <div className="bg-ink/[0.03] rounded p-3 text-ink/80 text-sm whitespace-pre-wrap">
                  {s.model_response}
                </div>
                <div className="mt-2"><VerdictTag verdict={verdict} /></div>
              </div>
            ))}
          </div>
        )}
        <p className="text-xs text-ink/40 mt-3">
          Up to 5 samples per (tier, verdict) are bundled. For all 1,400 responses see individual probes.
        </p>
      </div>
    </div>
  );
}

function refusalRate(stats: Record<Tier, { refusal: number; total: number }>): number {
  let r = 0, t = 0;
  for (const k of TIERS) {
    r += stats[k]?.refusal ?? 0;
    t += stats[k]?.total ?? 0;
  }
  return t === 0 ? 0 : r / t;
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
