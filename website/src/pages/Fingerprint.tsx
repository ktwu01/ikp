import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { useFingerprint } from "../data";
import Loading from "../components/Loading";
import type { FingerprintPair } from "../types";

const REGIME_COLOR: Record<string, string> = {
  "shared-base": "#7c3aed",
  lineage: "#059669",
  retrained: "#dc2626",
  independent: "#6b7280",
  "independent (small $n$)": "#9ca3af",
  "independent (small n)": "#9ca3af",
};

function regimeFor(hss: number, jaccard: number, bothW: number): string {
  if (bothW < 5) return "independent (small n)";
  if (hss >= 0.3 && jaccard >= 0.6) return "shared-base";
  if (hss >= 0.1 && jaccard >= 0.5) return "lineage";
  if (bothW >= 10) return "retrained";
  return "independent";
}

function normalizeClass(c: string | undefined): string {
  if (!c) return "independent";
  return c.replace("$n$", "n");
}

function colorFor(cls: string): string {
  return REGIME_COLOR[cls] ?? "#6b7280";
}

export default function Fingerprint() {
  const { data, loading } = useFingerprint();
  const [selectedFamily, setSelectedFamily] = useState<string | null>(null);

  const scatterData = useMemo(() => {
    if (!data) return [];
    return data.consecutive_pairs
      .filter((p) => p.both_wrong >= 5)
      .map((p) => ({ ...p, cls: normalizeClass(p.class) }));
  }, [data]);

  if (loading || !data) return <Loading what="fingerprint data" />;

  const selectedPairs = selectedFamily
    ? data.families.find((f) => f.family === selectedFamily)?.pairs ?? []
    : [];

  // Bucket cross-vendor outliers by HSS bucket
  const crossSorted = [...data.cross_vendor].sort((a, b) => b.hss - a.hss);

  return (
    <div className="space-y-10">
      <header>
        <h1 className="text-4xl font-bold tracking-tight">Knowledge fingerprinting</h1>
        <p className="mt-3 text-ink/70 max-w-3xl">
          The set of rare facts a model gets right — and the specific wrong answers it gives when it's
          wrong — is a training-free fingerprint. Combining Jaccard similarity, lift, and
          <em> hallucination similarity</em> (HSS) separates weight-sharing siblings from full retrains,
          even across closed-vendor version families.
        </p>
      </header>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KV label="Probes" value={`${data.n_probes}`} sub="T5–T6 rare facts" />
        <KV label="Models compared" value={`${data.n_models}`} />
        <KV label="Families" value={`${data.families.length}`} />
        <KV label="Cross-vendor outliers" value={`${data.cross_vendor.length}`} sub="HSS ≥ 0.20, ≥10 joint-wrong" />
      </div>

      <section className="bg-white border border-ink/10 rounded-lg p-6">
        <h2 className="text-lg font-semibold">Three regimes in the (J, HSS) plane</h2>
        <p className="text-sm text-ink/60 mt-1">
          Each dot is a consecutive-generation pair within a family (≥5 joint-wrong probes). Shared-base pairs
          sit high on HSS and J; retrained pairs collapse to the lower-left even when Jaccard is still large.
        </p>
        <div className="h-[500px] -ml-5 mt-4">
          <ResponsiveContainer>
            <ScatterChart margin={{ top: 10, right: 24, bottom: 40, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                type="number"
                dataKey="jaccard"
                domain={[0, 1]}
                stroke="#6b7280"
                fontSize={12}
                tickFormatter={(v) => v.toFixed(1)}
                label={{ value: "Jaccard similarity (correct-answer sets)", position: "insideBottom", offset: -16, fill: "#6b7280", fontSize: 13 }}
              />
              <YAxis
                type="number"
                dataKey="hss"
                domain={[0, 1]}
                stroke="#6b7280"
                fontSize={12}
                tickFormatter={(v) => v.toFixed(1)}
                label={{ value: "HSS (hallucination similarity)", angle: -90, position: "insideLeft", offset: 14, fill: "#6b7280", fontSize: 13 }}
              />
              <ZAxis range={[60, 60]} />
              <ReferenceLine y={0.3} stroke="#7c3aed" strokeDasharray="4 2" label={{ value: "shared-base 0.30", position: "insideTopRight", fill: "#7c3aed", fontSize: 11 }} />
              <ReferenceLine y={0.1} stroke="#059669" strokeDasharray="2 2" label={{ value: "lineage 0.10", position: "insideTopRight", fill: "#059669", fontSize: 11 }} />
              <Tooltip content={<PairTooltip />} />
              <Scatter
                data={scatterData}
                shape={(props: any) => {
                  const { cx, cy, payload } = props;
                  return (
                    <circle
                      cx={cx}
                      cy={cy}
                      r={5}
                      fill={colorFor(payload.cls)}
                      fillOpacity={0.75}
                      stroke={colorFor(payload.cls)}
                      strokeWidth={1}
                    />
                  );
                }}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-3 flex flex-wrap gap-4 text-xs text-ink/70">
          {["shared-base", "lineage", "retrained", "independent"].map((c) => (
            <span key={c} className="flex items-center gap-1.5">
              <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: colorFor(c) }} />
              {c}
            </span>
          ))}
        </div>
      </section>

      <section className="bg-white border border-ink/10 rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-ink/10 flex flex-wrap gap-3 items-baseline justify-between">
          <div>
            <h2 className="text-lg font-semibold">Within-family consecutive pairs</h2>
            <p className="text-xs text-ink/50 mt-1">
              Each vendor's family shows a pattern: some prefer retrains, others post-train. Click a family to see every pair.
            </p>
          </div>
          {selectedFamily && (
            <button onClick={() => setSelectedFamily(null)} className="text-xs text-accent hover:underline">
              Clear selection
            </button>
          )}
        </div>
        <div className="flex flex-wrap gap-1.5 p-4 border-b border-ink/5">
          {data.families.map((f) => (
            <button
              key={f.family}
              onClick={() => setSelectedFamily(f.family === selectedFamily ? null : f.family)}
              className={`text-xs px-2.5 py-1 rounded-full border transition ${
                selectedFamily === f.family
                  ? "bg-ink text-paper border-ink"
                  : "border-ink/20 text-ink/70 hover:bg-ink/5"
              }`}
            >
              {f.family}
            </button>
          ))}
        </div>
        {selectedFamily ? (
          <PairTable pairs={selectedPairs} />
        ) : (
          <PairTable pairs={data.consecutive_pairs} />
        )}
      </section>

      <section className="bg-white border border-ink/10 rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-ink/10">
          <h2 className="text-lg font-semibold">Cross-vendor outliers</h2>
          <p className="text-xs text-ink/50 mt-1">
            {data.cross_vendor.length} pairs across different vendors with HSS ≥ 0.20 and ≥10 joint-wrong probes
            — candidate distillation signals.
          </p>
        </div>
        <div className="overflow-x-auto max-h-[500px]">
          <table className="w-full text-sm">
            <thead className="bg-ink/5 text-left text-ink/60 uppercase text-xs tracking-wide sticky top-0">
              <tr>
                <th className="px-4 py-2">Model A</th>
                <th className="px-4 py-2">Model B</th>
                <th className="px-4 py-2 text-right">J</th>
                <th className="px-4 py-2 text-right">lift</th>
                <th className="px-4 py-2 text-right">HSS</th>
                <th className="px-4 py-2 text-right">both wrong</th>
                <th className="px-4 py-2 text-right text-ink/40">probes →</th>
              </tr>
            </thead>
            <tbody className="tabular-nums">
              {crossSorted.map((p, i) => {
                const pid = pairIdFor(p);
                return (
                  <tr key={`${p.a}|${p.b}|${i}`} className="border-t border-ink/5 hover:bg-ink/5">
                    <td className="px-4 py-2">
                      <Link to={`/models/${p.a}`} className="text-ink hover:text-accent">{p.a}</Link>
                    </td>
                    <td className="px-4 py-2">
                      <Link to={`/models/${p.b}`} className="text-ink hover:text-accent">{p.b}</Link>
                    </td>
                    <td className="px-4 py-2 text-right">{p.jaccard.toFixed(3)}</td>
                    <td className="px-4 py-2 text-right">{p.lift !== undefined ? p.lift.toFixed(2) : "—"}</td>
                    <td className={`px-4 py-2 text-right font-medium ${p.hss >= 0.3 ? "text-violet-700" : "text-ink"}`}>
                      {p.hss.toFixed(3)}
                    </td>
                    <td className="px-4 py-2 text-right text-ink/60">{p.both_wrong}</td>
                    <td className="px-4 py-2 text-right">
                      <Link
                        to={`/fingerprint/${pid}`}
                        className="text-xs text-accent hover:underline"
                        title="See the actual joint-wrong probes and both models' responses"
                      >
                        view&nbsp;responses →
                      </Link>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="bg-white border border-ink/10 rounded-lg p-6">
        <h2 className="text-lg font-semibold">Jaccard heatmap — 15 frontier models</h2>
        <p className="text-sm text-ink/60 mt-1">
          Within-vendor similarity (diagonal blocks) is consistently higher than cross-vendor. Jaccard alone
          doesn't separate shared-base from independent-but-competitive; see the scatter above for the sharper HSS view.
        </p>
        <div className="mt-5 overflow-x-auto">
          <HeatmapTable models={data.heatmap.models} matrix={data.heatmap.matrix} />
        </div>
      </section>

      <div className="bg-accent/5 border border-accent/20 rounded-lg p-5 text-sm text-ink/80 leading-relaxed">
        <strong className="text-accent">How to read regimes:</strong> <em>shared-base</em> means the two models
        likely share weights (e.g. a "-pro" or "-think" variant of the same checkpoint). <em>Lineage</em> is
        consistent with post-training / continued pretraining on a common ancestor. <em>Retrained</em> means
        the pair behaves statistically like independent trainings even when they share a family label —
        implying a full retrain, not a point release.
      </div>
    </div>
  );
}

function PairTable({ pairs }: { pairs: FingerprintPair[] }) {
  const sorted = [...pairs].sort((a, b) => b.hss - a.hss);
  const hasFamily = !!pairs[0]?.family;
  const hasLift = pairs.some((p) => p.lift !== undefined && p.lift !== null);
  const fmt = (v: number | undefined | null, digits: number) =>
    v === undefined || v === null || Number.isNaN(v) ? "—" : v.toFixed(digits);
  return (
    <div className="overflow-x-auto max-h-[520px]">
      <table className="w-full text-sm">
        <thead className="bg-ink/5 text-left text-ink/60 uppercase text-xs tracking-wide sticky top-0">
          <tr>
            {hasFamily && <th className="px-4 py-2">Family</th>}
            <th className="px-4 py-2">From</th>
            <th className="px-4 py-2">To</th>
            <th className="px-4 py-2 text-right">J</th>
            {hasLift && <th className="px-4 py-2 text-right">lift</th>}
            <th className="px-4 py-2 text-right">HSS</th>
            <th className="px-4 py-2 text-right">both wrong</th>
            <th className="px-4 py-2">regime</th>
            <th className="px-4 py-2 text-right text-ink/40">probes →</th>
          </tr>
        </thead>
        <tbody className="tabular-nums">
          {sorted.map((p, i) => {
            const cls = normalizeClass(p.class);
            const pid = pairIdFor(p);
            return (
              <tr key={`${p.a}|${p.b}|${i}`} className="border-t border-ink/5 hover:bg-ink/5">
                {hasFamily && <td className="px-4 py-2 text-ink/60">{p.family ?? ""}</td>}
                <td className="px-4 py-2">
                  <Link to={`/models/${p.a}`} className="text-ink hover:text-accent">{p.a}</Link>
                </td>
                <td className="px-4 py-2">
                  <Link to={`/models/${p.b}`} className="text-ink hover:text-accent">{p.b}</Link>
                </td>
                <td className="px-4 py-2 text-right">{fmt(p.jaccard, 3)}</td>
                {hasLift && <td className="px-4 py-2 text-right">{fmt(p.lift, 2)}</td>}
                <td className="px-4 py-2 text-right font-medium">{fmt(p.hss, 3)}</td>
                <td className="px-4 py-2 text-right text-ink/60">{p.both_wrong}</td>
                <td className="px-4 py-2">
                  <span
                    className="text-xs px-2 py-0.5 rounded-full"
                    style={{ background: colorFor(cls) + "20", color: colorFor(cls) }}
                  >
                    {cls}
                  </span>
                </td>
                <td className="px-4 py-2 text-right">
                  <Link
                    to={`/fingerprint/${pid}`}
                    className="text-xs text-accent hover:underline"
                    title="See the actual joint-wrong probes and both models' responses"
                  >
                    view&nbsp;responses →
                  </Link>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function pairIdFor(p: FingerprintPair): string {
  if (p.pair_id) return p.pair_id;
  const [a, b] = [p.a, p.b].sort();
  return `${a}__${b}`;
}

function HeatmapTable({ models, matrix }: { models: string[]; matrix: any[][] }) {
  return (
    <table className="text-xs border-collapse">
      <thead>
        <tr>
          <th className="p-1"></th>
          {models.map((m) => (
            <th key={m} className="p-1 text-left whitespace-nowrap">
              <div className="rotate-[-35deg] origin-left w-0 h-10 overflow-visible relative">
                <span className="absolute left-0 whitespace-nowrap text-ink/70">{m}</span>
              </div>
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {models.map((ra, i) => (
          <tr key={ra}>
            <td className="p-1 text-right text-ink/70 whitespace-nowrap pr-3">{ra}</td>
            {models.map((rb, j) => {
              const cell = matrix[i][j];
              const j_val = cell?.j ?? null;
              const hss = cell?.hss ?? null;
              const bg = j_val == null ? "#f3f4f6" : j_val >= 1
                ? "#1e3a8a"
                : `rgba(30,58,138,${(j_val ?? 0).toFixed(3)})`;
              const title = j_val == null
                ? `${ra} vs ${rb} — no data`
                : `J=${j_val.toFixed(3)} · HSS=${hss?.toFixed(3) ?? "—"} · both_w=${cell?.both_w ?? "—"}`;
              return (
                <td
                  key={rb}
                  title={title}
                  className="p-1"
                  style={{ minWidth: 36, width: 36, height: 24 }}
                >
                  <div
                    className="rounded-sm"
                    style={{
                      background: bg,
                      color: (j_val ?? 0) > 0.5 ? "white" : "rgba(11,13,18,0.7)",
                      height: 22,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 10,
                      fontVariantNumeric: "tabular-nums",
                    }}
                  >
                    {j_val != null ? j_val.toFixed(2).replace("0.", ".") : "—"}
                  </div>
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function PairTooltip({ active, payload }: any) {
  if (!active || !payload || !payload.length) return null;
  const p = payload[0].payload;
  return (
    <div className="bg-white border border-ink/15 shadow-lg rounded-lg p-3 text-xs space-y-0.5 max-w-xs">
      <div className="font-semibold text-ink text-sm">{p.a} → {p.b}</div>
      {p.family && <div className="text-ink/60">{p.family}</div>}
      <div className="grid grid-cols-2 gap-x-3 mt-1.5 tabular-nums">
        <div className="text-ink/50">Jaccard</div><div className="text-right">{p.jaccard.toFixed(3)}</div>
        <div className="text-ink/50">Lift</div><div className="text-right">{p.lift.toFixed(2)}</div>
        <div className="text-ink/50">HSS</div><div className="text-right font-medium">{p.hss.toFixed(3)}</div>
        <div className="text-ink/50">Both wrong</div><div className="text-right">{p.both_wrong}</div>
        <div className="text-ink/50">Same wrong</div><div className="text-right">{p.same_wrong}</div>
        <div className="text-ink/50">Regime</div>
        <div className="text-right" style={{ color: colorFor(p.cls) }}>{p.cls}</div>
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
