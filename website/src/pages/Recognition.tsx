import { useMemo, useState } from "react";
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { useRecognition } from "../data";
import { TIER_COLOR } from "../util";
import Loading from "../components/Loading";

export default function Recognition() {
  const { data, loading } = useRecognition();
  const [tierFilter, setTierFilter] = useState<string | null>(null);

  const filtered = useMemo(() => {
    if (!data) return [];
    const pts = tierFilter ? data.points.filter((p) => p.tier === tierFilter) : data.points;
    return pts
      .filter((p) => p.cited_by_count && p.cited_by_count > 0)
      .map((p) => ({
        x: Math.log10(p.cited_by_count!),
        y: p.recognition_rate,
        ...p,
      }));
  }, [data, tierFilter]);

  if (loading || !data) return <Loading what="recognition data" />;

  const tiers = ["T3", "T4", "T5", "T6", "T7"];

  return (
    <div className="space-y-10">
      <header>
        <h1 className="text-4xl font-bold tracking-tight">What makes a researcher recognized?</h1>
        <p className="mt-3 text-ink/70 max-w-3xl">
          For the {data.n} researcher probes with OpenAlex citations, recognition rate (how often models get the
          researcher's subfield right) correlates only moderately with citations (Spearman ρ = {data.spearman_log_citations.toFixed(3)}).
          Name uniqueness, practitioner-ecosystem amplification, and named-artifact visibility fill most of the
          residual.
        </p>
      </header>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KV label="Researchers" value={String(data.n)} />
        <KV label="Spearman ρ" value={data.spearman_log_citations.toFixed(3)} sub="log(citations) vs recognition" />
        <KV label="Pearson r" value={data.pearson_log_citations.toFixed(3)} sub="log(citations)" />
        <KV
          label="Median recog. (top quintile)"
          value={
            data.quintile_buckets.length
              ? (data.quintile_buckets[data.quintile_buckets.length - 1].median_recognition * 100).toFixed(0) + "%"
              : "—"
          }
        />
      </div>

      <section className="bg-white border border-ink/10 rounded-lg p-6">
        <div className="flex flex-wrap gap-3 justify-between items-baseline mb-3">
          <h2 className="text-lg font-semibold">Recognition rate vs log(citations)</h2>
          <div className="flex gap-1.5">
            <FilterChip active={tierFilter === null} onClick={() => setTierFilter(null)} label="All" />
            {tiers.map((t) => (
              <FilterChip
                key={t}
                active={tierFilter === t}
                onClick={() => setTierFilter(t === tierFilter ? null : t)}
                label={t}
                color={TIER_COLOR[t]}
              />
            ))}
          </div>
        </div>
        <div className="h-[500px] -ml-5">
          <ResponsiveContainer>
            <ScatterChart margin={{ top: 10, right: 24, bottom: 40, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                type="number"
                dataKey="x"
                stroke="#6b7280"
                fontSize={12}
                tickFormatter={(v) => {
                  const c = Math.pow(10, v);
                  if (c >= 1e6) return (c / 1e6).toFixed(0) + "M";
                  if (c >= 1e3) return (c / 1e3).toFixed(0) + "k";
                  return c.toFixed(0);
                }}
                label={{ value: "OpenAlex citations (log)", position: "insideBottom", offset: -16, fill: "#6b7280", fontSize: 13 }}
              />
              <YAxis
                type="number"
                dataKey="y"
                domain={[0, 1]}
                stroke="#6b7280"
                fontSize={12}
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
                label={{ value: "Recognition rate", angle: -90, position: "insideLeft", offset: 14, fill: "#6b7280", fontSize: 13 }}
              />
              <ZAxis range={[60, 60]} />
              <Tooltip content={<Tip />} />
              <Scatter
                data={filtered}
                shape={(props: any) => {
                  const { cx, cy, payload } = props;
                  return (
                    <circle
                      cx={cx}
                      cy={cy}
                      r={4.5}
                      fill={TIER_COLOR[payload.tier]}
                      fillOpacity={0.75}
                      stroke={TIER_COLOR[payload.tier]}
                      strokeWidth={1}
                    />
                  );
                }}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="bg-white border border-ink/10 rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-ink/10">
          <h2 className="text-lg font-semibold">Citation quintile buckets</h2>
          <p className="text-xs text-ink/50 mt-1">
            Monotonic curve, but big within-bucket variance: even at {">"}5000 citations, mean recognition barely
            passes 50%.
          </p>
        </div>
        <table className="w-full text-sm">
          <thead className="bg-ink/5 text-left text-ink/60 uppercase text-xs tracking-wide">
            <tr>
              <th className="px-4 py-2">Quintile</th>
              <th className="px-4 py-2 text-right">n</th>
              <th className="px-4 py-2 text-right">Median citations</th>
              <th className="px-4 py-2 text-right">Citation range</th>
              <th className="px-4 py-2 text-right">Median recognition</th>
              <th className="px-4 py-2">Visual</th>
            </tr>
          </thead>
          <tbody className="tabular-nums">
            {data.quintile_buckets.map((b, i) => {
              const pct = b.median_recognition * 100;
              return (
                <tr key={b.index} className="border-t border-ink/5">
                  <td className="px-4 py-2.5">Q{i + 1}</td>
                  <td className="px-4 py-2.5 text-right">{b.n}</td>
                  <td className="px-4 py-2.5 text-right">{b.median_citations.toLocaleString()}</td>
                  <td className="px-4 py-2.5 text-right text-ink/60">
                    {b.citations_range[0].toLocaleString()}–{b.citations_range[1].toLocaleString()}
                  </td>
                  <td className="px-4 py-2.5 text-right font-medium">{pct.toFixed(0)}%</td>
                  <td className="px-4 py-2.5">
                    <div className="h-2.5 bg-ink/[0.06] rounded w-[220px] relative">
                      <div className="absolute inset-y-0 left-0 bg-emerald-500/80 rounded" style={{ width: `${pct}%` }} />
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      <section className="bg-white border border-ink/10 rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-ink/10">
          <h2 className="text-lg font-semibold">All researchers ({data.n})</h2>
          <p className="text-xs text-ink/50 mt-1">
            Sortable by recognition or citations. Outliers — high citations but low recognition (or vice versa)
            — are usually name-collisions or artifact-driven visibility.
          </p>
        </div>
        <ResearcherTable rows={data.points} />
      </section>

      <div className="bg-accent/5 border border-accent/20 rounded-lg p-5 text-sm text-ink/80 leading-relaxed">
        <strong className="text-accent">Three residual factors.</strong>{" "}
        (1) <em>Named artifacts</em>: a tool with broad adoption (FlashAttention, SVMlight, Catapult/ClickNP) can
        outweigh an order of magnitude of citations.
        (2) <em>Name uniqueness</em>: common East Asian surnames yield ~22% recognition vs ~45% for uniquely spelled names,
        at matched citation counts.
        (3) <em>Practitioner ecosystem</em>: in ML/AI, no researcher in our sample falls below 43%;
        in Systems subfields without a named tool, the same citation range can reach 0%.
      </div>
    </div>
  );
}

function ResearcherTable({ rows }: { rows: any[] }) {
  const [sortKey, setSortKey] = useState<"recognition_rate" | "cited_by_count" | "h_index">("recognition_rate");
  const [dir, setDir] = useState<"asc" | "desc">("desc");

  const sorted = useMemo(() => {
    const s = [...rows].sort((a, b) => {
      const va = a[sortKey] ?? -1;
      const vb = b[sortKey] ?? -1;
      return dir === "desc" ? vb - va : va - vb;
    });
    return s;
  }, [rows, sortKey, dir]);

  function head(k: typeof sortKey, label: string) {
    return (
      <th
        className="px-4 py-2 text-right cursor-pointer hover:text-ink"
        onClick={() => (sortKey === k ? setDir(dir === "desc" ? "asc" : "desc") : (setSortKey(k), setDir("desc")))}
      >
        {label}{sortKey === k && (dir === "desc" ? " ↓" : " ↑")}
      </th>
    );
  }

  return (
    <div className="overflow-x-auto max-h-[640px]">
      <table className="w-full text-sm">
        <thead className="bg-ink/5 text-left text-ink/60 uppercase text-xs tracking-wide sticky top-0">
          <tr>
            <th className="px-4 py-2">Researcher</th>
            <th className="px-4 py-2">Tier</th>
            <th className="px-4 py-2">Field</th>
            {head("recognition_rate", "Recognition")}
            {head("cited_by_count", "Citations")}
            {head("h_index", "h-index")}
          </tr>
        </thead>
        <tbody className="tabular-nums">
          {sorted.slice(0, 200).map((p) => (
            <tr key={p.probe_id} className="border-t border-ink/5 hover:bg-ink/5">
              <td className="px-4 py-2">{p.name}</td>
              <td className="px-4 py-2">
                <span
                  className="text-xs px-1.5 py-0.5 rounded"
                  style={{ background: TIER_COLOR[p.tier] + "25", color: TIER_COLOR[p.tier] }}
                >
                  {p.tier}
                </span>
              </td>
              <td className="px-4 py-2 text-ink/60 max-w-[260px] truncate">{p.field ?? "—"}</td>
              <td className="px-4 py-2 text-right font-medium">{(p.recognition_rate * 100).toFixed(0)}%</td>
              <td className="px-4 py-2 text-right text-ink/70">
                {p.cited_by_count?.toLocaleString() ?? "—"}
              </td>
              <td className="px-4 py-2 text-right text-ink/60">{p.h_index ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {sorted.length > 200 && (
        <div className="text-xs text-ink/50 px-4 py-2 border-t border-ink/5">
          Showing first 200 of {sorted.length}. Sort to see others.
        </div>
      )}
    </div>
  );
}

function FilterChip({ active, onClick, label, color }: { active: boolean; onClick: () => void; label: string; color?: string }) {
  return (
    <button
      onClick={onClick}
      className="text-xs px-2.5 py-1 rounded-full border transition"
      style={{
        background: active ? (color ?? "#0b0d12") + "15" : "transparent",
        borderColor: active ? color ?? "#0b0d12" : "rgba(11,13,18,0.2)",
        color: active ? color ?? "#0b0d12" : "rgba(11,13,18,0.6)",
      }}
    >
      {label}
    </button>
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

function Tip({ active, payload }: any) {
  if (!active || !payload || !payload.length) return null;
  const p = payload[0].payload;
  return (
    <div className="bg-white border border-ink/15 shadow-lg rounded-lg p-3 text-xs max-w-xs">
      <div className="font-semibold text-sm">{p.name}</div>
      <div className="text-ink/60">{p.field ?? "—"} · {p.tier}</div>
      <div className="grid grid-cols-2 gap-x-3 mt-1.5 tabular-nums">
        <div className="text-ink/50">Recognition</div><div className="text-right font-medium">{(p.recognition_rate * 100).toFixed(0)}%</div>
        <div className="text-ink/50">Citations</div><div className="text-right">{p.cited_by_count?.toLocaleString() ?? "—"}</div>
        <div className="text-ink/50">h-index</div><div className="text-right">{p.h_index ?? "—"}</div>
      </div>
    </div>
  );
}
