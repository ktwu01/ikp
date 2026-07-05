import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useProbes } from "../data";
import Loading from "../components/Loading";
import { Tag } from "../components/Tag";
import { TIER_COLOR } from "../util";
import type { Tier } from "../types";

const TIERS: Tier[] = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"];
const PAGE = 60;

type SortKey = "id" | "correct" | "halluc";

export default function Probes() {
  const { data: probes, loading } = useProbes();
  const [query, setQuery] = useState("");
  const [tierFilter, setTierFilter] = useState<Tier | null>(null);
  const [domainFilter, setDomainFilter] = useState<string | null>(null);
  const [sourceFilter, setSourceFilter] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("id");
  const [sortDesc, setSortDesc] = useState(true);
  const [page, setPage] = useState(0);

  const toggleSort = (k: SortKey) => {
    if (sortKey === k) setSortDesc(!sortDesc);
    else { setSortKey(k); setSortDesc(true); }
    setPage(0);
  };

  const domains = useMemo(() => {
    if (!probes) return [];
    return Array.from(new Set(probes.map((p) => p.domain || "—"))).sort();
  }, [probes]);

  const sources = useMemo(() => {
    if (!probes) return [];
    return Array.from(new Set(probes.map((p) => p.source_type || "—"))).sort();
  }, [probes]);

  const filtered = useMemo(() => {
    if (!probes) return [];
    const f = probes.filter((p) => {
      if (tierFilter && p.tier !== tierFilter) return false;
      if (domainFilter && (p.domain || "—") !== domainFilter) return false;
      if (sourceFilter && (p.source_type || "—") !== sourceFilter) return false;
      if (query) {
        const q = query.toLowerCase();
        if (!p.question.toLowerCase().includes(q) && !p.answer.toLowerCase().includes(q) && !p.id.toLowerCase().includes(q))
          return false;
      }
      return true;
    });
    if (sortKey === "id") {
      return sortDesc ? [...f].sort((a, b) => b.id.localeCompare(a.id)) : [...f].sort((a, b) => a.id.localeCompare(b.id));
    }
    const k = sortKey === "correct" ? "correct_rate" : "halluc_rate";
    return [...f].sort((a, b) => {
      const av = (a[k] as number | undefined) ?? -1;
      const bv = (b[k] as number | undefined) ?? -1;
      return sortDesc ? bv - av : av - bv;
    });
  }, [probes, query, tierFilter, domainFilter, sourceFilter, sortKey, sortDesc]);

  if (loading || !probes) return <Loading what="probes" />;

  const tierCounts: Record<Tier, number> = TIERS.reduce((acc, t) => {
    acc[t] = filtered.filter((p) => p.tier === t).length;
    return acc;
  }, {} as Record<Tier, number>);

  const visible = filtered.slice(page * PAGE, (page + 1) * PAGE);
  const totalPages = Math.ceil(filtered.length / PAGE);

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-4xl font-bold tracking-tight">The 1,400 probes</h1>
        <p className="mt-3 text-ink/70 max-w-3xl">
          Every question in the IKP benchmark, organized by tier (T1 = easiest, T7 = hardest).
          Click any probe to see how all 201 models answered it.
        </p>
      </header>

      <div className="bg-white border border-ink/10 rounded-lg p-5 space-y-4">
        <div className="flex flex-wrap gap-3 items-center">
          <input
            type="text"
            placeholder="Search question, answer, or ID…"
            value={query}
            onChange={(e) => { setQuery(e.target.value); setPage(0); }}
            className="flex-1 min-w-[240px] px-3 py-2 border border-ink/15 rounded-md text-sm focus:outline-none focus:border-accent"
          />
          <Pill label="Source" value={sourceFilter} options={sources} onChange={(v) => { setSourceFilter(v); setPage(0); }} />
          <Pill label="Domain" value={domainFilter} options={domains} onChange={(v) => { setDomainFilter(v); setPage(0); }} />
        </div>
        <div className="flex flex-wrap gap-2 items-center">
          <button
            onClick={() => { setTierFilter(null); setPage(0); }}
            className={`text-xs px-3 py-1 rounded-full border ${tierFilter === null ? "bg-ink text-paper border-ink" : "border-ink/15 text-ink/60 hover:bg-ink/5"}`}
          >
            All tiers ({filtered.length})
          </button>
          {TIERS.map((t) => (
            <button
              key={t}
              onClick={() => { setTierFilter(tierFilter === t ? null : t); setPage(0); }}
              className="text-xs px-3 py-1 rounded-full border transition"
              style={{
                background: tierFilter === t ? TIER_COLOR[t] + "22" : "transparent",
                borderColor: tierFilter === t ? TIER_COLOR[t] : "rgba(11,13,18,0.15)",
                color: tierFilter === t ? TIER_COLOR[t] : "rgba(11,13,18,0.6)",
              }}
            >
              {t} ({tierCounts[t]})
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white border border-ink/10 rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-ink/5 text-left text-ink/60 uppercase text-xs tracking-wide">
            <tr>
              <th className="px-4 py-2 w-[110px]">ID</th>
              <th className="px-4 py-2 w-[60px]">Tier</th>
              <th className="px-4 py-2">Question</th>
              <th className="px-4 py-2 w-[180px]">Answer</th>
              <th
                className="px-4 py-2 w-[88px] text-right cursor-pointer select-none hover:text-ink"
                title="Fraction of evaluated models that answered this probe correctly. Click to sort."
                onClick={() => toggleSort("correct")}
              >
                Correct{sortKey === "correct" ? (sortDesc ? " ▼" : " ▲") : ""}
              </th>
              <th
                className="px-4 py-2 w-[88px] text-right cursor-pointer select-none hover:text-ink"
                title="Fraction of models that answered confidently wrong (excluding refusals). Click to sort."
                onClick={() => toggleSort("halluc")}
              >
                Halluc.{sortKey === "halluc" ? (sortDesc ? " ▼" : " ▲") : ""}
              </th>
              <th className="px-4 py-2 w-[100px]">Source</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((p) => (
              <tr key={p.id} className="border-t border-ink/5 hover:bg-ink/5">
                <td className="px-4 py-2.5">
                  <Link to={`/probes/${p.id}`} className="font-mono text-xs text-ink/60 hover:text-accent">
                    {p.id}
                  </Link>
                </td>
                <td className="px-4 py-2.5">
                  <Tag color={TIER_COLOR[p.tier]}>{p.tier}</Tag>
                </td>
                <td className="px-4 py-2.5">
                  <Link to={`/probes/${p.id}`} className="text-ink hover:text-accent">{p.question}</Link>
                  {p.domain && <div className="text-[11px] text-ink/40 mt-0.5">{p.domain}</div>}
                </td>
                <td className="px-4 py-2.5 font-mono text-xs text-ink/70">{p.answer}</td>
                <td className="px-4 py-2.5 text-right tabular-nums">
                  <RateCell value={p.correct_rate} n={p.n_models} tone="emerald" />
                </td>
                <td className="px-4 py-2.5 text-right tabular-nums">
                  <RateCell value={p.halluc_rate} n={p.n_models} tone="rose" />
                </td>
                <td className="px-4 py-2.5 text-xs text-ink/50">{p.source_type || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 text-sm">
          <button
            onClick={() => setPage(Math.max(0, page - 1))}
            disabled={page === 0}
            className="px-3 py-1 border border-ink/15 rounded disabled:opacity-30"
          >
            ← Prev
          </button>
          <span className="text-ink/60">
            Page {page + 1} of {totalPages} · showing {visible.length} of {filtered.length}
          </span>
          <button
            onClick={() => setPage(Math.min(totalPages - 1, page + 1))}
            disabled={page >= totalPages - 1}
            className="px-3 py-1 border border-ink/15 rounded disabled:opacity-30"
          >
            Next →
          </button>
        </div>
      )}
    </div>
  );
}

function RateCell({
  value,
  n,
  tone,
}: {
  value: number | undefined;
  n: number | undefined;
  tone: "emerald" | "rose";
}) {
  if (value === undefined || n === undefined || n === 0) {
    return <span className="text-ink/30 text-xs">—</span>;
  }
  const pct = value * 100;
  const bg = tone === "emerald" ? "rgba(16,185,129,0.16)" : "rgba(244,63,94,0.16)";
  const fg = tone === "emerald" ? "rgb(4,120,87)" : "rgb(159,18,57)";
  return (
    <div className="inline-flex flex-col items-end">
      <span className="text-sm font-medium" style={{ color: fg }}>
        {pct.toFixed(0)}%
      </span>
      <div className="w-12 h-1 rounded-full mt-0.5" style={{ background: "rgba(11,13,18,0.06)" }}>
        <div
          className="h-1 rounded-full"
          style={{ width: `${Math.min(100, pct)}%`, background: bg, borderColor: fg }}
        />
      </div>
    </div>
  );
}

function Pill({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string | null;
  options: string[];
  onChange: (v: string | null) => void;
}) {
  return (
    <label className="flex items-center gap-2 text-xs text-ink/60">
      <span className="uppercase tracking-wider">{label}</span>
      <select
        value={value || ""}
        onChange={(e) => onChange(e.target.value || null)}
        className="border border-ink/15 rounded px-2 py-1 text-sm bg-white max-w-[180px]"
      >
        <option value="">All</option>
        {options.map((o) => (
          <option key={o} value={o}>{o}</option>
        ))}
      </select>
    </label>
  );
}
