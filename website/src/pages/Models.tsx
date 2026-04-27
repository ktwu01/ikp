import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useModels } from "../data";
import { formatPercent, formatParams, vendorColor } from "../util";
import Loading from "../components/Loading";
import TierBar from "../components/TierBar";
import { VendorTag } from "../components/Tag";
import type { ModelSummary } from "../types";

type SortKey = "model" | "vendor" | "params_B" | "accuracy" | "raw_accuracy";

export default function Models() {
  const { data: models, loading } = useModels();
  const [query, setQuery] = useState("");
  const [vendorFilter, setVendorFilter] = useState<string | null>(null);
  const [archFilter, setArchFilter] = useState<"all" | "dense" | "moe">("all");
  const [typeFilter, setTypeFilter] = useState<"all" | "open" | "proprietary">("all");
  const [thinkFilter, setThinkFilter] = useState<"all" | "thinking" | "instant">("all");
  const [sortKey, setSortKey] = useState<SortKey>("accuracy");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const vendors = useMemo(() => {
    if (!models) return [];
    return Array.from(new Set(models.map((m) => m.vendor || "unknown"))).sort();
  }, [models]);

  const filtered = useMemo(() => {
    if (!models) return [];
    let xs = models.filter((m) => {
      if (vendorFilter && m.vendor !== vendorFilter) return false;
      if (archFilter !== "all" && m.arch !== archFilter) return false;
      if (typeFilter !== "all" && m.type !== typeFilter) return false;
      if (thinkFilter === "thinking" && !m.thinking) return false;
      if (thinkFilter === "instant" && m.thinking) return false;
      if (query) {
        const q = query.toLowerCase();
        if (
          !m.model.toLowerCase().includes(q) &&
          !(m.vendor || "").toLowerCase().includes(q) &&
          !(m.family || "").toLowerCase().includes(q)
        ) {
          return false;
        }
      }
      return true;
    });
    xs = [...xs].sort((a, b) => {
      const av = pick(a, sortKey);
      const bv = pick(b, sortKey);
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === "string") return sortDir === "asc" ? av.localeCompare(bv as string) : (bv as string).localeCompare(av);
      return sortDir === "asc" ? (av as number) - (bv as number) : (bv as number) - (av as number);
    });
    return xs;
  }, [models, query, vendorFilter, archFilter, typeFilter, thinkFilter, sortKey, sortDir]);

  if (loading || !models) return <Loading what="models" />;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-4xl font-bold tracking-tight">Models</h1>
        <p className="mt-3 text-ink/70 max-w-3xl">
          All {models.length} models evaluated on IKP. Click any model to drill into its per-tier
          accuracy and inspect actual responses.
        </p>
      </header>

      <div className="bg-white border border-ink/10 rounded-lg p-5 space-y-4">
        <div className="flex flex-wrap gap-3 items-center">
          <input
            type="text"
            placeholder="Search model, vendor, family…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="flex-1 min-w-[200px] px-3 py-2 border border-ink/15 rounded-md text-sm focus:outline-none focus:border-accent"
          />
          <Select label="Type" value={typeFilter} onChange={(v) => setTypeFilter(v as any)} options={[["all", "All"], ["open", "Open"], ["proprietary", "Proprietary"]]} />
          <Select label="Arch" value={archFilter} onChange={(v) => setArchFilter(v as any)} options={[["all", "All"], ["dense", "Dense"], ["moe", "MoE"]]} />
          <Select label="Mode" value={thinkFilter} onChange={(v) => setThinkFilter(v as any)} options={[["all", "All"], ["instant", "Instant"], ["thinking", "Thinking"]]} />
        </div>
        <div className="flex flex-wrap gap-1.5">
          <button
            onClick={() => setVendorFilter(null)}
            className={`text-xs px-2 py-0.5 rounded-full border ${
              vendorFilter === null ? "bg-ink text-paper border-ink" : "border-ink/15 text-ink/60 hover:bg-ink/5"
            }`}
          >
            All vendors
          </button>
          {vendors.map((v) => (
            <button
              key={v}
              onClick={() => setVendorFilter(vendorFilter === v ? null : v)}
              className="text-xs px-2 py-0.5 rounded-full border transition"
              style={{
                background: vendorFilter === v ? vendorColor(v) + "22" : "transparent",
                borderColor: vendorFilter === v ? vendorColor(v) : "rgba(11,13,18,0.15)",
                color: vendorFilter === v ? vendorColor(v) : "rgba(11,13,18,0.6)",
              }}
            >
              {v}
            </button>
          ))}
        </div>
        <div className="text-xs text-ink/50">
          {filtered.length} of {models.length} models shown
        </div>
      </div>

      <div className="bg-white border border-ink/10 rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-ink/5 text-left text-ink/60 uppercase text-xs tracking-wide">
              <tr>
                <Th sortKey="model" current={sortKey} dir={sortDir} onSort={(k) => toggleSort(k, sortKey, sortDir, setSortKey, setSortDir)}>
                  Model
                </Th>
                <Th sortKey="vendor" current={sortKey} dir={sortDir} onSort={(k) => toggleSort(k, sortKey, sortDir, setSortKey, setSortDir)}>
                  Vendor
                </Th>
                <Th sortKey="params_B" current={sortKey} dir={sortDir} onSort={(k) => toggleSort(k, sortKey, sortDir, setSortKey, setSortDir)} align="right">
                  Params
                </Th>
                <th className="px-3 py-2">Per-tier</th>
                <Th sortKey="accuracy" current={sortKey} dir={sortDir} onSort={(k) => toggleSort(k, sortKey, sortDir, setSortKey, setSortDir)} align="right">
                  Penalized
                </Th>
                <Th sortKey="raw_accuracy" current={sortKey} dir={sortDir} onSort={(k) => toggleSort(k, sortKey, sortDir, setSortKey, setSortDir)} align="right">
                  Raw
                </Th>
              </tr>
            </thead>
            <tbody className="tabular-nums">
              {filtered.map((m) => (
                <tr key={m.model} className="border-t border-ink/5 hover:bg-ink/5">
                  <td className="px-3 py-2.5">
                    <Link to={`/models/${m.model}`} className="font-medium text-ink hover:text-accent">
                      {m.model}
                    </Link>
                    <div className="text-[11px] text-ink/40 mt-0.5">
                      {m.family || "—"} · {m.arch || "?"}
                      {m.thinking && <span className="ml-1.5 text-violet-600">· thinking</span>}
                      {m.type && <span className="ml-1.5">· {m.type}</span>}
                    </div>
                  </td>
                  <td className="px-3 py-2.5">
                    <VendorTag vendor={m.vendor} />
                  </td>
                  <td className="px-3 py-2.5 text-right text-ink/70">
                    {formatParams(m.params_B)}
                    {m.arch === "moe" && m.active_B && (
                      <span className="text-ink/40 text-xs"> ({formatParams(m.active_B)}A)</span>
                    )}
                  </td>
                  <td className="px-3 py-2.5 w-[180px]">
                    <TierBar tier_accuracy={m.tier_accuracy} />
                  </td>
                  <td className="px-3 py-2.5 text-right font-medium">{formatPercent(m.accuracy)}</td>
                  <td className="px-3 py-2.5 text-right text-ink/60">{formatPercent(m.raw_accuracy)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function pick(m: ModelSummary, k: SortKey): string | number | null {
  if (k === "model") return m.model;
  if (k === "vendor") return m.vendor;
  return m[k];
}

function toggleSort(
  k: SortKey,
  current: SortKey,
  dir: "asc" | "desc",
  setKey: (k: SortKey) => void,
  setDir: (d: "asc" | "desc") => void,
) {
  if (k === current) {
    setDir(dir === "asc" ? "desc" : "asc");
  } else {
    setKey(k);
    setDir(k === "model" || k === "vendor" ? "asc" : "desc");
  }
}

function Th({
  children,
  sortKey,
  current,
  dir,
  onSort,
  align = "left",
}: {
  children: React.ReactNode;
  sortKey: SortKey;
  current: SortKey;
  dir: "asc" | "desc";
  onSort: (k: SortKey) => void;
  align?: "left" | "right";
}) {
  const active = sortKey === current;
  return (
    <th
      className={`px-3 py-2 cursor-pointer select-none ${align === "right" ? "text-right" : "text-left"} ${
        active ? "text-ink" : ""
      }`}
      onClick={() => onSort(sortKey)}
    >
      {children} {active ? (dir === "asc" ? "↑" : "↓") : ""}
    </th>
  );
}

function Select({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: [string, string][];
}) {
  return (
    <label className="flex items-center gap-2 text-xs text-ink/60">
      <span className="uppercase tracking-wider">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="border border-ink/15 rounded px-2 py-1 text-sm bg-white"
      >
        {options.map(([v, l]) => (
          <option key={v} value={v}>
            {l}
          </option>
        ))}
      </select>
    </label>
  );
}
