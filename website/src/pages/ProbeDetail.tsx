import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useTier, useModels } from "../data";
import Loading from "../components/Loading";
import { Tag, VendorTag, VerdictTag } from "../components/Tag";
import { TIER_COLOR, VERDICT_COLOR, formatPercent } from "../util";
import type { Tier } from "../types";

const VERDICTS = ["CORRECT", "WRONG", "REFUSAL"] as const;
type V = (typeof VERDICTS)[number];

export default function ProbeDetail() {
  const { id } = useParams<{ id: string }>();
  const tier = (id?.split("_")[1] || "T1") as Tier;
  const { data: tierFile, loading } = useTier(tier);
  const { data: models } = useModels();
  const [verdictFilter, setVerdictFilter] = useState<V | null>(null);

  const probe = tierFile?.probes[id || ""];

  const responses = useMemo(() => {
    if (!probe || !models) return [];
    const vendorMap = new Map(models.map((m) => [m.model, m]));
    return Object.entries(probe.responses)
      .map(([model, r]) => ({ model, ...r, meta: vendorMap.get(model) }))
      .sort((a, b) => {
        const va = (a.meta?.accuracy ?? 0) - (b.meta?.accuracy ?? 0);
        return -va;
      });
  }, [probe, models]);

  const counts = useMemo(() => {
    const c: Record<V, number> = { CORRECT: 0, WRONG: 0, REFUSAL: 0 };
    for (const r of responses) {
      if (r.verdict in c) c[r.verdict as V]++;
    }
    return c;
  }, [responses]);

  if (loading || !tierFile) return <Loading what={`probe ${id}`} />;
  if (!probe) {
    return (
      <div className="text-center py-20">
        <p className="text-ink/60">Probe not found: <code>{id}</code></p>
        <Link to="/probes" className="text-accent underline">Back to probes</Link>
      </div>
    );
  }

  const total = responses.length;
  const filtered = verdictFilter ? responses.filter((r) => r.verdict === verdictFilter) : responses;

  return (
    <div className="space-y-6">
      <div>
        <Link to="/probes" className="text-sm text-ink/50 hover:text-accent">← All probes</Link>
        <div className="flex items-center gap-3 mt-2">
          <Tag color={TIER_COLOR[tier]}>{tier}</Tag>
          <code className="text-xs text-ink/50">{id}</code>
          {probe.domain && <span className="text-xs text-ink/50">domain: {probe.domain}</span>}
          {probe.source_type && <span className="text-xs text-ink/50">source: {probe.source_type}</span>}
        </div>
        <h1 className="text-2xl font-semibold mt-3 leading-snug">{probe.question}</h1>
        <div className="mt-2 text-sm text-ink/60">
          Gold answer: <span className="font-mono text-ink">{probe.answer}</span>
        </div>
        {probe.evidence && (
          <div className="mt-4 bg-ink/5 border border-ink/10 rounded-lg p-4 text-sm">
            <div className="text-xs uppercase tracking-wider text-ink/50 mb-2">
              Gold evidence bundle (used by the 4-way judge)
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-2">
              {probe.evidence.primary_subfield && (
                <div>
                  <span className="text-ink/50">Primary subfield:</span>{" "}
                  <span className="font-mono text-ink">{probe.evidence.primary_subfield}</span>
                </div>
              )}
              {probe.evidence.secondary_subfields.length > 0 && (
                <div>
                  <span className="text-ink/50">Secondary:</span>{" "}
                  <span className="font-mono text-ink/80">
                    {probe.evidence.secondary_subfields.join(", ")}
                  </span>
                </div>
              )}
              {probe.evidence.affiliations.length > 0 && (
                <div>
                  <span className="text-ink/50">Affiliations:</span>{" "}
                  <span className="text-ink/80">{probe.evidence.affiliations.join(", ")}</span>
                </div>
              )}
              {probe.evidence.named_systems.length > 0 && (
                <div>
                  <span className="text-ink/50">Named systems:</span>{" "}
                  <span className="font-mono text-ink/80">
                    {probe.evidence.named_systems.join(", ")}
                  </span>
                </div>
              )}
              {probe.evidence.co_authors.length > 0 && (
                <div className="md:col-span-2">
                  <span className="text-ink/50">Co-authors:</span>{" "}
                  <span className="text-ink/80">{probe.evidence.co_authors.join(", ")}</span>
                </div>
              )}
              {probe.evidence.venues.length > 0 && (
                <div className="md:col-span-2">
                  <span className="text-ink/50">Venues:</span>{" "}
                  <span className="text-ink/80">{probe.evidence.venues.join("; ")}</span>
                </div>
              )}
              {probe.evidence.top_works.length > 0 && (
                <div className="md:col-span-2 mt-1">
                  <div className="text-ink/50 mb-1">Top works (cited):</div>
                  <ul className="space-y-1 ml-4 list-disc text-ink/80">
                    {probe.evidence.top_works.slice(0, 5).map((w, i) => (
                      <li key={i}>
                        <span className="text-ink">{w.title}</span>{" "}
                        <span className="text-ink/50">({w.year}, cited {w.cited})</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
            <div className="text-xs text-ink/40 mt-3">
              A response receives <strong>CORRECT-STRONG</strong> when it names the primary or
              secondary subfield <em>and</em> at least one matching evidence item;{" "}
              <strong>CORRECT-WEAK</strong> when it names the subfield but cites no specific item.
            </div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-3 gap-3">
        {VERDICTS.map((v) => (
          <button
            key={v}
            onClick={() => setVerdictFilter(verdictFilter === v ? null : v)}
            className={`text-left bg-white border rounded-lg p-4 transition ${
              verdictFilter === v ? "border-current" : "border-ink/10 hover:border-ink/30"
            }`}
            style={{ color: verdictFilter === v ? VERDICT_COLOR[v] : undefined }}
          >
            <div className="text-[11px] uppercase tracking-wider font-medium" style={{ color: VERDICT_COLOR[v] }}>{v}</div>
            <div className="mt-1 text-2xl font-bold tabular-nums">{counts[v]}</div>
            <div className="text-xs text-ink/50">{formatPercent(counts[v] / total, 0)} of {total} models</div>
          </button>
        ))}
      </div>

      {verdictFilter && (
        <button onClick={() => setVerdictFilter(null)} className="text-xs text-ink/60 hover:text-accent">
          ← Show all verdicts
        </button>
      )}

      <div className="bg-white border border-ink/10 rounded-lg overflow-hidden">
        <div className="px-6 py-3 border-b border-ink/10 flex items-center justify-between">
          <h2 className="text-sm font-semibold">{filtered.length} model responses (sorted by overall accuracy)</h2>
        </div>
        <div className="divide-y divide-ink/5 max-h-[800px] overflow-y-auto">
          {filtered.map((r) => (
            <div key={r.model} className="px-6 py-4">
              <div className="flex items-center justify-between gap-3 mb-2">
                <div className="flex items-center gap-3">
                  <Link to={`/models/${r.model}`} className="font-medium text-ink hover:text-accent text-sm">
                    {r.model}
                  </Link>
                  <VendorTag vendor={r.meta?.vendor} />
                  {r.meta && (
                    <span className="text-xs text-ink/40 tabular-nums">
                      overall: {formatPercent(r.meta.accuracy, 0)}
                    </span>
                  )}
                </div>
                <VerdictTag verdict={r.verdict} />
              </div>
              <div className="text-sm text-ink/80 whitespace-pre-wrap bg-ink/[0.02] rounded p-3">
                {r.response || <span className="text-ink/40 italic">— no response —</span>}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
