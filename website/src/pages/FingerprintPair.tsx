import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import type { FingerprintPairDetail, PairJointWrong, Tier } from "../types";

const TIERS: Tier[] = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"];

export default function FingerprintPair() {
  const { pairId } = useParams<{ pairId: string }>();
  const [detail, setDetail] = useState<FingerprintPairDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!pairId) return;
    setDetail(null);
    setError(null);
    fetch(`data/fingerprint_pairs/${pairId}.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`pair file not found (status ${r.status})`);
        return r.json();
      })
      .then(setDetail)
      .catch((e) => setError(e.message ?? String(e)));
  }, [pairId]);

  if (error) {
    return (
      <div className="space-y-3">
        <Link to="/fingerprint" className="text-sm text-ink/60 hover:text-accent">← back to Fingerprint</Link>
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-800 text-sm">
          Could not load pair <code>{pairId}</code>: {error}
        </div>
      </div>
    );
  }
  if (!detail) {
    return <div className="text-ink/50 text-sm">Loading pair…</div>;
  }

  return <PairDetailView detail={detail} />;
}

function PairDetailView({ detail }: { detail: FingerprintPairDetail }) {
  const [filter, setFilter] = useState<"all" | "same" | "different">("all");
  const [tierFilter, setTierFilter] = useState<Tier | "all">("all");
  const [view, setView] = useState<"both_wrong" | "disagreement">("both_wrong");

  const filteredJointWrong = useMemo(() => {
    return detail.joint_wrong.filter((p) => {
      if (filter === "same" && !p.same_wrong) return false;
      if (filter === "different" && p.same_wrong) return false;
      if (tierFilter !== "all" && p.tier !== tierFilter) return false;
      return true;
    });
  }, [detail.joint_wrong, filter, tierFilter]);

  const filteredDisagreement = useMemo(() => {
    return detail.disagreement.filter((p) =>
      tierFilter === "all" || p.tier === tierFilter
    );
  }, [detail.disagreement, tierFilter]);

  const sameRate = detail.n_joint_wrong > 0
    ? detail.n_same_wrong / detail.n_joint_wrong
    : 0;

  return (
    <div className="space-y-6">
      <div>
        <Link to="/fingerprint" className="text-sm text-ink/60 hover:text-accent">← back to Fingerprint</Link>
      </div>

      <header className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">
          <Link to={`/models/${detail.a}`} className="text-ink hover:text-accent">{detail.a}</Link>
          <span className="text-ink/40 mx-3">↔</span>
          <Link to={`/models/${detail.b}`} className="text-ink hover:text-accent">{detail.b}</Link>
        </h1>
        <div className="text-sm text-ink/60 flex flex-wrap gap-x-6 gap-y-1">
          {detail.family && <span>family: <strong>{detail.family}</strong></span>}
          {detail.vendor_a && detail.vendor_b && (
            <span>vendors: <strong>{detail.vendor_a}</strong> ↔ <strong>{detail.vendor_b}</strong></span>
          )}
          <span>regime: <strong>{detail.class}</strong></span>
          <span>sources: {detail.sources.join(", ")}</span>
        </div>
      </header>

      <section className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
        <KV label="HSS" value={detail.hss.toFixed(3)} sub={hssRegimeNote(detail.hss)} />
        <KV label="Jaccard (J)" value={detail.jaccard.toFixed(3)} sub="wrong-set overlap on T5–T6" />
        <KV label="Both wrong" value={`${detail.n_joint_wrong}`} sub="probes where neither model answered correctly" />
        <KV
          label="Same wrong"
          value={`${detail.n_same_wrong} (${(sameRate * 100).toFixed(1)}%)`}
          sub="of joint-wrong: identical hallucinated answer"
        />
      </section>

      <section className="bg-amber-50/60 border border-amber-300/60 rounded-lg p-4 text-xs text-ink/80">
        <strong>How to read:</strong> "Both wrong" probes are the denominator for HSS — if both
        models hallucinate the <em>same</em> wrong answer, that's evidence they share a training
        signal (weights, base, distillation). The <strong>same wrong</strong> column highlights
        the probes that drive the HSS score upward.
      </section>

      <div className="flex flex-wrap items-center gap-3">
        <ViewToggle view={view} setView={setView} />
        {view === "both_wrong" && (
          <SameToggle filter={filter} setFilter={setFilter} />
        )}
        <TierFilter tier={tierFilter} setTier={setTierFilter} />
        <span className="text-xs text-ink/50 ml-auto">
          showing {view === "both_wrong" ? filteredJointWrong.length : filteredDisagreement.length}
          {" "}of {view === "both_wrong" ? detail.n_joint_wrong : detail.disagreement.length} probes
        </span>
      </div>

      {view === "both_wrong" && <JointWrongTable rows={filteredJointWrong} a={detail.a} b={detail.b} />}
      {view === "disagreement" && (
        <DisagreementTable rows={filteredDisagreement} a={detail.a} b={detail.b} />
      )}
    </div>
  );
}

function KV({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-white border border-ink/10 rounded-lg p-3">
      <div className="text-xs text-ink/50 uppercase tracking-wide">{label}</div>
      <div className="text-xl font-semibold text-ink mt-0.5">{value}</div>
      {sub && <div className="text-xs text-ink/50 mt-0.5">{sub}</div>}
    </div>
  );
}

function ViewToggle({
  view,
  setView,
}: {
  view: "both_wrong" | "disagreement";
  setView: (v: "both_wrong" | "disagreement") => void;
}) {
  return (
    <div className="inline-flex rounded-md border border-ink/15 overflow-hidden text-xs">
      <button
        onClick={() => setView("both_wrong")}
        className={`px-3 py-1.5 ${view === "both_wrong" ? "bg-ink text-paper" : "bg-white text-ink/70 hover:bg-ink/5"}`}
      >
        Both wrong
      </button>
      <button
        onClick={() => setView("disagreement")}
        className={`px-3 py-1.5 border-l border-ink/15 ${view === "disagreement" ? "bg-ink text-paper" : "bg-white text-ink/70 hover:bg-ink/5"}`}
      >
        Disagreement
      </button>
    </div>
  );
}

function SameToggle({
  filter,
  setFilter,
}: {
  filter: "all" | "same" | "different";
  setFilter: (v: "all" | "same" | "different") => void;
}) {
  const opts: { val: typeof filter; label: string }[] = [
    { val: "all", label: "All" },
    { val: "same", label: "Same wrong" },
    { val: "different", label: "Different wrong" },
  ];
  return (
    <div className="inline-flex rounded-md border border-ink/15 overflow-hidden text-xs">
      {opts.map((o, i) => (
        <button
          key={o.val}
          onClick={() => setFilter(o.val)}
          className={`px-3 py-1.5 ${i > 0 ? "border-l border-ink/15" : ""} ${
            filter === o.val ? "bg-ink text-paper" : "bg-white text-ink/70 hover:bg-ink/5"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function TierFilter({
  tier,
  setTier,
}: {
  tier: Tier | "all";
  setTier: (v: Tier | "all") => void;
}) {
  return (
    <div className="inline-flex rounded-md border border-ink/15 overflow-hidden text-xs">
      <button
        onClick={() => setTier("all")}
        className={`px-3 py-1.5 ${tier === "all" ? "bg-ink text-paper" : "bg-white text-ink/70 hover:bg-ink/5"}`}
      >
        All tiers
      </button>
      {TIERS.map((t) => (
        <button
          key={t}
          onClick={() => setTier(t)}
          className={`px-3 py-1.5 border-l border-ink/15 ${
            tier === t ? "bg-ink text-paper" : "bg-white text-ink/70 hover:bg-ink/5"
          }`}
        >
          {t}
        </button>
      ))}
    </div>
  );
}

function JointWrongTable({ rows, a, b }: { rows: PairJointWrong[]; a: string; b: string }) {
  if (rows.length === 0) {
    return <div className="text-sm text-ink/50 italic">No probes match this filter.</div>;
  }
  return (
    <div className="bg-white border border-ink/10 rounded-lg divide-y divide-ink/5">
      {rows.map((r) => (
        <div key={r.probe_id} className="p-4 space-y-2">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-xs text-ink/60">
            <Link to={`/probes/${r.probe_id}`} className="font-mono text-ink/80 hover:text-accent">
              {r.probe_id}
            </Link>
            <span className="px-2 py-0.5 rounded bg-ink/10 text-ink/70">{r.tier}</span>
            {r.domain && <span>{r.domain}</span>}
            {r.same_wrong && (
              <span className="px-2 py-0.5 rounded bg-amber-100 text-amber-800 font-medium">
                same wrong answer
              </span>
            )}
          </div>
          <div className="text-sm text-ink">
            <span className="text-ink/50 text-xs uppercase tracking-wide mr-2">Q:</span>
            {r.question}
          </div>
          <div className="text-sm">
            <span className="text-ink/50 text-xs uppercase tracking-wide mr-2">Gold:</span>
            <span className="text-emerald-700 font-medium">{r.gold_answer}</span>
          </div>
          <div className="grid sm:grid-cols-2 gap-3 mt-2">
            <ResponseBox model={a} response={r.response_a} verdict={r.verdict_a} same={r.same_wrong} />
            <ResponseBox model={b} response={r.response_b} verdict={r.verdict_b} same={r.same_wrong} />
          </div>
        </div>
      ))}
    </div>
  );
}

function DisagreementTable({
  rows,
  a,
  b,
}: {
  rows: { probe_id: string; tier: Tier; question: string; gold_answer: string; correct_a: boolean; correct_b: boolean; response_a: string | null; response_b: string | null; verdict_a: string; verdict_b: string }[];
  a: string;
  b: string;
}) {
  if (rows.length === 0) {
    return <div className="text-sm text-ink/50 italic">No probes match this filter.</div>;
  }
  return (
    <div className="bg-white border border-ink/10 rounded-lg divide-y divide-ink/5">
      {rows.map((r) => (
        <div key={r.probe_id} className="p-4 space-y-2">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-xs text-ink/60">
            <Link to={`/probes/${r.probe_id}`} className="font-mono text-ink/80 hover:text-accent">
              {r.probe_id}
            </Link>
            <span className="px-2 py-0.5 rounded bg-ink/10 text-ink/70">{r.tier}</span>
          </div>
          <div className="text-sm text-ink">
            <span className="text-ink/50 text-xs uppercase tracking-wide mr-2">Q:</span>
            {r.question}
          </div>
          <div className="text-sm">
            <span className="text-ink/50 text-xs uppercase tracking-wide mr-2">Gold:</span>
            <span className="text-emerald-700 font-medium">{r.gold_answer}</span>
          </div>
          <div className="grid sm:grid-cols-2 gap-3 mt-2">
            <ResponseBox
              model={a}
              response={r.response_a}
              verdict={r.verdict_a}
              correct={r.correct_a}
            />
            <ResponseBox
              model={b}
              response={r.response_b}
              verdict={r.verdict_b}
              correct={r.correct_b}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function ResponseBox({
  model,
  response,
  verdict,
  correct,
  same,
}: {
  model: string;
  response: string | null;
  verdict: string;
  correct?: boolean;
  same?: boolean;
}) {
  const tone =
    correct === true
      ? "border-emerald-300 bg-emerald-50/60"
      : same
        ? "border-amber-300 bg-amber-50/60"
        : "border-ink/15 bg-ink/[0.02]";
  return (
    <div className={`border rounded-lg p-3 text-sm ${tone}`}>
      <div className="flex items-center justify-between text-xs text-ink/60 mb-1">
        <Link to={`/models/${model}`} className="font-mono text-ink/80 hover:text-accent">
          {model}
        </Link>
        <span
          className={`px-1.5 py-0.5 rounded text-[10px] font-medium uppercase tracking-wide ${
            verdict === "CORRECT"
              ? "bg-emerald-100 text-emerald-700"
              : verdict === "REFUSAL"
                ? "bg-slate-100 text-slate-600"
                : "bg-rose-100 text-rose-700"
          }`}
        >
          {verdict}
        </span>
      </div>
      <div className="text-ink whitespace-pre-wrap break-words">
        {response && response.trim().length > 0 ? response : <span className="text-ink/40 italic">(empty)</span>}
      </div>
    </div>
  );
}

function hssRegimeNote(hss: number): string {
  if (hss >= 0.30) return "shared-base regime";
  if (hss >= 0.10) return "lineage regime";
  return "independent retrain";
}
