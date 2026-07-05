import { usePipeline } from "../data";
import Loading from "../components/Loading";

export default function Pipeline() {
  const { data, loading } = usePipeline();

  if (loading || !data) return <Loading what="pipeline" />;

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-4xl font-bold tracking-tight">The IKP pipeline</h1>
        <p className="mt-3 text-ink/70 max-w-3xl">
          From seed question generation to scaling-curve fit. Each stage's output feeds the next; the
          calibration step is what gives every probe its tier.
        </p>
      </header>

      <ol className="space-y-4">
        {data.stages.map((s, i) => (
          <li key={s.id} className="bg-white border border-ink/10 rounded-lg p-6 relative overflow-hidden">
            <div className="absolute left-0 top-0 bottom-0 w-1 bg-accent/60" />
            <div className="flex items-start justify-between gap-6">
              <div className="flex-1">
                <div className="flex items-center gap-3">
                  <span className="bg-accent/10 text-accent text-xs font-bold uppercase tracking-wider px-2 py-1 rounded">
                    Stage {i + 1}
                  </span>
                  <h2 className="text-lg font-semibold">{s.name.replace(/^\d+\.\s*/, "")}</h2>
                </div>
                <p className="mt-2 text-ink/70 leading-relaxed text-sm max-w-3xl">{s.description}</p>
                <div className="mt-3 flex flex-wrap gap-4 text-xs text-ink/60">
                  <span><strong className="text-ink/80">Output:</strong> {s.output}</span>
                  <span><strong className="text-ink/80">Count:</strong> {s.count}</span>
                  <code className="bg-ink/[0.05] text-ink/70 px-2 py-0.5 rounded font-mono">{s.script}</code>
                </div>
              </div>
              <div className="text-right shrink-0 hidden md:block">
                <div className="text-3xl font-bold text-ink/15 tabular-nums">{String(i + 1).padStart(2, "0")}</div>
              </div>
            </div>
          </li>
        ))}
      </ol>

      <div className="bg-accent/5 border border-accent/20 rounded-lg p-6">
        <h2 className="text-lg font-semibold text-accent mb-2">Why these seven stages?</h2>
        <p className="text-sm text-ink/80 leading-relaxed">
          The key insight is <em>tier calibration</em> — using six landmark models of varying sizes to
          rank every candidate question by intrinsic difficulty. This decouples the benchmark's
          difficulty distribution from any single model's idiosyncrasies, and gives us a stable
          7-tier scaffold (T1 = all six landmarks correct; T7 = only Gemini 3.1 Pro correct). The
          final scaling-curve fit then maps overall IKP accuracy to log<sub>10</sub>(parameters)
          with R² &gt; 0.9 across 93 open-weight models.
        </p>
      </div>
    </div>
  );
}
