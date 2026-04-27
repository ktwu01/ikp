import { useMemo } from "react";
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { useBenchmarks, useDensing } from "../data";
import Loading from "../components/Loading";
import { vendorColor } from "../util";
import type { BenchmarkEntry } from "../types";

export default function Densing() {
  const { data, loading } = useDensing();
  const { data: bench } = useBenchmarks();

  const scatter = useMemo(() => {
    if (!data) return [];
    return data.partial_residuals.map((p) => ({
      x: p.months,
      y: p.resid,
      ...p,
    }));
  }, [data]);

  const linePoints = useMemo(() => {
    if (!data || !scatter.length) return { observed: [], densing: [] };
    const xMin = Math.min(...scatter.map((p) => p.x)) - 0.5;
    const xMax = Math.max(...scatter.map((p) => p.x)) + 0.5;
    // Observed line: y = b2 * (months - mean_months); residuals are already mean-zero
    const meanX = scatter.reduce((a, b) => a + b.x, 0) / scatter.length;
    const b2 = data.fit.b2_time;
    const b2d = data.densing_prediction.b2;
    return {
      observed: [
        { x: xMin, y: b2 * (xMin - meanX) },
        { x: xMax, y: b2 * (xMax - meanX) },
      ],
      densing: [
        { x: xMin, y: b2d * (xMin - meanX) },
        { x: xMax, y: b2d * (xMax - meanX) },
      ],
    };
  }, [data, scatter]);

  if (loading || !data) return <Loading what="Densing-Law data" />;

  const f = data.fit;

  return (
    <div className="space-y-10">
      <header>
        <h1 className="text-4xl font-bold tracking-tight">Densing-Law falsification</h1>
        <p className="mt-3 text-ink/70 max-w-3xl">
          The Densing Law claims capability-per-parameter doubles every ~3.5 months. If it held for factual
          knowledge, IKP accuracy would grow <code className="bg-ink/5 px-1.5 rounded">+{data.densing_prediction.b2.toFixed(4)}/month</code>
          {" "}at fixed model size. Across {data.n} dated open-weight models, the observed time coefficient is
          <strong> indistinguishable from zero</strong> — supporting the incompressibility thesis for factual knowledge.
        </p>
      </header>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KV label="Observed β₂ (time)" value={`${f.b2_time >= 0 ? "+" : ""}${f.b2_time.toFixed(4)}/mo`} sub="≈ zero" accent />
        <KV label="95% bootstrap CI" value={`[${f.ci95_b2[0].toFixed(4)}, ${f.ci95_b2[1].toFixed(4)}]`} />
        <KV label="Densing prediction" value={`+${data.densing_prediction.b2.toFixed(4)}/mo`} sub="3.5-mo doubling" danger />
        <KV label="R² gain from time" value={`${f.r2_gain_from_time >= 0 ? "+" : ""}${f.r2_gain_from_time.toFixed(4)}`} sub={`base R² = ${f.baseline_r_squared.toFixed(3)}`} />
      </div>

      <section className="bg-white border border-ink/10 rounded-lg p-6">
        <h2 className="text-lg font-semibold">Time trend after partialling out log₁₀(N)</h2>
        <p className="text-sm text-ink/60 mt-1">
          Each point is an open-weight model released between 2023-09 and 2026-04. The y-axis shows residual
          IKP accuracy once model-size is removed. Hover to see which model.
        </p>
        <div className="h-[480px] -ml-5 mt-4">
          <ResponsiveContainer>
            <ComposedChart margin={{ top: 10, right: 24, bottom: 40, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                type="number"
                dataKey="x"
                name="months (since 2024-01-01)"
                stroke="#6b7280"
                fontSize={12}
                domain={["dataMin - 0.5", "dataMax + 0.5"]}
                tickFormatter={(v) => {
                  const monthsFromRef = v as number;
                  const date = new Date(2024, 0, 1);
                  date.setMonth(date.getMonth() + Math.round(monthsFromRef));
                  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}`;
                }}
                label={{ value: "Release date", position: "insideBottom", offset: -16, fill: "#6b7280", fontSize: 13 }}
              />
              <YAxis
                type="number"
                dataKey="y"
                stroke="#6b7280"
                fontSize={12}
                tickFormatter={(v) => v.toFixed(2)}
                label={{ value: "IKP residual (after log₁₀N)", angle: -90, position: "insideLeft", offset: 14, fill: "#6b7280", fontSize: 13 }}
              />
              <ZAxis range={[60, 60]} />
              <ReferenceLine y={0} stroke="#9ca3af" />
              <Tooltip content={<Tip />} />
              <Line
                data={linePoints.densing}
                dataKey="y"
                stroke="#dc2626"
                strokeWidth={2}
                strokeDasharray="6 3"
                dot={false}
                isAnimationActive={false}
                name="Densing-Law prediction"
              />
              <Line
                data={linePoints.observed}
                dataKey="y"
                stroke="#1d4ed8"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
                name="Observed (≈ 0)"
              />
              <Scatter
                data={scatter}
                shape={(props: any) => {
                  const { cx, cy, payload } = props;
                  return (
                    <circle
                      cx={cx}
                      cy={cy}
                      r={4.5}
                      fill={vendorColor(payload.vendor)}
                      fillOpacity={0.8}
                      stroke={vendorColor(payload.vendor)}
                      strokeWidth={1}
                    />
                  );
                }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="flex gap-6 mt-3 text-xs">
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-5 h-0.5 bg-blue-700" /> observed trend
          </span>
          <span className="flex items-center gap-1.5">
            <span
              className="inline-block w-5 h-0.5"
              style={{ background: "repeating-linear-gradient(90deg, #dc2626 0 4px, transparent 4px 7px)" }}
            />
            Densing-Law prediction
          </span>
        </div>
      </section>

      <section className="bg-white border border-ink/10 rounded-lg p-6">
        <h2 className="text-lg font-semibold">Raw accuracy vs model size (colored by release date)</h2>
        <p className="text-sm text-ink/60 mt-1">
          The clean log-linear scaling with parameters is unperturbed by recency: 2026 models sit on the same
          line as 2023 ones of equivalent size.
        </p>
        <div className="h-[480px] -ml-5 mt-4">
          <ResponsiveContainer>
            <ComposedChart margin={{ top: 10, right: 24, bottom: 40, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                type="number"
                dataKey="log10_params"
                stroke="#6b7280"
                fontSize={12}
                tickFormatter={(v) => {
                  const p = Math.pow(10, v);
                  if (p < 100) return p.toFixed(0) + "B";
                  if (p < 1000) return p.toFixed(0) + "B";
                  return (p / 1000).toFixed(1) + "T";
                }}
                ticks={[0, 0.5, 1, 1.5, 2, 2.5, 3].map((v) => v)}
                label={{ value: "Parameters (log scale)", position: "insideBottom", offset: -16, fill: "#6b7280", fontSize: 13 }}
              />
              <YAxis
                type="number"
                dataKey="pen_acc"
                domain={[0, 1]}
                stroke="#6b7280"
                fontSize={12}
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
                label={{ value: "IKP accuracy", angle: -90, position: "insideLeft", offset: 14, fill: "#6b7280", fontSize: 13 }}
              />
              <ZAxis range={[50, 50]} />
              <Tooltip content={<RawTip />} />
              <Scatter
                data={data.points}
                shape={(props: any) => {
                  const { cx, cy, payload } = props;
                  // Color by months (recency): darker = newer
                  const months = payload.months;
                  const t = Math.min(Math.max((months + 4) / 30, 0), 1);
                  const c = `rgb(${Math.round(255 * (1 - t))}, ${Math.round(120 * (1 - t))}, ${Math.round(50 + 180 * t)})`;
                  return <circle cx={cx} cy={cy} r={4.5} fill={c} fillOpacity={0.82} stroke={c} strokeWidth={1} />;
                }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="flex gap-3 mt-3 text-xs text-ink/60 items-center">
          <span>Color: release date</span>
          <div className="h-2 w-48" style={{ background: "linear-gradient(to right, rgb(255,120,50), rgb(0,0,230))" }} />
          <span>2023 → 2026</span>
        </div>
      </section>

      <div className="bg-accent/5 border border-accent/20 rounded-lg p-5 text-sm text-ink/80 leading-relaxed">
        <strong className="text-accent">Interpretation.</strong> The Densing Law <em>is</em> real for reasoning and
        procedural benchmarks — but procedural capability compresses under better architectures and training recipes.
        Factual knowledge cannot be reconstructed from other facts and is bounded below by Shannon entropy, so it
        does not compress with time at fixed model size. Benchmark saturation is therefore evidence that standard
        benchmarks have stopped measuring the incompressible component of scaling, not that scaling has stopped
        mattering.
      </div>

      {bench && (
        <BenchmarkComparison data={bench} />
      )}
    </div>
  );
}

function BenchmarkComparison({ data }: { data: NonNullable<ReturnType<typeof useBenchmarks>["data"]> }) {
  return (
    <section className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Why not just use MMLU?</h2>
        <p className="mt-2 text-sm text-ink/70 max-w-3xl">
          We collected vendor-published official scores from primary sources (model cards, system cards,
          technical reports) for four widely-reported knowledge benchmarks, then fit each as a parameter-count
          proxy. On every matched subset, IKP explains substantially more variance in log<sub>10</sub>(N) than
          the standard benchmark does — and the time-drift pattern matches the incompressibility thesis:
          reasoning-heavy benchmarks gain roughly <code className="bg-ink/5 px-1 rounded">+1–2 pp/month</code>
          {" "}at fixed model size (Densing-Law confirmed); pure-factual ones (SimpleQA, IKP) stay flat.
        </p>
      </div>

      <div className="overflow-x-auto bg-white border border-ink/10 rounded-lg">
        <table className="min-w-full text-sm tabular-nums">
          <thead className="bg-ink/5 text-left text-ink/60 text-xs uppercase tracking-wider">
            <tr>
              <th className="px-4 py-2.5 font-semibold">Metric</th>
              <th className="px-4 py-2.5 font-semibold text-right">N</th>
              <th className="px-4 py-2.5 font-semibold text-right">R² vs log₁₀(N)</th>
              <th className="px-4 py-2.5 font-semibold text-right">IKP R² (same N)</th>
              <th className="px-4 py-2.5 font-semibold text-right">Time slope (pp/mo)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-ink/10">
            <tr className="bg-blue-50/40">
              <td className="px-4 py-2.5 font-semibold">IKP (full set)</td>
              <td className="px-4 py-2.5 text-right">{data.n_total}</td>
              <td className="px-4 py-2.5 text-right font-bold text-blue-700">{data.ikp_full_fit.r2.toFixed(3)}</td>
              <td className="px-4 py-2.5 text-right text-ink/40">—</td>
              <td className="px-4 py-2.5 text-right">{fmtTime(data.ikp_full_joint.slope_months)}</td>
            </tr>
            {data.benchmarks.map((b) => (
              <tr key={b.key}>
                <td className="px-4 py-2.5 font-medium">{b.label}</td>
                <td className="px-4 py-2.5 text-right">{b.n}</td>
                <td className="px-4 py-2.5 text-right">{b.benchmark_fit.r2.toFixed(3)}</td>
                <td className="px-4 py-2.5 text-right text-blue-700">{b.ikp_fit_same_subset.r2.toFixed(3)}</td>
                <td className={`px-4 py-2.5 text-right ${b.benchmark_joint.slope_months > 0.5 ? "text-red-700 font-medium" : ""}`}>
                  {fmtTime(b.benchmark_joint.slope_months)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-xs text-ink/50 px-4 py-3 border-t border-ink/10">
          Sample sizes differ because vendors do not report all benchmarks for all models. With-search /
          tools-augmented configurations (e.g. GLM-4-32B SimpleQA 88) are excluded.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {data.benchmarks.map((b) => (
          <BenchmarkPanel key={b.key} entry={b} />
        ))}
      </div>

      <div className="bg-ink/5 border border-ink/10 rounded-lg p-5 text-sm text-ink/80 leading-relaxed">
        <strong>Three findings.</strong> (i) On every matched subset, IKP wins on R²; the gap is largest for
        reasoning-heavy GPQA Diamond (0.52 vs 0.90) and smallest for purely-factual SimpleQA (0.90 vs 0.99).
        (ii) Reasoning benchmarks drift sharply: GPQA Diamond gains ≈2 pp/month at fixed N, meaning a 33B
        model improves by ≈24 points across one year of releases without growing — invalidating these
        benchmarks as parameter proxies. (iii) Factual benchmarks behave like IKP: SimpleQA's time slope is
        statistically indistinguishable from zero, supporting the broader claim that the incompressibility
        property holds for the factual subspace specifically rather than for "benchmarks" as a category.
      </div>
    </section>
  );
}

function BenchmarkPanel({ entry }: { entry: BenchmarkEntry }) {
  const xs = entry.points.map((p) => p.log10_params);
  const xMin = Math.min(...xs) - 0.2;
  const xMax = Math.max(...xs) + 0.2;
  const benchLine = [
    { x: xMin, y: entry.benchmark_fit.intercept + entry.benchmark_fit.slope * xMin },
    { x: xMax, y: entry.benchmark_fit.intercept + entry.benchmark_fit.slope * xMax },
  ];
  const ikpLine = [
    { x: xMin, y: entry.ikp_fit_same_subset.intercept + entry.ikp_fit_same_subset.slope * xMin },
    { x: xMax, y: entry.ikp_fit_same_subset.intercept + entry.ikp_fit_same_subset.slope * xMax },
  ];

  const benchPts = entry.points.map((p) => ({ x: p.log10_params, y: p.score, ...p }));
  const ikpPts = entry.points.map((p) => ({ x: p.log10_params, y: p.ikp, ...p }));

  return (
    <div className="bg-white border border-ink/10 rounded-lg p-4">
      <div className="flex items-baseline justify-between mb-1">
        <h3 className="font-semibold">{entry.label}</h3>
        <div className="text-xs text-ink/60 tabular-nums">
          <span className="text-red-700">{entry.label} R²={entry.benchmark_fit.r2.toFixed(3)}</span>
          {" · "}
          <span className="text-blue-700">IKP R²={entry.ikp_fit_same_subset.r2.toFixed(3)}</span>
          {" · "}
          N={entry.n}
        </div>
      </div>
      <div className="h-[260px] -ml-4">
        <ResponsiveContainer>
          <ComposedChart margin={{ top: 8, right: 12, bottom: 28, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              type="number"
              dataKey="x"
              stroke="#6b7280"
              fontSize={11}
              domain={[xMin, xMax]}
              tickFormatter={(v) => {
                const p = Math.pow(10, v);
                if (p < 1) return p.toFixed(1) + "B";
                if (p < 1000) return p.toFixed(0) + "B";
                return (p / 1000).toFixed(1) + "T";
              }}
              label={{ value: "log₁₀(params, B)", position: "insideBottom", offset: -14, fill: "#6b7280", fontSize: 11 }}
            />
            <YAxis
              type="number"
              dataKey="y"
              domain={[0, 100]}
              stroke="#6b7280"
              fontSize={11}
              tickFormatter={(v) => `${v}%`}
            />
            <ZAxis range={[40, 40]} />
            <Tooltip content={<BenchTip benchLabel={entry.label} />} />
            <Line data={benchLine} dataKey="y" stroke="#b91c1c" strokeWidth={1.8} dot={false} isAnimationActive={false} name={entry.label} />
            <Line data={ikpLine} dataKey="y" stroke="#1d4ed8" strokeWidth={1.8} strokeDasharray="5 4" dot={false} isAnimationActive={false} name="IKP" />
            <Scatter
              data={benchPts}
              shape={(props: any) => (
                <circle cx={props.cx} cy={props.cy} r={4} fill="#b91c1c" fillOpacity={0.8} stroke="white" strokeWidth={0.5} />
              )}
            />
            <Scatter
              data={ikpPts}
              shape={(props: any) => (
                <path
                  transform={`translate(${props.cx - 4},${props.cy - 4})`}
                  d="M0 0 L8 8 M0 8 L8 0"
                  stroke="#1d4ed8"
                  strokeWidth={1.4}
                  opacity={0.7}
                />
              )}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="flex gap-4 mt-1 text-[11px] text-ink/60">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-full bg-red-700" />
          {entry.label}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3" style={{ color: "#1d4ed8" }}>×</span>
          IKP (same models)
        </span>
      </div>
    </div>
  );
}

function fmtTime(v: number) {
  const sign = v >= 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}`;
}

function BenchTip({ active, payload, benchLabel }: any) {
  if (!active || !payload || !payload.length) return null;
  const p = payload[0]?.payload;
  if (!p || !p.model) return null;
  return (
    <div className="bg-white border border-ink/15 shadow-lg rounded-lg p-3 text-xs max-w-xs">
      <div className="font-semibold text-sm">{p.model}</div>
      <div className="text-ink/60">{p.vendor} · {p.release_date}</div>
      <div className="grid grid-cols-2 gap-x-3 mt-1.5 tabular-nums">
        <div className="text-ink/50">Params</div><div className="text-right">{Math.pow(10, p.log10_params).toFixed(1)}B</div>
        <div className="text-ink/50">{benchLabel}</div><div className="text-right">{p.score?.toFixed(1)}%</div>
        <div className="text-ink/50">IKP</div><div className="text-right">{p.ikp?.toFixed(1)}%</div>
      </div>
    </div>
  );
}

function KV({ label, value, sub, accent, danger }: { label: string; value: string; sub?: string; accent?: boolean; danger?: boolean }) {
  return (
    <div className={`border rounded-lg p-4 ${accent ? "bg-blue-50 border-blue-200" : danger ? "bg-red-50 border-red-200" : "bg-white border-ink/10"}`}>
      <div className="text-[11px] uppercase tracking-wider text-ink/50 font-medium">{label}</div>
      <div className={`mt-1 text-2xl font-bold tabular-nums ${accent ? "text-blue-700" : danger ? "text-red-700" : ""}`}>{value}</div>
      {sub && <div className="text-xs text-ink/50">{sub}</div>}
    </div>
  );
}

function Tip({ active, payload }: any) {
  if (!active || !payload || !payload.length) return null;
  const p = payload[0].payload;
  if (!p || !p.model) return null;
  return (
    <div className="bg-white border border-ink/15 shadow-lg rounded-lg p-3 text-xs max-w-xs">
      <div className="font-semibold text-sm">{p.model}</div>
      <div className="text-ink/60">{p.vendor}{p.thinking ? " · thinking" : ""}</div>
      <div className="grid grid-cols-2 gap-x-3 mt-1.5 tabular-nums">
        <div className="text-ink/50">Month idx</div><div className="text-right">{p.months.toFixed(1)}</div>
        <div className="text-ink/50">log₁₀(N)</div><div className="text-right">{p.log10_params.toFixed(2)}</div>
        <div className="text-ink/50">Residual</div><div className="text-right">{p.resid.toFixed(3)}</div>
      </div>
    </div>
  );
}

function RawTip({ active, payload }: any) {
  if (!active || !payload || !payload.length) return null;
  const p = payload[0].payload;
  return (
    <div className="bg-white border border-ink/15 shadow-lg rounded-lg p-3 text-xs max-w-xs">
      <div className="font-semibold text-sm">{p.model}</div>
      <div className="text-ink/60">{p.vendor} · {p.release_date}</div>
      <div className="grid grid-cols-2 gap-x-3 mt-1.5 tabular-nums">
        <div className="text-ink/50">Params</div><div className="text-right">{p.params_B}B</div>
        <div className="text-ink/50">Accuracy</div><div className="text-right">{(p.pen_acc * 100).toFixed(1)}%</div>
      </div>
    </div>
  );
}
