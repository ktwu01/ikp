import { useMemo } from "react";
import { Link } from "react-router-dom";
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { useCalibration, useDensing, useFingerprint, useModels } from "../data";
import { formatPercent, vendorColor } from "../util";
import type { CalibrationData, ProprietaryEstimate } from "../types";

const X_TICKS = [1, 3, 10, 30, 100, 300, 1000];

export default function Home() {
  const { data: calib } = useCalibration();
  const { data: models } = useModels();
  const { data: densing } = useDensing();
  const { data: fp } = useFingerprint();

  return (
    <div className="space-y-12">
      <section className="text-center max-w-3xl mx-auto pt-2">
        <div className="inline-block text-xs uppercase tracking-widest text-accent font-medium mb-3">
          Incompressible Knowledge Probes
        </div>
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight leading-[1.1] text-ink">
          Measuring what frontier models know — and how big they are.
        </h1>
        <p className="text-base md:text-lg text-ink/70 mt-4 leading-relaxed">
          IKP is a 1,400-question benchmark spanning 7 tiers of obscurity. Factual knowledge is
          {" "}<em>incompressible</em> — it cannot be derived by reasoning or architectural
          improvements — making it a clean proxy for parametric storage capacity.
        </p>
        <div className="mt-5 flex gap-3 justify-center text-sm flex-wrap">
          <a
            href="https://arxiv.org/pdf/2604.24827"
            target="_blank"
            rel="noopener noreferrer"
            className="px-4 py-2 rounded-md bg-ink text-paper hover:bg-ink/80 transition"
          >
            Paper
          </a>
          <a
            href="https://github.com/19PINE-AI/ikp"
            target="_blank"
            rel="noopener noreferrer"
            className="px-4 py-2 rounded-md border border-ink/20 text-ink hover:bg-ink/5 transition"
          >
            GitHub
          </a>
          <Link
            to="/calibration"
            className="px-4 py-2 rounded-md border border-ink/20 text-ink hover:bg-ink/5 transition"
          >
            Explore calibration
          </Link>
        </div>
      </section>

      {calib && <CalibrationMini cal={calib} />}

      <section className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="md:col-span-2 bg-white border border-ink/10 rounded-lg p-5">
          <div className="text-xs uppercase tracking-wider text-ink/50 font-medium">Calibration fit</div>
          <div className="mt-2 flex items-end gap-6 flex-wrap">
            <div>
              <div className="text-4xl font-bold tabular-nums leading-none">
                {calib ? calib.fit.r_squared.toFixed(3) : "…"}
              </div>
              <div className="text-xs text-ink/50 mt-1.5">
                R² · {calib ? `${calib.n_calibration} open models` : "…"}
              </div>
            </div>
            <div className="border-l border-ink/10 pl-6">
              <div className="text-2xl font-semibold tabular-nums text-ink/80 leading-none">
                {calib ? calib.loo_cv.r_squared.toFixed(3) : "…"}
              </div>
              <div className="text-xs text-ink/50 mt-1.5">
                LOO-CV · median {calib ? `${calib.loo_cv.median_fold_err.toFixed(2)}× err` : "…"}
              </div>
            </div>
          </div>
        </div>
        <Stat
          label="Models evaluated"
          value={models ? String(models.length) : "…"}
          sub={`across ${calib?.vendors.length ?? "…"} vendors`}
        />
        <Stat
          label="Proprietary estimates"
          value={calib ? String(calib.n_proprietary) : "…"}
          sub="size-from-knowledge"
        />
      </section>

      <section className="grid md:grid-cols-3 gap-4">
        <NavCard
          to="/calibration"
          title="Calibration"
          desc="Log-linear fit, LOO-CV validation, and the MoE total-vs-active analysis. Hover any of 76 points."
        />
        <NavCard
          to="/tiers"
          title="Per-tier heatmap"
          desc="Every model's T1–T7 pattern. T1–T2 saturate; T5–T7 discriminate only frontier models."
        />
        <NavCard
          to="/densing"
          title="Densing-Law falsification"
          desc={`Time coef ≈ 0/month vs Densing's +${(densing?.densing_prediction.b2 ?? 0.0129).toFixed(4)}/month. Rejected at p<10⁻¹⁵.`}
          accent
        />
        <NavCard
          to="/fingerprint"
          title="Knowledge fingerprinting"
          desc={`HSS-based regime classification across ${fp?.families.length ?? "30+"} families and ${fp?.cross_vendor.length ?? "100+"} cross-vendor outliers.`}
        />
        <NavCard
          to="/generations"
          title="Generations & families"
          desc="Cross-generation trajectories for 13 major families + GPT-5 size stratification (nano→pro)."
        />
        <NavCard
          to="/recognition"
          title="What do models know about researchers?"
          desc="Citations vs recognition (ρ=0.58) — residual driven by name uniqueness, named artifacts, field density."
        />
        <NavCard
          to="/hallucination"
          title="Hallucination by vendor"
          desc="On beyond-capability probes, Anthropic refuses ~90%; Google and Microsoft confidently guess ~60%."
        />
        <NavCard
          to="/thinking"
          title="Thinking mode"
          desc="27 base/think pairs. Mean +2.4pp, peaks at T3–T4, vanishes at T7. Retrieval aid, not new knowledge."
        />
        <NavCard
          to="/models"
          title="All 188 models"
          desc="Drill into any model: per-tier accuracy, sample correct/wrong/refusal responses."
        />
        <NavCard
          to="/probes"
          title="The 1,400 probes"
          desc="Browse by tier and domain. Inspect how every model answered each probe."
        />
        <NavCard
          to="/pipeline"
          title="The pipeline"
          desc="Seed generation → external grounding → tier calibration → filtering → evaluation → scaling fit."
        />
      </section>

      {calib && <ProprietaryEstimates cal={calib} />}
    </div>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-white border border-ink/10 rounded-lg p-5">
      <div className="text-xs uppercase tracking-wider text-ink/50 font-medium">{label}</div>
      <div className="mt-2 text-3xl font-bold tabular-nums">{value}</div>
      {sub && <div className="text-xs text-ink/50 mt-1">{sub}</div>}
    </div>
  );
}

function NavCard({ to, title, desc, tag, accent }: { to: string; title: string; desc: string; tag?: string; accent?: boolean }) {
  return (
    <Link
      to={to}
      className={`block border rounded-lg p-5 transition group ${
        accent
          ? "bg-accent/5 border-accent/20 hover:border-accent/50"
          : "bg-white border-ink/10 hover:border-accent/40 hover:shadow-sm"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="text-lg font-semibold group-hover:text-accent transition-colors">{title} →</div>
        {tag && (
          <span className="text-[10px] uppercase tracking-wider font-medium text-ink/40 shrink-0 mt-1">
            {tag}
          </span>
        )}
      </div>
      <div className="text-sm text-ink/60 mt-2 leading-relaxed">{desc}</div>
    </Link>
  );
}

function CalibrationMini({ cal }: { cal: CalibrationData }) {
  const { points, lineData } = useMemo(() => {
    const points = cal.calibration_points.map((p) => ({
      x: Math.log10(p.params_B),
      y: p.accuracy,
      ...p,
    }));
    const xMin = Math.min(...points.map((p) => p.x)) - 0.1;
    const xMax = Math.max(...points.map((p) => p.x)) + 0.4;
    const lineData = [
      { x: xMin, y: cal.fit.slope * xMin + cal.fit.intercept },
      { x: xMax, y: cal.fit.slope * xMax + cal.fit.intercept },
    ];
    return { points, lineData };
  }, [cal]);

  return (
    <section className="bg-white border border-ink/10 rounded-lg p-6">
      <div className="flex justify-between items-start gap-3 flex-wrap mb-3">
        <div>
          <h2 className="text-lg font-semibold">Accuracy vs parameter count</h2>
          <p className="text-xs text-ink/50 mt-1">
            {cal.n_calibration} open-weight models, log-linear fit. Each 10× in parameters ≈ +
            {(cal.fit.slope * 100).toFixed(0)}pp accuracy.
          </p>
        </div>
        <Link to="/calibration" className="text-sm text-accent hover:underline shrink-0">
          Full calibration page →
        </Link>
      </div>
      <div className="h-[420px] -ml-6">
        <ResponsiveContainer>
          <ComposedChart margin={{ top: 10, right: 24, bottom: 36, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              type="number"
              dataKey="x"
              domain={["dataMin - 0.1", "dataMax + 0.4"]}
              tickFormatter={(v) => {
                const p = Math.pow(10, v);
                if (p < 1) return p.toFixed(1);
                if (p < 1000) return `${Math.round(p)}B`;
                return `${(p / 1000).toFixed(1)}T`;
              }}
              ticks={X_TICKS.map((t) => Math.log10(t))}
              stroke="#6b7280"
              fontSize={12}
              label={{ value: "Total parameters (log scale)", position: "insideBottom", offset: -16, fill: "#6b7280", fontSize: 13 }}
            />
            <YAxis
              type="number"
              dataKey="y"
              domain={[0, 1]}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              stroke="#6b7280"
              fontSize={12}
              label={{ value: "IKP accuracy (penalized)", angle: -90, position: "insideLeft", offset: 14, fill: "#6b7280", fontSize: 13 }}
            />
            <ZAxis range={[60, 60]} />
            <Tooltip content={<MiniTooltip />} />
            <Line
              data={lineData}
              dataKey="y"
              stroke="#1e40af"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
              legendType="none"
            />
            <Scatter
              data={points}
              shape={(props: any) => {
                const { cx, cy, payload } = props;
                return (
                  <circle
                    cx={cx}
                    cy={cy}
                    r={payload.arch === "moe" ? 5.5 : 4.5}
                    fill={vendorColor(payload.vendor)}
                    fillOpacity={payload.arch === "moe" ? 0.55 : 0.85}
                    stroke={vendorColor(payload.vendor)}
                    strokeWidth={1.2}
                  />
                );
              }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function MiniTooltip({ active, payload }: any) {
  if (!active || !payload || !payload.length) return null;
  const p = payload[0].payload;
  return (
    <div className="bg-white border border-ink/15 shadow-lg rounded-lg p-3 text-xs space-y-0.5 max-w-xs">
      <div className="font-semibold text-ink text-sm">{p.model}</div>
      {p.vendor && <div className="text-ink/60">{p.vendor} · {p.family}</div>}
      <div className="grid grid-cols-2 gap-x-3 mt-1.5 tabular-nums">
        <div className="text-ink/50">Params</div>
        <div className="text-right">{p.params_B}B {p.arch === "moe" && p.active_B ? `(${p.active_B}A)` : ""}</div>
        <div className="text-ink/50">Accuracy</div>
        <div className="text-right font-medium">{(p.y * 100).toFixed(1)}%</div>
      </div>
    </div>
  );
}

function ProprietaryEstimates({ cal }: { cal: CalibrationData }) {
  const rows = useMemo<ProprietaryEstimate[]>(
    () => [...cal.proprietary_estimates].sort((a, b) => b.estimated_B - a.estimated_B),
    [cal],
  );
  return (
    <section className="bg-white border border-ink/10 rounded-lg overflow-hidden">
      <div className="px-6 py-4 border-b border-ink/10 flex items-start justify-between gap-3 flex-wrap">
        <div>
          <h2 className="text-lg font-semibold">Proprietary parameter estimates ({rows.length})</h2>
          <p className="text-xs text-ink/50 mt-1">
            Closed-source models sized by inverting the calibration curve. Distilled rows
            (<span className="text-amber-700">‡</span>) divide effective capacity by the
            {" "}{cal.distillation?.boost.toFixed(2) ?? "—"}× distill boost. 90% PI ≈ ±
            {cal.fit.pi_factor?.toFixed(2) ?? "—"}× either side.
          </p>
        </div>
        <Link to="/calibration#proprietary" className="text-sm text-accent hover:underline shrink-0">
          Full table →
        </Link>
      </div>
      <div className="overflow-x-auto max-h-[640px]">
        <table className="w-full text-sm">
          <thead className="bg-ink/5 text-left text-ink/60 uppercase text-xs tracking-wide sticky top-0">
            <tr>
              <th className="px-4 py-2">Model</th>
              <th className="px-4 py-2">Vendor</th>
              <th className="px-4 py-2 text-right">Pen. acc</th>
              <th className="px-4 py-2 text-right">Raw acc</th>
              <th className="px-4 py-2 text-right">Est. params</th>
              <th className="px-4 py-2 text-right text-ink/40">90% PI</th>
            </tr>
          </thead>
          <tbody className="tabular-nums">
            {rows.map((m) => {
              const distilled = m.regime === "distilled";
              return (
                <tr key={m.model} className="border-t border-ink/5 hover:bg-ink/5">
                  <td className="px-4 py-2">
                    <Link to={`/models/${m.model}`} className="text-ink hover:text-accent font-medium">
                      {m.model}
                    </Link>
                    {m.thinking && <span className="ml-2 text-xs text-violet-600">thinking</span>}
                    {distilled && (
                      <span
                        className="ml-2 text-xs text-amber-700"
                        title={`Distilled from ${m.distill_anchor ?? "teacher"}; eff. capacity ÷ ${cal.distillation?.boost.toFixed(2)}× boost`}
                      >
                        ‡ distilled
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-2 text-ink/60">{m.vendor}</td>
                  <td className="px-4 py-2 text-right">{formatPercent(m.accuracy)}</td>
                  <td className="px-4 py-2 text-right text-ink/60">{formatPercent(m.raw_accuracy)}</td>
                  <td className="px-4 py-2 text-right font-medium">
                    {formatBigParams(m.estimated_B)}
                    {distilled && (
                      <span className="ml-1 text-xs font-normal text-ink/40">
                        ({formatBigParams(m.estimated_B_eff)} eff.)
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-2 text-right text-ink/40 text-xs">
                    [{formatBigParams(m.pi_lo)}–{formatBigParams(m.pi_hi)}]
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function formatBigParams(b: number | null | undefined): string {
  if (b == null || isNaN(b)) return "—";
  if (b >= 1000) return `~${(b / 1000).toFixed(1)}T`;
  if (b >= 100) return `~${Math.round(b / 10) * 10}B`;
  return `~${b.toFixed(0)}B`;
}
