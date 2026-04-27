import { useMemo, useState } from "react";
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
import { useCalibration } from "../data";
import { formatPercent, vendorColor } from "../util";
import Loading from "../components/Loading";

const X_TICKS = [1, 3, 10, 30, 100, 300, 1000];

export default function Calibration() {
  const { data: cal, loading } = useCalibration();
  const [showExcluded, setShowExcluded] = useState(false);
  const [highlightVendor, setHighlightVendor] = useState<string | null>(null);

  const { points, lineData, excluded } = useMemo(() => {
    if (!cal) return { points: [], lineData: [], excluded: [] };
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
    const excluded = cal.excluded_points
      .filter((e) => e.params_B != null && e.accuracy != null)
      .map((e) => ({ x: Math.log10(e.params_B!), y: e.accuracy, ...e }));
    return { points, lineData, excluded };
  }, [cal]);

  if (loading || !cal) return <Loading what="calibration data" />;

  return (
    <div className="space-y-10">
      <header>
        <h1 className="text-4xl font-bold tracking-tight">Calibration curve</h1>
        <p className="mt-3 text-ink/70 max-w-3xl">
          Each point is an open-weight model with known parameter count. The OLS regression line maps
          IKP accuracy to log<sub>10</sub>(parameters). Inverting this regression yields parameter
          estimates for proprietary models.
        </p>
      </header>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KV label="R² (penalized)" value={cal.fit.r_squared.toFixed(4)} />
        <KV label="Slope" value={cal.fit.slope.toFixed(3)} sub="≈16pp per 10×" />
        <KV label="Residual SE" value={cal.fit.residual_se.toFixed(4)} sub="±2.6× 90% PI" />
        <KV label="LOO-CV R²" value={cal.loo_cv.r_squared.toFixed(4)} sub={`median ${cal.loo_cv.median_fold_err.toFixed(2)}× err`} />
      </div>

      <div className="bg-white border border-ink/10 rounded-lg p-6">
        <div className="flex justify-between flex-wrap gap-3 items-start mb-3">
          <div>
            <h2 className="text-lg font-semibold">Accuracy vs parameter count</h2>
            <p className="text-xs text-ink/50">Click vendor chips to highlight. Hover points for details.</p>
          </div>
          <label className="flex items-center gap-2 text-sm text-ink/60 cursor-pointer">
            <input
              type="checkbox"
              checked={showExcluded}
              onChange={(e) => setShowExcluded(e.target.checked)}
              className="rounded"
            />
            Show excluded outliers
          </label>
        </div>

        <div className="h-[480px] -ml-6">
          <ResponsiveContainer>
            <ComposedChart margin={{ top: 10, right: 24, bottom: 36, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                type="number"
                dataKey="x"
                name="log₁₀(params)"
                domain={["dataMin - 0.1", "dataMax + 0.4"]}
                tickFormatter={(v) => {
                  const p = Math.pow(10, v);
                  if (p < 1) return p.toFixed(1);
                  if (p < 100) return p.toFixed(0) + "B";
                  if (p < 1000) return p.toFixed(0) + "B";
                  return (p / 1000).toFixed(1) + "T";
                }}
                ticks={X_TICKS.map((t) => Math.log10(t))}
                stroke="#6b7280"
                fontSize={12}
                label={{ value: "Total parameters (log scale)", position: "insideBottom", offset: -16, fill: "#6b7280", fontSize: 13 }}
              />
              <YAxis
                type="number"
                dataKey="y"
                name="accuracy"
                domain={[0, 1]}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                stroke="#6b7280"
                fontSize={12}
                label={{ value: "IKP accuracy (penalized)", angle: -90, position: "insideLeft", offset: 14, fill: "#6b7280", fontSize: 13 }}
              />
              <ZAxis range={[60, 60]} />
              <Tooltip content={<PointTooltip />} />
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
                  const dim = highlightVendor && payload.vendor !== highlightVendor;
                  return (
                    <circle
                      cx={cx}
                      cy={cy}
                      r={payload.arch === "moe" ? 5.5 : 4.5}
                      fill={vendorColor(payload.vendor)}
                      fillOpacity={dim ? 0.12 : payload.arch === "moe" ? 0.55 : 0.85}
                      stroke={dim ? "transparent" : vendorColor(payload.vendor)}
                      strokeWidth={1.2}
                    />
                  );
                }}
              />
              {showExcluded && (
                <Scatter
                  data={excluded}
                  shape={(props: any) => {
                    const { cx, cy } = props;
                    return (
                      <g>
                        <circle cx={cx} cy={cy} r={5} fill="none" stroke="#ef4444" strokeWidth={1.5} strokeDasharray="2 2" />
                      </g>
                    );
                  }}
                />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-4 flex flex-wrap gap-2">
          <button
            onClick={() => setHighlightVendor(null)}
            className={`text-xs px-2.5 py-1 rounded-full border ${
              highlightVendor === null ? "bg-ink text-paper border-ink" : "border-ink/20 text-ink/60 hover:bg-ink/5"
            }`}
          >
            All vendors
          </button>
          {cal.vendors.map((v) => (
            <button
              key={v}
              onClick={() => setHighlightVendor(highlightVendor === v ? null : v)}
              className="text-xs px-2.5 py-1 rounded-full border transition"
              style={{
                background: highlightVendor === v ? vendorColor(v) + "20" : "transparent",
                borderColor: highlightVendor === v ? vendorColor(v) : "rgba(11,13,18,0.15)",
                color: highlightVendor === v ? vendorColor(v) : "rgba(11,13,18,0.6)",
              }}
            >
              <span className="inline-block w-2 h-2 rounded-full mr-1.5 align-middle" style={{ background: vendorColor(v) }} />
              {v}
            </button>
          ))}
        </div>
      </div>

      {showExcluded && excluded.length > 0 && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-5">
          <h3 className="font-semibold text-red-800 mb-2">Excluded outliers</h3>
          <p className="text-sm text-red-700 mb-3">
            These models have known parameter counts but are excluded from calibration due to pathological
            behavior (excessive refusals or extreme score-to-size mismatch).
          </p>
          <ul className="text-sm space-y-1">
            {cal.excluded_points.map((e) => (
              <li key={e.model} className="text-red-900/80">
                <code className="text-xs bg-red-100 px-1.5 py-0.5 rounded">{e.model}</code> — {e.params_B}B,
                accuracy {formatPercent(e.accuracy)} — <em>{e.reason}</em>
              </li>
            ))}
          </ul>
        </div>
      )}

      <LooCVChart cal={cal} />

      <MoeChart cal={cal} />

      <div id="proprietary" className="bg-white border border-ink/10 rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-ink/10">
          <h2 className="text-lg font-semibold">Proprietary parameter estimates ({cal.proprietary_estimates.length})</h2>
          <p className="text-xs text-ink/50 mt-1">
            Inverting the calibration on closed-source IKP scores. <strong>Pen.</strong> = penalized
            accuracy (correct − 0.5·wrong) / total used to fit the curve; <strong>Raw</strong> =
            unpenalized correct / total. Distilled rows (<span className="text-amber-700">‡</span>)
            show <em>actual</em> param estimate (= effective ÷ {cal.distillation?.boost.toFixed(2)}×
            distillation boost from the V4 Flash/Pro anchor pair); the effective single-regime
            value is in parentheses. 90% PI ≈ ±{cal.fit.pi_factor?.toFixed(2)}× in either direction.
          </p>
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
              {cal.proprietary_estimates.map((m) => {
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
      </div>
    </div>
  );
}

function formatBigParams(b: number | null | undefined): string {
  if (b == null || isNaN(b)) return "—";
  if (b >= 1000) return `~${(b / 1000).toFixed(1)}T`;
  if (b >= 100) return `~${Math.round(b / 10) * 10}B`;
  return `~${b.toFixed(0)}B`;
}

function PointTooltip({ active, payload }: any) {
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
        {p.arch && (
          <>
            <div className="text-ink/50">Arch</div>
            <div className="text-right">{p.arch}</div>
          </>
        )}
        {p.thinking && (
          <>
            <div className="text-ink/50">Mode</div>
            <div className="text-right">thinking</div>
          </>
        )}
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

const DEC_TICKS = [1, 3, 10, 30, 100, 300, 1000];

function LooCVChart({ cal }: { cal: any }) {
  const preds = cal.loo_cv.predictions ?? [];
  const points = preds.map((p: any) => ({
    x: Math.log10(p.actual_B),
    y: Math.log10(p.pred_B),
    ...p,
  }));
  if (!points.length) return null;
  const minX = Math.min(...points.map((p: any) => Math.min(p.x, p.y))) - 0.1;
  const maxX = Math.max(...points.map((p: any) => Math.max(p.x, p.y))) + 0.1;
  const diag = [
    { x: minX, y: minX },
    { x: maxX, y: maxX },
  ];
  const band2x = [
    { x: minX, y: minX + Math.log10(2) },
    { x: maxX, y: maxX + Math.log10(2) },
  ];
  const band2xLo = [
    { x: minX, y: minX - Math.log10(2) },
    { x: maxX, y: maxX - Math.log10(2) },
  ];

  return (
    <div className="bg-white border border-ink/10 rounded-lg p-6">
      <h2 className="text-lg font-semibold">Leave-one-out cross-validation</h2>
      <p className="text-xs text-ink/50 mt-1">
        Each point: refit without this model, predict its size from accuracy. Diagonal is perfect; bands are
        within 2×. Hover for details.
      </p>
      <div className="h-[460px] -ml-5 mt-4">
        <ResponsiveContainer>
          <ComposedChart margin={{ top: 10, right: 24, bottom: 40, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              type="number"
              dataKey="x"
              domain={[minX, maxX]}
              tickFormatter={(v) => paramLabel(v)}
              ticks={DEC_TICKS.map((t) => Math.log10(t)).filter((v) => v >= minX && v <= maxX)}
              stroke="#6b7280"
              fontSize={12}
              label={{ value: "Actual params (log)", position: "insideBottom", offset: -16, fill: "#6b7280", fontSize: 13 }}
            />
            <YAxis
              type="number"
              dataKey="y"
              domain={[minX, maxX]}
              tickFormatter={(v) => paramLabel(v)}
              ticks={DEC_TICKS.map((t) => Math.log10(t)).filter((v) => v >= minX && v <= maxX)}
              stroke="#6b7280"
              fontSize={12}
              label={{ value: "LOO predicted params (log)", angle: -90, position: "insideLeft", offset: 14, fill: "#6b7280", fontSize: 13 }}
            />
            <ZAxis range={[50, 50]} />
            <Tooltip content={<LooTip />} />
            <Line data={band2x} dataKey="y" stroke="#10b981" strokeDasharray="3 3" strokeWidth={1.2} dot={false} isAnimationActive={false} />
            <Line data={band2xLo} dataKey="y" stroke="#10b981" strokeDasharray="3 3" strokeWidth={1.2} dot={false} isAnimationActive={false} />
            <Line data={diag} dataKey="y" stroke="#1d4ed8" strokeWidth={2} dot={false} isAnimationActive={false} />
            <Scatter
              data={points}
              shape={(props: any) => {
                const { cx, cy, payload } = props;
                const err = payload.fold_err ?? 1;
                const color = err > 3 ? "#dc2626" : err > 2 ? "#f59e0b" : "#1d4ed8";
                return <circle cx={cx} cy={cy} r={4.5} fill={color} fillOpacity={0.75} stroke={color} strokeWidth={1} />;
              }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function MoeChart({ cal }: { cal: any }) {
  const moe = cal.moe;
  if (!moe?.points?.length) return null;
  const points = moe.points.map((p: any) => ({
    ...p,
    xTotal: Math.log10(p.params_B),
    xActive: Math.log10(p.active_B),
  }));
  return (
    <div className="bg-white border border-ink/10 rounded-lg p-6">
      <h2 className="text-lg font-semibold">MoE models: total vs active parameters</h2>
      <p className="text-xs text-ink/50 mt-1">
        Total parameters predict IKP accuracy (R² = {moe.total?.r_squared.toFixed(2)}); active parameters don't
        (R² = {moe.active?.r_squared.toFixed(2)}). Factual knowledge is stored across all expert weights.
      </p>
      <div className="grid md:grid-cols-2 gap-6 mt-4">
        <MoeScatter
          title={`Total params — R² = ${moe.total?.r_squared.toFixed(2)}`}
          data={points}
          xKey="xTotal"
          bKey="params_B"
          fit={moe.total}
        />
        <MoeScatter
          title={`Active params — R² = ${moe.active?.r_squared.toFixed(2)}`}
          data={points}
          xKey="xActive"
          bKey="active_B"
          fit={moe.active}
        />
      </div>
    </div>
  );
}

function MoeScatter({ title, data, xKey, bKey, fit }: { title: string; data: any[]; xKey: string; bKey: string; fit: any }) {
  const xs = data.map((d) => d[xKey]);
  const minX = Math.min(...xs) - 0.1;
  const maxX = Math.max(...xs) + 0.2;
  const line = fit
    ? [
        { x: minX, y: fit.intercept + fit.slope * minX },
        { x: maxX, y: fit.intercept + fit.slope * maxX },
      ]
    : [];
  return (
    <div>
      <div className="text-sm font-medium text-ink/70 mb-1">{title}</div>
      <div className="h-[300px] -ml-5">
        <ResponsiveContainer>
          <ComposedChart margin={{ top: 5, right: 12, bottom: 34, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              type="number"
              dataKey="x"
              domain={[minX, maxX]}
              tickFormatter={(v) => paramLabel(v)}
              ticks={DEC_TICKS.map((t) => Math.log10(t)).filter((v) => v >= minX && v <= maxX)}
              stroke="#6b7280"
              fontSize={11}
              label={{ value: xKey === "xTotal" ? "Total params" : "Active params", position: "insideBottom", offset: -14, fill: "#6b7280", fontSize: 11 }}
            />
            <YAxis
              type="number"
              dataKey="y"
              domain={[0.1, 0.9]}
              tickFormatter={(v) => `${Math.round(v * 100)}%`}
              stroke="#6b7280"
              fontSize={11}
            />
            <ZAxis range={[45, 45]} />
            <Tooltip content={(props: any) => <MoeTip {...props} bKey={bKey} />} />
            <Line data={line} dataKey="y" stroke="#1d4ed8" strokeWidth={2} dot={false} isAnimationActive={false} />
            <Scatter
              data={data.map((d) => ({ ...d, x: d[xKey], y: d.accuracy }))}
              shape={(props: any) => {
                const { cx, cy, payload } = props;
                const color = vendorColor(payload.vendor);
                return <circle cx={cx} cy={cy} r={5} fill={color} fillOpacity={0.75} stroke={color} strokeWidth={1} />;
              }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function MoeTip({ active, payload, bKey }: any) {
  if (!active || !payload || !payload.length) return null;
  const p = payload[0].payload;
  return (
    <div className="bg-white border border-ink/15 shadow-lg rounded-lg p-3 text-xs">
      <div className="font-semibold text-sm">{p.model}</div>
      <div className="text-ink/60">{p.vendor}</div>
      <div className="tabular-nums">
        <div>{bKey === "params_B" ? "Total" : "Active"}: {p[bKey]}B</div>
        <div>Accuracy: {(p.accuracy * 100).toFixed(1)}%</div>
      </div>
    </div>
  );
}

function LooTip({ active, payload }: any) {
  if (!active || !payload || !payload.length) return null;
  const p = payload[0].payload;
  return (
    <div className="bg-white border border-ink/15 shadow-lg rounded-lg p-3 text-xs">
      <div className="font-semibold text-sm">{p.model}</div>
      <div className="text-ink/60">{p.vendor}</div>
      <div className="tabular-nums">
        <div>Actual: {p.actual_B.toFixed(1)}B</div>
        <div>LOO pred: {p.pred_B.toFixed(1)}B</div>
        {p.fold_err != null && <div>Fold err: {p.fold_err.toFixed(2)}×</div>}
      </div>
    </div>
  );
}

function paramLabel(v: number) {
  const p = Math.pow(10, v);
  if (p < 1) return p.toFixed(1);
  if (p < 1000) return `${Math.round(p)}B`;
  return `${(p / 1000).toFixed(1)}T`;
}
