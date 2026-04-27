import { useMemo } from "react";
import { Link } from "react-router-dom";
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
import { useHallucination } from "../data";
import Loading from "../components/Loading";
import { vendorColor } from "../util";

export default function Hallucination() {
  const { data, loading } = useHallucination();

  const scatter = useMemo(() => {
    if (!data) return [];
    return data.per_model.map((m) => ({
      x: m.accuracy,
      y: m.halluc_rate,
      ...m,
    }));
  }, [data]);

  if (loading || !data) return <Loading what="hallucination data" />;

  return (
    <div className="space-y-10">
      <header>
        <h1 className="text-4xl font-bold tracking-tight">Hallucination calibration</h1>
        <p className="mt-3 text-ink/70 max-w-3xl">
          On T5–T7 probes (which are beyond most models' knowledge), vendors differ enormously in whether they
          refuse or confidently guess. Anthropic models refuse on ~90% of beyond-capability probes; Microsoft and
          Google models confidently hallucinate on 58–66%.
        </p>
      </header>

      <section className="bg-white border border-ink/10 rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-ink/10">
          <h2 className="text-lg font-semibold">By vendor</h2>
          <p className="text-xs text-ink/50 mt-1">Mean hallucination rate on T5–T7 (wrong / total). Higher = more confident guessing.</p>
        </div>
        <table className="w-full text-sm tabular-nums">
          <thead className="bg-ink/5 text-left text-ink/60 uppercase text-xs tracking-wide">
            <tr>
              <th className="px-4 py-2">Vendor</th>
              <th className="px-4 py-2 text-right">Models</th>
              <th className="px-4 py-2 text-right">Mean rate</th>
              <th className="px-4 py-2 text-right">Range</th>
              <th className="px-4 py-2 w-[40%]">Visual</th>
            </tr>
          </thead>
          <tbody>
            {data.vendors.map((v) => {
              const meanPct = v.mean_halluc * 100;
              return (
                <tr key={v.vendor} className="border-t border-ink/5">
                  <td className="px-4 py-2.5 font-medium flex items-center gap-2">
                    <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: vendorColor(v.vendor) }} />
                    {v.vendor}
                  </td>
                  <td className="px-4 py-2.5 text-right text-ink/60">{v.n_models}</td>
                  <td className="px-4 py-2.5 text-right font-medium">{meanPct.toFixed(1)}%</td>
                  <td className="px-4 py-2.5 text-right text-ink/60">
                    {(v.min * 100).toFixed(0)}–{(v.max * 100).toFixed(0)}%
                  </td>
                  <td className="px-4 py-2.5">
                    <div className="h-2.5 bg-ink/[0.06] rounded relative">
                      <div
                        className="absolute inset-y-0 left-0 rounded"
                        style={{ width: `${meanPct}%`, background: vendorColor(v.vendor) }}
                      />
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      <section className="bg-white border border-ink/10 rounded-lg p-6">
        <h2 className="text-lg font-semibold">Accuracy vs hallucination rate</h2>
        <p className="text-xs text-ink/50 mt-1">
          Every model positioned: high-accuracy / low-hallucination corner is the safety-conscious frontier cluster.
          The penalized IKP score docks 0.5pp per hallucination to incentivize calibration.
        </p>
        <div className="h-[500px] -ml-5 mt-4">
          <ResponsiveContainer>
            <ScatterChart margin={{ top: 10, right: 24, bottom: 40, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                type="number"
                dataKey="x"
                domain={[0, 0.9]}
                stroke="#6b7280"
                fontSize={12}
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
                label={{ value: "IKP accuracy (penalized)", position: "insideBottom", offset: -16, fill: "#6b7280", fontSize: 13 }}
              />
              <YAxis
                type="number"
                dataKey="y"
                domain={[0, 1]}
                stroke="#6b7280"
                fontSize={12}
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
                label={{ value: "T5–T7 hallucination rate", angle: -90, position: "insideLeft", offset: 14, fill: "#6b7280", fontSize: 13 }}
              />
              <ZAxis range={[50, 50]} />
              <Tooltip content={<Tip />} />
              <Scatter
                data={scatter}
                shape={(props: any) => {
                  const { cx, cy, payload } = props;
                  return (
                    <circle
                      cx={cx}
                      cy={cy}
                      r={4}
                      fill={vendorColor(payload.vendor)}
                      fillOpacity={0.75}
                      stroke={vendorColor(payload.vendor)}
                      strokeWidth={1}
                    />
                  );
                }}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  );
}

function Tip({ active, payload }: any) {
  if (!active || !payload || !payload.length) return null;
  const p = payload[0].payload;
  return (
    <div className="bg-white border border-ink/15 shadow-lg rounded-lg p-3 text-xs max-w-xs">
      <Link to={`/models/${p.model}`} className="font-semibold text-sm text-ink hover:text-accent block">{p.model}</Link>
      <div className="text-ink/60">{p.vendor}</div>
      <div className="grid grid-cols-2 gap-x-3 mt-1.5 tabular-nums">
        <div className="text-ink/50">Accuracy</div><div className="text-right">{(p.accuracy * 100).toFixed(1)}%</div>
        <div className="text-ink/50">T5–T7 halluc</div><div className="text-right font-medium">{(p.halluc_rate * 100).toFixed(1)}%</div>
        <div className="text-ink/50">Wrong / total</div><div className="text-right">{p.t5_t7_wrong} / {p.t5_t7_total}</div>
      </div>
    </div>
  );
}
