import { Link, NavLink, Route, Routes } from "react-router-dom";
import { classNames } from "./util";
import Home from "./pages/Home";
import Calibration from "./pages/Calibration";
import Models from "./pages/Models";
import ModelDetail from "./pages/ModelDetail";
import Probes from "./pages/Probes";
import ProbeDetail from "./pages/ProbeDetail";
import Pipeline from "./pages/Pipeline";
import Thinking from "./pages/Thinking";
import TierHeatmap from "./pages/TierHeatmap";
import Densing from "./pages/Densing";
import Fingerprint from "./pages/Fingerprint";
import FingerprintPair from "./pages/FingerprintPair";
import Recognition from "./pages/Recognition";
import Hallucination from "./pages/Hallucination";
import Generations from "./pages/Generations";

type NavItem = { to: string; label: string; end?: boolean };

const NAV: NavItem[] = [
  { to: "/", label: "Overview", end: true },
  { to: "/calibration", label: "Calibration" },
  { to: "/tiers", label: "Tiers" },
  { to: "/densing", label: "Densing Law" },
  { to: "/fingerprint", label: "Fingerprint" },
  { to: "/generations", label: "Generations" },
  { to: "/recognition", label: "Recognition" },
  { to: "/hallucination", label: "Hallucination" },
  { to: "/thinking", label: "Thinking" },
  { to: "/models", label: "Models" },
  { to: "/probes", label: "Probes" },
  { to: "/pipeline", label: "Pipeline" },
];

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-ink/10 bg-paper/85 backdrop-blur-sm sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center gap-6">
          <Link to="/" className="font-bold text-lg tracking-tight flex items-center gap-2 shrink-0">
            <span className="bg-accent text-white w-6 h-6 inline-flex items-center justify-center rounded text-sm">I</span>
            <span>IKP</span>
          </Link>
          <nav className="flex gap-0.5 text-sm items-center">
            {NAV.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                end={n.end}
                className={({ isActive }) =>
                  classNames(
                    "px-2.5 py-1.5 rounded transition-colors whitespace-nowrap",
                    isActive ? "bg-ink text-paper" : "text-ink/70 hover:bg-ink/5 hover:text-ink"
                  )
                }
              >
                {n.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-10">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/calibration" element={<Calibration />} />
          <Route path="/tiers" element={<TierHeatmap />} />
          <Route path="/densing" element={<Densing />} />
          <Route path="/fingerprint" element={<Fingerprint />} />
          <Route path="/fingerprint/:pairId" element={<FingerprintPair />} />
          <Route path="/generations" element={<Generations />} />
          <Route path="/recognition" element={<Recognition />} />
          <Route path="/hallucination" element={<Hallucination />} />
          <Route path="/models" element={<Models />} />
          <Route path="/models/:name" element={<ModelDetail />} />
          <Route path="/probes" element={<Probes />} />
          <Route path="/probes/:id" element={<ProbeDetail />} />
          <Route path="/thinking" element={<Thinking />} />
          <Route path="/pipeline" element={<Pipeline />} />
        </Routes>
      </main>

      <footer className="border-t border-ink/10 mt-12 py-8 text-sm text-ink/60">
        <div className="max-w-7xl mx-auto px-6 flex items-center justify-between flex-wrap gap-4">
          <div>
            <strong>Incompressible Knowledge Probes</strong> — interactive companion to the IKP paper.
          </div>
          <div className="flex gap-4">
            <a href="https://github.com/19PINE-AI/ikp" className="hover:text-ink underline">GitHub</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
