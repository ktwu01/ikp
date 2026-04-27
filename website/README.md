# IKP companion website

Interactive React + Vite + TailwindCSS site that visualizes the
calibration curve, per-tier accuracy, fingerprint heatmap, Densing-Law
falsification, and per-model / per-probe drilldowns from the paper.

Live: <https://01.me/research/ikp>

## Stack

- **React 18** + **TypeScript** + **react-router-dom**
- **Vite 5** (dev server + production build)
- **TailwindCSS 3**
- **Recharts** for plots
- Static-only output (`website/dist/`); no server runtime needed

## Layout

```
website/
├── index.html              # Vite entry HTML
├── package.json            # npm scripts: dev, build, preview, prepare-data
├── vite.config.ts          # base "./" — relative asset paths, deploys at any subpath
├── tsconfig.json           # TS config
├── tailwind.config.js      # Tailwind config
├── postcss.config.js
├── public/
│   └── data/               # ← generated JSON, consumed at runtime
│       ├── calibration.json     # n=89, R²=0.917 fit + LOO + scatter
│       ├── densing.json         # 96-model time-trend bundle
│       ├── fingerprint.json     # within-family + cross-vendor outliers
│       ├── hallucination.json   # vendor-level hallucination rates
│       ├── generations.json     # family / generation trajectories
│       ├── thinking_pairs.json  # 27 base/think pairs
│       ├── recognition.json     # researcher recognition vs citations
│       ├── pipeline.json        # 7-stage probe-generation pipeline
│       ├── models.json          # all 188 evaluated models
│       ├── probes.json          # all 1,400 probes
│       ├── index.json
│       ├── models/<model>.json  # per-model probe-level responses
│       └── tiers/T1..T7.json    # per-tier probe + response bundles
├── scripts/
│   └── prepare_data.py     # rebuilds public/data/* from data/results/
└── src/
    ├── App.tsx             # router shell
    ├── main.tsx
    ├── types.ts
    ├── components/
    └── pages/
        ├── Home.tsx
        ├── Calibration.tsx
        ├── Densing.tsx
        ├── Fingerprint.tsx
        ├── Hallucination.tsx
        ├── Pipeline.tsx
        ├── Models.tsx
        ├── Probes.tsx
        └── …
```

## One-time setup

```bash
cd website
npm install
```

(`node_modules/` is checked in for offline reproducibility on this repo
copy; on a fresh clone, `npm install` will pull deps.)

## Data refresh

Always re-run this after a new model lands in `data/results/` —
otherwise the website will silently render stale numbers:

```bash
make website                                  # from repo root
# equivalent to:
python3 website/scripts/prepare_data.py
```

`prepare_data.py` reads `data/results/evaluation_summary.json`,
`data/results/<model>.json`, `configs/all_models.json`, and the
researcher / Densing CSVs, applies the same `CALIBRATION_EXCLUDE`
list as the paper's `scripts/loo_cv_analysis.py`, and rewrites every
file under `website/public/data/`. It prints a summary of n /
R² / Densing CI so you can sanity-check that the website matches
the paper before deploying.

## Development

```bash
make website-dev         # data refresh + npm install + vite dev server
# → http://localhost:5173
```

Vite hot-reloads `src/` on save. JSON under `public/data/` is served
as static; reload the tab after rerunning `prepare_data.py`.

## Production build

```bash
make website-build       # data refresh + tsc -b + vite build
# Output: website/dist/   (static, deploy this directory)
```

Inspect the build:

```bash
make website-preview     # serves website/dist/ on http://localhost:4173
```

### Subpath deploys

The build uses Vite `base: "./"` and react-router's `HashRouter`, so the
same `dist/` works at any subpath (root, `/research/ikp/`, GitHub Pages,
etc.) with no rebuild. Routes look like `…/research/ikp/#/calibration`.

## Deploy

The build output (`website/dist/`) is a static bundle — any static
host (S3 + CloudFront, GitHub Pages, Cloudflare Pages, Netlify,
Vercel, plain nginx) will serve it. The Makefile ships a default
rsync target:

```bash
make website-deploy
# rsync -avz --delete website/dist/ bojieli@01.me:/var/www/01.me/research/ikp/
```

Override the destination per invocation:

```bash
make website-deploy \
  DEPLOY_HOST=user@yourhost \
  DEPLOY_PATH=/var/www/your-path/
```

For nginx-served `01.me/research/ikp/`, a plain alias is enough — the
hash router keeps every route on `index.html`, so no SPA fallback is
required:

```nginx
location /research/ikp/ {
    alias /var/www/01.me/research/ikp/;
}
```

### GitHub Pages

```bash
make website-build
# Then publish website/dist/ to the gh-pages branch:
cd website && \
  git --work-tree=dist add --all && \
  git --work-tree=dist commit -m "deploy" && \
  git push origin HEAD:gh-pages --force
```

## Sanity checks before deploy

1. `make website-build` exits 0
2. `website/public/data/calibration.json` reports `n=89`, `R²≈0.917`
   (paper-canonical numbers)
3. `make website-preview` and click through `/calibration`,
   `/densing`, `/fingerprint`, `/models/<model>`, `/probes` — each
   should load without console errors.
4. The header in the top right shows `188 models · 1,400 probes ·
   27 vendors`.

If any of those drift, re-run `make website` (data refresh) before
rebuilding.
