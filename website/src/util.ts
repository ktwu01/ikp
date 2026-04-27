export function formatPercent(v: number, decimals = 1): string {
  return (v * 100).toFixed(decimals) + "%";
}

export function formatParams(b: number | null | undefined): string {
  if (b == null) return "—";
  if (b >= 1000) return `${(b / 1000).toFixed(2)}T`;
  if (b >= 100) return `${b.toFixed(0)}B`;
  if (b >= 10) return `${b.toFixed(1)}B`;
  return `${b.toFixed(2)}B`;
}

export const TIER_COLOR: Record<string, string> = {
  T1: "#10b981",
  T2: "#22c55e",
  T3: "#84cc16",
  T4: "#eab308",
  T5: "#f97316",
  T6: "#ef4444",
  T7: "#7c2d12",
};

export const VERDICT_COLOR: Record<string, string> = {
  CORRECT: "#10b981",
  WRONG: "#ef4444",
  REFUSAL: "#94a3b8",
};

export const VENDOR_COLOR: Record<string, string> = {
  anthropic: "#cc785c",
  openai: "#10a37f",
  google: "#4285f4",
  meta: "#0866ff",
  alibaba: "#ff6a00",
  deepseek: "#4d6bfe",
  xai: "#000000",
  mistral: "#ff7000",
  moonshot: "#7c3aed",
  zhipu: "#1a73e8",
  bytedance: "#fe2c55",
  amazon: "#ff9900",
  microsoft: "#5e5e5e",
  nvidia: "#76b900",
  cohere: "#39594d",
  ibm: "#0530ad",
  baidu: "#2932e1",
  minimax: "#000088",
  stepfun: "#5b21b6",
  xiaomi: "#ff6700",
  tencent: "#0052d9",
  huggingface: "#ffd21e",
  ai21: "#5e5cff",
  allenai: "#1d4ed8",
  inclusionai: "#06b6d4",
};

export function vendorColor(vendor: string | null | undefined): string {
  if (!vendor) return "#6b7280";
  return VENDOR_COLOR[vendor] || "#6b7280";
}

export function classNames(...xs: (string | false | null | undefined)[]): string {
  return xs.filter(Boolean).join(" ");
}
