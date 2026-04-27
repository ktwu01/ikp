import { vendorColor } from "../util";

export function VendorTag({ vendor }: { vendor: string | null | undefined }) {
  if (!vendor) return null;
  return (
    <span
      className="inline-flex items-center gap-1.5 text-xs font-medium px-2 py-0.5 rounded-full"
      style={{
        background: vendorColor(vendor) + "1a",
        color: vendorColor(vendor),
      }}
    >
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: vendorColor(vendor) }} />
      {vendor}
    </span>
  );
}

export function Tag({ children, color = "#6b7280" }: { children: React.ReactNode; color?: string }) {
  return (
    <span
      className="inline-block text-xs font-medium px-2 py-0.5 rounded"
      style={{ background: color + "1a", color }}
    >
      {children}
    </span>
  );
}

export function VerdictTag({ verdict }: { verdict: string }) {
  const colors: Record<string, string> = {
    CORRECT: "#10b981",
    WRONG: "#ef4444",
    REFUSAL: "#6b7280",
  };
  return <Tag color={colors[verdict] || "#6b7280"}>{verdict}</Tag>;
}
