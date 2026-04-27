export default function Loading({ what }: { what?: string }) {
  return (
    <div className="flex items-center justify-center py-20">
      <div className="flex items-center gap-3 text-ink/60">
        <div className="w-4 h-4 border-2 border-ink/20 border-t-ink/70 rounded-full animate-spin" />
        <span className="text-sm">Loading {what || ""}…</span>
      </div>
    </div>
  );
}
