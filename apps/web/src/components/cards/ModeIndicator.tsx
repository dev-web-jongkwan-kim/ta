const modeLabels: Record<string, { label: string; tone: string; desc: string }> = {
  live: { label: "Live", tone: "bg-emerald/10 text-emerald", desc: "Real orders active" },
  shadow: { label: "Shadow", tone: "bg-amber/10 text-amber", desc: "Simulated orders" },
  off: { label: "Off", tone: "bg-rose/10 text-rose", desc: "Trading disabled" },
};

export default function ModeIndicator({ mode }: { mode: string }) {
  const { label, tone, desc } = modeLabels[mode] ?? {
    label: mode.toUpperCase(),
    tone: "bg-slate/10 text-slate",
    desc: "Runtime mode",
  };
  return (
    <div className="rounded-2xl bg-white/90 p-6 shadow-sm border border-ink/5">
      <div className="flex items-center justify-between">
        <div>
          <p className="font-display text-lg">Runtime Mode</p>
          <p className="text-sm text-slate">{desc}</p>
        </div>
        <span className={`rounded-full px-3 py-1 text-sm font-semibold ${tone}`}>{label}</span>
      </div>
    </div>
  );
}
