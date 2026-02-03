const modeConfig: Record<string, { label: string; dotClass: string; badgeClass: string; desc: string }> = {
  live: {
    label: "LIVE",
    dotClass: "status-dot-success",
    badgeClass: "badge-success",
    desc: "Real orders active",
  },
  shadow: {
    label: "SHADOW",
    dotClass: "status-dot-warning",
    badgeClass: "badge-warning",
    desc: "Simulated orders",
  },
  off: {
    label: "OFF",
    dotClass: "status-dot-danger",
    badgeClass: "badge-danger",
    desc: "Trading disabled",
  },
};

export default function ModeIndicator({ mode }: { mode: string }) {
  const config = modeConfig[mode] ?? {
    label: mode.toUpperCase(),
    dotClass: "status-dot-muted",
    badgeClass: "badge-neutral",
    desc: "Unknown mode",
  };

  return (
    <div className="card p-5">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-display font-semibold text-foreground">Runtime Mode</h3>
        <span className={`status-dot ${config.dotClass}`} />
      </div>
      <div className="flex items-center justify-between">
        <span className="text-sm text-foreground-muted">{config.desc}</span>
        <span className={`badge ${config.badgeClass}`}>{config.label}</span>
      </div>
    </div>
  );
}
