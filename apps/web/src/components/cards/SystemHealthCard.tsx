interface Metric {
  label: string;
  value: string;
}

export default function SystemHealthCard({ metrics }: { metrics: Metric[] }) {
  return (
    <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
      <h3 className="font-display text-lg">System Health</h3>
      <div className="mt-4 grid grid-cols-2 gap-4">
        {metrics.map((metric) => (
          <div key={metric.label} className="rounded-xl bg-mist px-4 py-3">
            <div className="text-xs text-slate uppercase">{metric.label}</div>
            <div className="text-lg font-display text-ink">{metric.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
