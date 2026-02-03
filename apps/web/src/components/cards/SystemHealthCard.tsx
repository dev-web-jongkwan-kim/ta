interface Metric {
  label: string;
  value: string;
  ok?: boolean;
}

export default function SystemHealthCard({ metrics }: { metrics: Metric[] }) {
  return (
    <div className="card p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-display font-semibold text-foreground">System Health</h3>
        <span className="status-dot status-dot-success" title="All systems operational" />
      </div>
      <div className="grid grid-cols-2 gap-3">
        {metrics.map((metric) => (
          <div
            key={metric.label}
            className="bg-background-tertiary/50 rounded-lg px-3 py-2.5"
          >
            <div className="metric-label mb-1">{metric.label}</div>
            <div className={`font-mono text-sm font-medium ${
              metric.ok === false ? "text-danger" :
              metric.ok === true ? "text-success" :
              "text-foreground"
            }`}>
              {metric.value}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
