interface DriftMetric {
  symbol: string;
  psi: string;
  missing: string;
  latency: string;
}

export default function DriftCard({ metrics }: { metrics: DriftMetric[] }) {
  return (
    <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
      <h3 className="font-display text-lg">Drift & Quality</h3>
      <div className="mt-4 space-y-3">
        {metrics.map((metric) => (
          <div key={metric.symbol} className="flex items-center justify-between text-sm">
            <span className="font-display">{metric.symbol}</span>
            <span className="text-slate">PSI {metric.psi}</span>
            <span className="text-slate">Missing {metric.missing}</span>
            <span className="text-slate">Lag {metric.latency}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
