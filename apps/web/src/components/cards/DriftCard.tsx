"use client";

import { InfoTooltip } from "@/components/ui/Tooltip";

interface DriftMetric {
  symbol: string;
  psi: string | number;
  missing: string | number;
  latency: string | number;
  outlier_count?: number;
}

export default function DriftCard({ metrics }: { metrics: DriftMetric[] }) {
  const formatPsi = (psi: string | number) => {
    const val = typeof psi === "string" ? parseFloat(psi) : psi;
    return val.toFixed(2);
  };

  const formatMissing = (missing: string | number) => {
    if (typeof missing === "string" && missing.includes("%")) return missing;
    const val = typeof missing === "string" ? parseFloat(missing) : missing;
    return `${(val * 100).toFixed(2)}%`;
  };

  const formatLatency = (latency: string | number) => {
    if (typeof latency === "string" && latency.includes("s")) return latency;
    const val = typeof latency === "string" ? parseFloat(latency) : latency;
    return `${(val / 1000).toFixed(1)}s`;
  };

  const getPsiBadge = (psi: string | number) => {
    const val = typeof psi === "string" ? parseFloat(psi) : psi;
    if (val < 0.1) return "badge-success";
    if (val < 0.2) return "badge-warning";
    return "badge-danger";
  };

  return (
    <div className="card p-5">
      <h3 className="font-display font-semibold text-foreground mb-4">Drift & Quality</h3>

      {/* Header */}
      <div className="flex items-center justify-between text-xs text-foreground-muted border-b border-border pb-2 mb-3">
        <span className="w-20">Symbol</span>
        <span className="flex items-center gap-1">
          PSI <InfoTooltip term="PSI" />
        </span>
        <span className="flex items-center gap-1">
          Missing <InfoTooltip term="Missing" />
        </span>
        <span className="flex items-center gap-1">
          Latency <InfoTooltip term="Latency" />
        </span>
      </div>

      <div className="space-y-2.5">
        {metrics.length === 0 ? (
          <div className="text-center text-foreground-muted text-sm py-4">
            No drift metrics
          </div>
        ) : (
          metrics.map((metric) => (
            <div key={metric.symbol} className="flex items-center justify-between text-sm">
              <span className="font-mono font-medium w-20 text-foreground">{metric.symbol}</span>
              <span className={`badge ${getPsiBadge(metric.psi)}`}>
                {formatPsi(metric.psi)}
              </span>
              <span className="font-mono text-foreground-secondary">{formatMissing(metric.missing)}</span>
              <span className="font-mono text-foreground-secondary">{formatLatency(metric.latency)}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
