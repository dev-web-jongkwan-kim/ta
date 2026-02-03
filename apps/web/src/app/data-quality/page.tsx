"use client";

import { useEffect, useState } from "react";
import DriftCard from "@/components/cards/DriftCard";
import { getClientApiBase } from "@/lib/client-api";

interface DriftMetric {
  ts: string;
  symbol: string;
  schema_version: number;
  psi: number;
  missing_rate: number;
  latency_ms: number;
  outlier_count: number;
}

export default function DataQualityPage() {
  const [metrics, setMetrics] = useState<DriftMetric[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const apiBase = getClientApiBase();
        const res = await fetch(`${apiBase}/api/data-quality/summary`);
        if (!res.ok) throw new Error("Failed to fetch data quality metrics");
        const data: DriftMetric[] = await res.json();

        const latestBySymbol = new Map<string, DriftMetric>();
        for (const metric of data) {
          const existing = latestBySymbol.get(metric.symbol);
          if (!existing || new Date(metric.ts) > new Date(existing.ts)) {
            latestBySymbol.set(metric.symbol, metric);
          }
        }

        setMetrics(Array.from(latestBySymbol.values()));
      } catch (e) {
        console.error("Failed to load data quality:", e);
        setError(e instanceof Error ? e.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  const cardMetrics = metrics.map((m) => ({
    symbol: m.symbol,
    psi: m.psi,
    missing: m.missing_rate,
    latency: m.latency_ms,
    outlier_count: m.outlier_count,
  }));

  return (
    <div className="space-y-6">
      {loading ? (
        <div className="card p-6">
          <div className="space-y-3">
            <div className="skeleton h-8 w-48 rounded" />
            <div className="skeleton h-32 rounded" />
          </div>
        </div>
      ) : error ? (
        <div className="card p-6">
          <div className="text-center py-8">
            <div className="text-danger mb-2">{error}</div>
            <div className="text-sm text-foreground-muted">Cannot connect to API.</div>
          </div>
        </div>
      ) : cardMetrics.length === 0 ? (
        <div className="card p-6">
          <div className="text-center text-foreground-muted py-8">
            No drift metrics recorded yet.
          </div>
        </div>
      ) : (
        <DriftCard metrics={cardMetrics} />
      )}

      {/* Legend */}
      <div className="card p-5">
        <h3 className="font-display font-semibold text-foreground mb-4">PSI Interpretation</h3>
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-success" />
            <span className="text-foreground-secondary">&lt; 0.1: Stable</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-warning" />
            <span className="text-foreground-secondary">0.1 ~ 0.2: Warning</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-danger" />
            <span className="text-foreground-secondary">&gt; 0.2: Drift</span>
          </div>
        </div>
      </div>
    </div>
  );
}
