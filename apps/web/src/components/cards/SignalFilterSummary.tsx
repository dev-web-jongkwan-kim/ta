"use client";

import { useEffect, useState } from "react";
import { getClientApiBase } from "@/lib/client-api";

interface FilterStats {
  total: number;
  passed: number;
  blocked: number;
  byReason: Record<string, number>;
}

interface FilterThresholds {
  EV_MIN: number;
  Q05_MIN: number;
  MAE_MAX: number;
}

export default function SignalFilterSummary() {
  const [stats, setStats] = useState<FilterStats | null>(null);
  const [thresholds, setThresholds] = useState<FilterThresholds | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const apiBase = getClientApiBase();

        const signalsRes = await fetch(`${apiBase}/api/signals/latest?limit=100`);
        const signals = await signalsRes.json();

        const settingsRes = await fetch(`${apiBase}/api/settings`);
        const settings = await settingsRes.json();

        const byReason: Record<string, number> = {};
        let passed = 0;
        let blocked = 0;

        for (const sig of signals) {
          const reasons = Array.isArray(sig.block_reason_codes)
            ? sig.block_reason_codes
            : [];

          if (reasons.length === 0) {
            passed++;
          } else {
            blocked++;
            for (const reason of reasons) {
              byReason[reason] = (byReason[reason] || 0) + 1;
            }
          }
        }

        setStats({ total: signals.length, passed, blocked, byReason });
        setThresholds({
          EV_MIN: settings.ev_min ?? 0,
          Q05_MIN: settings.q05_min ?? -0.002,
          MAE_MAX: settings.mae_max ?? 0.01,
        });
      } catch (e) {
        console.error("Failed to load signal stats:", e);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="card p-5">
        <h3 className="font-display font-semibold text-foreground mb-4">Signal Filters</h3>
        <div className="space-y-2">
          <div className="skeleton h-8 rounded" />
          <div className="skeleton h-8 rounded" />
          <div className="skeleton h-8 rounded" />
        </div>
      </div>
    );
  }

  const reasonLabels: Record<string, string> = {
    EV_MIN: "EV too low",
    Q05_MIN: "Q05 risk",
    MAE_MAX: "MAE high",
    MISSING_BLOCK: "Missing data",
    DAILY_LOSS: "Daily loss",
    MAX_POSITIONS: "Max positions",
  };

  const passRate = stats ? Math.round((stats.passed / stats.total) * 100) || 0 : 0;

  return (
    <div className="card p-5">
      <h3 className="font-display font-semibold text-foreground mb-4">Signal Filters</h3>

      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="text-center">
          <div className="font-mono text-xl font-semibold text-foreground">{stats?.total ?? 0}</div>
          <div className="metric-label">Total</div>
        </div>
        <div className="text-center">
          <div className="font-mono text-xl font-semibold text-success">{stats?.passed ?? 0}</div>
          <div className="metric-label">Passed</div>
        </div>
        <div className="text-center">
          <div className="font-mono text-xl font-semibold text-danger">{stats?.blocked ?? 0}</div>
          <div className="metric-label">Blocked</div>
        </div>
      </div>

      {/* Pass Rate Bar */}
      <div className="mb-4">
        <div className="flex justify-between text-xs mb-1">
          <span className="text-foreground-muted">Pass Rate</span>
          <span className="font-mono text-foreground">{passRate}%</span>
        </div>
        <div className="h-2 bg-background-tertiary rounded-full overflow-hidden">
          <div
            className="h-full bg-success rounded-full transition-all duration-300"
            style={{ width: `${passRate}%` }}
          />
        </div>
      </div>

      {/* Block Reasons */}
      {stats && Object.keys(stats.byReason).length > 0 && (
        <div className="space-y-1.5 pt-3 border-t border-border">
          <div className="metric-label mb-2">Block Reasons</div>
          {Object.entries(stats.byReason)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 4)
            .map(([reason, count]) => (
              <div key={reason} className="flex items-center justify-between text-sm">
                <span className="text-foreground-secondary">{reasonLabels[reason] || reason}</span>
                <span className="badge badge-danger">{count}</span>
              </div>
            ))}
        </div>
      )}

      {/* Thresholds */}
      {thresholds && (
        <div className="grid grid-cols-3 gap-2 mt-4 pt-3 border-t border-border">
          <div className="bg-background-tertiary/50 rounded p-2 text-center">
            <div className="text-2xs text-foreground-muted">EV_MIN</div>
            <div className="font-mono text-xs text-foreground">{((thresholds.EV_MIN) * 100).toFixed(2)}%</div>
          </div>
          <div className="bg-background-tertiary/50 rounded p-2 text-center">
            <div className="text-2xs text-foreground-muted">Q05_MIN</div>
            <div className="font-mono text-xs text-foreground">{((thresholds.Q05_MIN) * 100).toFixed(2)}%</div>
          </div>
          <div className="bg-background-tertiary/50 rounded p-2 text-center">
            <div className="text-2xs text-foreground-muted">MAE_MAX</div>
            <div className="font-mono text-xs text-foreground">{((thresholds.MAE_MAX) * 100).toFixed(2)}%</div>
          </div>
        </div>
      )}
    </div>
  );
}
