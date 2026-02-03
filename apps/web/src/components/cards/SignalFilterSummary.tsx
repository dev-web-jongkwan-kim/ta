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

        // Fetch signals
        const signalsRes = await fetch(`${apiBase}/api/signals/latest?limit=100`);
        const signals = await signalsRes.json();

        // Fetch settings for thresholds
        const settingsRes = await fetch(`${apiBase}/api/settings`);
        const settings = await settingsRes.json();

        // Calculate stats
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

        setStats({
          total: signals.length,
          passed,
          blocked,
          byReason,
        });

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
      <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
        <h3 className="font-display text-lg">Signal Filters</h3>
        <div className="mt-4 text-sm text-slate">Loading...</div>
      </div>
    );
  }

  const reasonLabels: Record<string, string> = {
    EV_MIN: "EV too low",
    Q05_MIN: "Q05 risk too high",
    MAE_MAX: "MAE too high",
    MISSING_BLOCK: "Missing data",
    DAILY_LOSS: "Daily loss limit",
    MAX_POSITIONS: "Max positions",
  };

  return (
    <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
      <h3 className="font-display text-lg">Signal Filters</h3>

      {/* Summary */}
      <div className="mt-4 grid grid-cols-3 gap-4 text-center">
        <div>
          <div className="text-2xl font-display">{stats?.total ?? 0}</div>
          <div className="text-xs text-slate">Total</div>
        </div>
        <div>
          <div className="text-2xl font-display text-emerald-600">{stats?.passed ?? 0}</div>
          <div className="text-xs text-slate">Passed</div>
        </div>
        <div>
          <div className="text-2xl font-display text-rose-500">{stats?.blocked ?? 0}</div>
          <div className="text-xs text-slate">Blocked</div>
        </div>
      </div>

      {/* Block Reasons */}
      <div className="mt-4 space-y-2">
        <div className="text-xs font-medium text-slate uppercase">Block Reasons</div>
        {stats && Object.entries(stats.byReason)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 5)
          .map(([reason, count]) => (
            <div key={reason} className="flex items-center justify-between text-sm">
              <span className="text-slate">{reasonLabels[reason] || reason}</span>
              <span className="font-medium">{count}</span>
            </div>
          ))
        }
      </div>

      {/* Thresholds */}
      <div className="mt-4 pt-4 border-t border-ink/10 space-y-2">
        <div className="text-xs font-medium text-slate uppercase">Current Thresholds</div>
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="bg-mist rounded-lg p-2 text-center">
            <div className="font-medium">EV_MIN</div>
            <div className="text-slate">{((thresholds?.EV_MIN ?? 0) * 100).toFixed(2)}%</div>
          </div>
          <div className="bg-mist rounded-lg p-2 text-center">
            <div className="font-medium">Q05_MIN</div>
            <div className="text-slate">{((thresholds?.Q05_MIN ?? 0) * 100).toFixed(2)}%</div>
          </div>
          <div className="bg-mist rounded-lg p-2 text-center">
            <div className="font-medium">MAE_MAX</div>
            <div className="text-slate">{((thresholds?.MAE_MAX ?? 0) * 100).toFixed(2)}%</div>
          </div>
        </div>
      </div>
    </div>
  );
}
