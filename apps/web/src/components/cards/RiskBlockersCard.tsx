"use client";

import { InfoTooltip, TERM_DEFINITIONS } from "@/components/ui/Tooltip";

interface Blocker {
  reason: string;
  count: string | number;
}

const BLOCKER_LABELS: Record<string, string> = {
  MARGIN_LIMIT: "Margin Limit",
  MAX_POSITIONS: "Max Positions",
  DATA_STALE: "Data Stale",
  DAILY_LOSS: "Daily Loss",
  CIRCUIT_BREAKER: "Circuit Breaker",
};

export default function RiskBlockersCard({ blockers }: { blockers: Blocker[] }) {
  const getCountStyle = (count: string | number) => {
    const val = typeof count === "string" ? parseInt(count) : count;
    if (val === 0) return "badge-success";
    if (val < 5) return "badge-warning";
    return "badge-danger";
  };

  return (
    <div className="card p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="font-display font-semibold text-foreground">Risk Blockers</h3>
          <p className="text-xs text-foreground-muted mt-0.5">Trade blocking events</p>
        </div>
        {blockers.length > 0 && (
          <span className="badge badge-danger">{blockers.reduce((sum, b) => sum + (typeof b.count === "string" ? parseInt(b.count) : b.count), 0)}</span>
        )}
      </div>

      <div className="space-y-2">
        {blockers.length === 0 ? (
          <div className="text-center text-foreground-muted text-sm py-4">
            No risk events
          </div>
        ) : (
          blockers.map((blocker) => (
            <div key={blocker.reason} className="flex items-center justify-between py-1.5">
              <span className="text-sm text-foreground-secondary flex items-center gap-1">
                {BLOCKER_LABELS[blocker.reason] || blocker.reason}
                {TERM_DEFINITIONS[blocker.reason] && (
                  <InfoTooltip term={blocker.reason} />
                )}
              </span>
              <span className={`badge ${getCountStyle(blocker.count)}`}>
                {blocker.count}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
