"use client";

import { useEffect, useState } from "react";
import RiskBlockersCard from "@/components/cards/RiskBlockersCard";
import { getClientApiBase } from "@/lib/client-api";
import { InfoTooltip } from "@/components/ui/Tooltip";

interface RiskEvent {
  ts: string;
  event_type: string;
  symbol: string;
  severity: string;
  message: string;
  details: any;
}

export default function RiskPage() {
  const [events, setEvents] = useState<RiskEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const apiBase = getClientApiBase();
        const res = await fetch(`${apiBase}/api/risk/state`);
        if (!res.ok) throw new Error("Failed to fetch risk state");
        const data: RiskEvent[] = await res.json();
        setEvents(data);
      } catch (e) {
        console.error("Failed to load risk state:", e);
        setError(e instanceof Error ? e.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  const blockerCounts = events.reduce((acc, event) => {
    const type = event.event_type;
    acc[type] = (acc[type] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const defaultBlockerTypes = ["MARGIN_LIMIT", "MAX_POSITIONS", "DATA_STALE", "DAILY_LOSS"];
  const displayBlockers = defaultBlockerTypes.map((type) => ({
    reason: type,
    count: blockerCounts[type] || 0,
  }));

  const recentEvents = events.slice(0, 10);

  const getSeverityBadge = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case "critical":
        return "badge-danger";
      case "warning":
        return "badge-warning";
      case "info":
        return "bg-accent-muted text-accent";
      default:
        return "badge-neutral";
    }
  };

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
            <div className="text-sm text-foreground-muted">
              Cannot connect to API.
            </div>
          </div>
        </div>
      ) : (
        <>
          <RiskBlockersCard blockers={displayBlockers} />

          {/* Recent Risk Events */}
          <div className="card p-5">
            <h3 className="font-display font-semibold text-foreground mb-1">Recent Risk Events</h3>
            <p className="text-xs text-foreground-muted mb-4">Recent risk event logs</p>

            <div className="space-y-2">
              {recentEvents.length === 0 ? (
                <div className="text-center text-foreground-muted text-sm py-6">
                  No risk events recorded yet.
                </div>
              ) : (
                recentEvents.map((event, idx) => (
                  <div
                    key={`${event.ts}-${idx}`}
                    className="flex items-start gap-3 p-3 bg-background-tertiary/50 rounded-lg"
                  >
                    <span className={`badge ${getSeverityBadge(event.severity)}`}>
                      {event.severity || "INFO"}
                    </span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 text-sm">
                        <span className="font-medium text-foreground">{event.event_type}</span>
                        {event.symbol && (
                          <span className="font-mono text-foreground-muted">{event.symbol}</span>
                        )}
                      </div>
                      <div className="text-xs text-foreground-muted mt-1 truncate">{event.message}</div>
                      <div className="text-xs text-foreground-muted/60 mt-1">
                        {new Date(event.ts).toLocaleString()}
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Risk Settings */}
          <div className="card p-5">
            <h3 className="font-display font-semibold text-foreground mb-4">Risk Settings</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="bg-background-tertiary/50 rounded-lg p-3">
                <div className="flex items-center text-foreground-muted text-xs">
                  Max Positions <InfoTooltip term="MAX_POSITIONS" />
                </div>
                <div className="font-mono text-lg font-semibold text-foreground mt-1">5</div>
              </div>
              <div className="bg-background-tertiary/50 rounded-lg p-3">
                <div className="flex items-center text-foreground-muted text-xs">
                  Daily Loss Limit <InfoTooltip term="DAILY_LOSS" />
                </div>
                <div className="font-mono text-lg font-semibold text-foreground mt-1">2%</div>
              </div>
              <div className="bg-background-tertiary/50 rounded-lg p-3">
                <div className="text-foreground-muted text-xs">Leverage</div>
                <div className="font-mono text-lg font-semibold text-foreground mt-1">3x</div>
              </div>
              <div className="bg-background-tertiary/50 rounded-lg p-3">
                <div className="text-foreground-muted text-xs">Position Size</div>
                <div className="font-mono text-lg font-semibold text-foreground mt-1">5%</div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
