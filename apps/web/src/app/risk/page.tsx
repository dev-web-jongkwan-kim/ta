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

interface Settings {
  max_positions: number;
  daily_loss_limit: number;
  leverage: number;
  position_size: number;
  initial_capital: number;
  ev_min: number;
  q05_min: number;
  mae_max: number;
}

export default function RiskPage() {
  const [events, setEvents] = useState<RiskEvent[]>([]);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const apiBase = getClientApiBase();
        const [eventsRes, settingsRes] = await Promise.all([
          fetch(`${apiBase}/api/risk/state`),
          fetch(`${apiBase}/api/settings`),
        ]);

        if (!eventsRes.ok) throw new Error("Failed to fetch risk state");
        if (!settingsRes.ok) throw new Error("Failed to fetch settings");

        const eventsData: RiskEvent[] = await eventsRes.json();
        const settingsData: Settings = await settingsRes.json();

        setEvents(eventsData);
        setSettings(settingsData);
        setLastUpdated(new Date());
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
      {/* Header */}
      <div>
        <h1 className="text-2xl font-display font-semibold text-foreground">Risk Management</h1>
        <p className="text-sm text-foreground-muted mt-1">
          리스크 이벤트 모니터링 및 거래 제한 설정을 관리합니다.
        </p>
      </div>

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
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-display font-semibold text-foreground">Risk Settings</h3>
              {lastUpdated && (
                <span className="text-xs text-foreground-muted">
                  Updated: {lastUpdated.toLocaleTimeString()} (30s interval)
                </span>
              )}
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="bg-background-tertiary/50 rounded-lg p-3">
                <div className="flex items-center text-foreground-muted text-xs">
                  Max Positions <InfoTooltip term="MAX_POSITIONS" />
                </div>
                <div className="font-mono text-lg font-semibold text-foreground mt-1">
                  {settings?.max_positions ?? "-"}
                </div>
              </div>
              <div className="bg-background-tertiary/50 rounded-lg p-3">
                <div className="flex items-center text-foreground-muted text-xs">
                  Daily Loss Limit <InfoTooltip term="DAILY_LOSS" />
                </div>
                <div className="font-mono text-lg font-semibold text-foreground mt-1">
                  {settings?.daily_loss_limit ? `${(settings.daily_loss_limit * 100).toFixed(0)}%` : "-"}
                </div>
              </div>
              <div className="bg-background-tertiary/50 rounded-lg p-3">
                <div className="text-foreground-muted text-xs">Leverage</div>
                <div className="font-mono text-lg font-semibold text-foreground mt-1">
                  {settings?.leverage ? `${settings.leverage}x` : "-"}
                </div>
              </div>
              <div className="bg-background-tertiary/50 rounded-lg p-3">
                <div className="text-foreground-muted text-xs">Position Size</div>
                <div className="font-mono text-lg font-semibold text-foreground mt-1">
                  {settings?.position_size ? `$${settings.position_size}` : "-"}
                </div>
              </div>
            </div>

            {/* Additional Settings */}
            {settings && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
                <div className="bg-background-tertiary/50 rounded-lg p-3">
                  <div className="text-foreground-muted text-xs">Initial Capital</div>
                  <div className="font-mono text-lg font-semibold text-foreground mt-1">
                    ${settings.initial_capital}
                  </div>
                </div>
                <div className="bg-background-tertiary/50 rounded-lg p-3">
                  <div className="text-foreground-muted text-xs">EV Min</div>
                  <div className="font-mono text-lg font-semibold text-foreground mt-1">
                    {(settings.ev_min * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="bg-background-tertiary/50 rounded-lg p-3">
                  <div className="text-foreground-muted text-xs">Q05 Min</div>
                  <div className="font-mono text-lg font-semibold text-foreground mt-1">
                    {(settings.q05_min * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="bg-background-tertiary/50 rounded-lg p-3">
                  <div className="text-foreground-muted text-xs">MAE Max</div>
                  <div className="font-mono text-lg font-semibold text-foreground mt-1">
                    {(settings.mae_max * 100).toFixed(2)}%
                  </div>
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
