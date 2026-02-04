"use client";

import { useEffect, useState } from "react";

interface TradingStats {
  started_at: string | null;
  running_time_sec: number;
  total_trades: number;
  wins: number;
  losses: number;
  total_pnl: number;
  gross_profit: number;
  gross_loss: number;
  win_rate: number | null;
  profit_factor: number | null;
  avg_hold_min: number | null;
  best_trade: number | null;
  worst_trade: number | null;
  mode?: string;
}

interface TradingSummaryProps {
  sessionId?: string;
  refreshInterval?: number;
}

export default function TradingSummary({
  sessionId,
  refreshInterval = 5000
}: TradingSummaryProps) {
  const [stats, setStats] = useState<TradingStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const url = sessionId
          ? `/api/trading/stats?session_id=${sessionId}`
          : "/api/trading/stats";
        const res = await fetch(url);
        const data = await res.json();
        setStats(data);
      } catch (error) {
        console.error("Failed to fetch trading stats:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
    const interval = setInterval(fetchStats, refreshInterval);
    return () => clearInterval(interval);
  }, [sessionId, refreshInterval]);

  const formatDuration = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  const formatPnl = (value: number): string => {
    const sign = value >= 0 ? "+" : "";
    return `${sign}${value.toFixed(2)}`;
  };

  if (loading) {
    return (
      <div className="card p-5">
        <div className="animate-pulse">
          <div className="h-5 bg-muted rounded w-40 mb-4" />
          <div className="grid grid-cols-5 gap-4">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="h-16 bg-muted rounded" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="card p-5">
        <p className="text-foreground-muted">No trading data available</p>
      </div>
    );
  }

  return (
    <div className="card p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-display font-semibold text-foreground">
          Trading Summary
        </h3>
        {stats.started_at && (
          <span className="text-sm text-foreground-muted">
            Since: {new Date(stats.started_at).toLocaleDateString()}
          </span>
        )}
      </div>

      {/* Main Stats Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4 mb-4">
        {/* Total Trades */}
        <div className="bg-muted/50 rounded-lg p-3">
          <p className="text-xs text-foreground-muted mb-1">Total Trades</p>
          <p className="text-xl font-mono font-semibold text-foreground">
            {stats.total_trades}
          </p>
        </div>

        {/* Win Rate */}
        <div className="bg-muted/50 rounded-lg p-3">
          <p className="text-xs text-foreground-muted mb-1">Win Rate</p>
          <p className="text-xl font-mono font-semibold text-foreground">
            {stats.win_rate !== null ? `${stats.win_rate.toFixed(1)}%` : "-"}
          </p>
          {stats.total_trades > 0 && (
            <p className="text-xs text-foreground-muted">
              ({stats.wins}/{stats.total_trades})
            </p>
          )}
        </div>

        {/* Total PnL */}
        <div className="bg-muted/50 rounded-lg p-3">
          <p className="text-xs text-foreground-muted mb-1">Total PnL</p>
          <p className={`text-xl font-mono font-semibold ${
            stats.total_pnl >= 0 ? "text-success" : "text-danger"
          }`}>
            {formatPnl(stats.total_pnl)}
          </p>
          <p className="text-xs text-foreground-muted">USDT</p>
        </div>

        {/* Profit Factor */}
        <div className="bg-muted/50 rounded-lg p-3">
          <p className="text-xs text-foreground-muted mb-1">Profit Factor</p>
          <p className={`text-xl font-mono font-semibold ${
            (stats.profit_factor ?? 0) >= 1 ? "text-success" : "text-danger"
          }`}>
            {stats.profit_factor !== null ? stats.profit_factor.toFixed(2) : "-"}
          </p>
        </div>

        {/* Avg Hold Time */}
        <div className="bg-muted/50 rounded-lg p-3">
          <p className="text-xs text-foreground-muted mb-1">Avg Hold</p>
          <p className="text-xl font-mono font-semibold text-foreground">
            {stats.avg_hold_min !== null ? `${stats.avg_hold_min}m` : "-"}
          </p>
        </div>
      </div>

      {/* Secondary Stats */}
      <div className="flex flex-wrap gap-4 text-sm text-foreground-secondary border-t border-border pt-3">
        {stats.running_time_sec > 0 && (
          <span>
            Running: <span className="font-mono">{formatDuration(stats.running_time_sec)}</span>
          </span>
        )}
        {stats.best_trade !== null && (
          <span>
            Best: <span className="font-mono text-success">{formatPnl(stats.best_trade)}</span>
          </span>
        )}
        {stats.worst_trade !== null && (
          <span>
            Worst: <span className="font-mono text-danger">{formatPnl(stats.worst_trade)}</span>
          </span>
        )}
        {stats.gross_profit > 0 && (
          <span>
            Gross Profit: <span className="font-mono text-success">{formatPnl(stats.gross_profit)}</span>
          </span>
        )}
        {stats.gross_loss > 0 && (
          <span>
            Gross Loss: <span className="font-mono text-danger">-{stats.gross_loss.toFixed(2)}</span>
          </span>
        )}
      </div>
    </div>
  );
}
