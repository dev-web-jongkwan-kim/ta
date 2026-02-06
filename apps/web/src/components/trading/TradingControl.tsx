"use client";

import { useEffect, useState } from "react";

type TradingMode = "off" | "shadow" | "live";

interface TradingStatus {
  is_running: boolean;
  mode: TradingMode;
  session_id?: string;
  started_at?: string;
  running_time_sec?: number;
  total_trades?: number;
  total_pnl?: number;
  win_rate?: number | null;
}

interface TradingControlProps {
  compact?: boolean;
  onStatusChange?: (status: TradingStatus) => void;
}

export default function TradingControl({ compact = false, onStatusChange }: TradingControlProps) {
  const [status, setStatus] = useState<TradingStatus>({
    is_running: false,
    mode: "off",
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch current status on mount
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch("/api/trading/status");
        const data = await res.json();
        setStatus(data);
        onStatusChange?.(data);
      } catch (err) {
        console.error("Failed to fetch trading status:", err);
      }
    };

    fetchStatus();
  }, [onStatusChange]);

  // SSE for real-time updates
  useEffect(() => {
    if (!status.is_running) return;

    const eventSource = new EventSource("/api/trading/stream");

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        // Update stats on position events
        if (data.type === "position_closed") {
          // Refetch status to get updated stats
          fetch("/api/trading/status")
            .then((res) => res.json())
            .then((newStatus) => {
              setStatus(newStatus);
              onStatusChange?.(newStatus);
            })
            .catch(console.error);
        }
      } catch {
        // Ignore parse errors (heartbeats)
      }
    };

    return () => {
      eventSource.close();
    };
  }, [status.is_running, onStatusChange]);

  const handleStart = async (newMode: "shadow" | "live") => {
    if (status.is_running && status.mode === newMode) {
      // Already running in this mode, do nothing
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Stop current session if running
      if (status.is_running) {
        await fetch("/api/trading/stop", { method: "POST" });
      }

      // Start new session
      const res = await fetch(`/api/trading/start?mode=${newMode}`, {
        method: "POST",
      });
      const data = await res.json();

      if (data.status === "error") {
        setError(data.message);
      } else {
        setStatus({
          is_running: true,
          mode: newMode,
          session_id: data.session_id,
          started_at: data.started_at,
          running_time_sec: 0,
          total_trades: 0,
          total_pnl: 0,
        });
      }
    } catch (err) {
      setError("Failed to start trading");
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleStop = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const res = await fetch("/api/trading/stop", { method: "POST" });
      const data = await res.json();

      if (data.status === "error") {
        setError(data.message);
      } else {
        setStatus({
          is_running: false,
          mode: "off",
        });
      }
    } catch (err) {
      setError("Failed to stop trading");
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const formatDuration = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  if (compact) {
    return (
      <div className="card p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className={`status-dot ${
              status.mode === "live" ? "status-dot-success" :
              status.mode === "shadow" ? "status-dot-warning" :
              "status-dot-muted"
            }`} />
            <div>
              <span className="font-medium text-foreground">
                {status.mode === "off" ? "Trading Off" :
                 status.mode === "shadow" ? "Shadow Mode" :
                 "Live Trading"}
              </span>
              {status.is_running && status.running_time_sec !== undefined && (
                <span className="text-xs text-foreground-muted ml-2">
                  {formatDuration(status.running_time_sec)}
                </span>
              )}
            </div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => handleStart("shadow")}
              disabled={isLoading}
              className={`btn btn-sm ${status.mode === "shadow" ? "btn-primary" : "btn-secondary"}`}
            >
              Shadow
            </button>
            <button
              onClick={() => handleStart("live")}
              disabled={isLoading}
              className={`btn btn-sm ${status.mode === "live" ? "bg-success text-white hover:bg-success/90" : "btn-secondary"}`}
            >
              Live
            </button>
            {status.is_running && (
              <button
                onClick={handleStop}
                disabled={isLoading}
                className="btn btn-sm btn-danger"
              >
                Stop
              </button>
            )}
          </div>
        </div>
        {error && (
          <p className="text-xs text-danger mt-2">{error}</p>
        )}
      </div>
    );
  }

  return (
    <div className="card p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-display font-semibold text-foreground">Trading Control</h3>
        <div className="flex items-center gap-2">
          <span className={`status-dot ${
            status.mode === "live" ? "status-dot-success" :
            status.mode === "shadow" ? "status-dot-warning" :
            "status-dot-muted"
          }`} />
          <span className={`text-sm font-medium ${
            status.mode === "live" ? "text-success" :
            status.mode === "shadow" ? "text-warning" :
            "text-foreground-muted"
          }`}>
            {status.mode === "off" ? "Stopped" :
             status.mode === "shadow" ? "Shadow" :
             "Live"}
          </span>
        </div>
      </div>

      {/* Running stats */}
      {status.is_running && (
        <div className="flex gap-4 mb-4 text-sm">
          {status.running_time_sec !== undefined && (
            <span className="text-foreground-secondary">
              Running: <span className="font-mono">{formatDuration(status.running_time_sec)}</span>
            </span>
          )}
          {status.total_trades !== undefined && (
            <span className="text-foreground-secondary">
              Trades: <span className="font-mono">{status.total_trades}</span>
            </span>
          )}
          {status.total_pnl !== undefined && (
            <span className={`${status.total_pnl >= 0 ? "text-success" : "text-danger"}`}>
              PnL: <span className="font-mono">{status.total_pnl >= 0 ? "+" : ""}{status.total_pnl.toFixed(2)}</span>
            </span>
          )}
        </div>
      )}

      <div className="flex flex-wrap gap-3">
        <button
          onClick={() => handleStart("shadow")}
          disabled={isLoading}
          className={`btn flex-1 ${
            status.mode === "shadow"
              ? "bg-warning text-white hover:bg-warning/90"
              : "btn-secondary"
          }`}
        >
          <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
          </svg>
          {isLoading ? "Starting..." : status.mode === "shadow" ? "Shadow Running" : "Start Shadow"}
        </button>

        <button
          onClick={() => handleStart("live")}
          disabled={isLoading}
          className={`btn flex-1 ${
            status.mode === "live"
              ? "bg-success text-white hover:bg-success/90"
              : "btn-secondary"
          }`}
        >
          <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          {isLoading ? "Starting..." : status.mode === "live" ? "Live Running" : "Start Live"}
        </button>

        {status.is_running && (
          <button
            onClick={handleStop}
            disabled={isLoading}
            className="btn btn-danger"
          >
            <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
            </svg>
            {isLoading ? "Stopping..." : "Stop"}
          </button>
        )}
      </div>

      {error && (
        <p className="text-sm text-danger mt-3">{error}</p>
      )}

      <p className="text-xs text-foreground-muted mt-3">
        {status.mode === "off" && "Select a mode to start trading."}
        {status.mode === "shadow" && "Shadow mode: Simulating trades without real orders."}
        {status.mode === "live" && "Live mode: Real orders are being placed."}
      </p>
    </div>
  );
}
