"use client";

import { useEffect, useState } from "react";

interface Position {
  symbol: string;
  side: "LONG" | "SHORT";
  qty: number;
  entry_price: number;
  entry_time: string;
  sl_price: number | null;
  tp_price: number | null;
  trade_group_id: string | null;
  // Computed fields
  current_price?: number;
  unrealized_pnl?: number;
  unrealized_pnl_pct?: number;
}

interface OpenPositionsProps {
  onPositionUpdate?: (positions: Position[]) => void;
}

export default function OpenPositions({ onPositionUpdate }: OpenPositionsProps) {
  const [positions, setPositions] = useState<Position[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Initial fetch
  useEffect(() => {
    const fetchPositions = async () => {
      try {
        const res = await fetch("/api/trading/positions");
        const data = await res.json();
        setPositions(data.positions || []);
        onPositionUpdate?.(data.positions || []);
      } catch (err) {
        console.error("Failed to fetch positions:", err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchPositions();
  }, [onPositionUpdate]);

  // SSE for real-time updates
  useEffect(() => {
    const eventSource = new EventSource("/api/trading/stream");

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === "position_opened") {
          setPositions((prev) => {
            // Avoid duplicates
            if (prev.some((p) => p.symbol === data.symbol)) {
              return prev;
            }
            const newPosition: Position = {
              symbol: data.symbol,
              side: data.side,
              qty: data.qty,
              entry_price: data.entry_price,
              entry_time: data.entry_time,
              sl_price: data.sl_price,
              tp_price: data.tp_price,
              trade_group_id: null,
            };
            const updated = [...prev, newPosition];
            onPositionUpdate?.(updated);
            return updated;
          });
        }

        if (data.type === "position_closed") {
          setPositions((prev) => {
            const updated = prev.filter((p) => p.symbol !== data.symbol);
            onPositionUpdate?.(updated);
            return updated;
          });
        }
      } catch (err) {
        // Ignore parse errors (heartbeats, etc.)
      }
    };

    eventSource.onerror = () => {
      console.warn("SSE connection error, will retry...");
    };

    return () => {
      eventSource.close();
    };
  }, [onPositionUpdate]);

  const formatTime = (isoString: string): string => {
    const date = new Date(isoString);
    return date.toLocaleTimeString("ko-KR", {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const formatDuration = (isoString: string): string => {
    const entryTime = new Date(isoString);
    const now = new Date();
    const diffMs = now.getTime() - entryTime.getTime();
    const diffMin = Math.floor(diffMs / 60000);

    if (diffMin < 60) {
      return `${diffMin}m`;
    }
    const hours = Math.floor(diffMin / 60);
    const mins = diffMin % 60;
    return `${hours}h ${mins}m`;
  };

  if (isLoading) {
    return (
      <div className="card p-4">
        <h3 className="font-display font-semibold text-foreground mb-4">
          Open Positions
        </h3>
        <div className="skeleton h-20 rounded" />
      </div>
    );
  }

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-display font-semibold text-foreground">
          Open Positions
        </h3>
        <span className="badge badge-neutral">{positions.length}</span>
      </div>

      {positions.length === 0 ? (
        <div className="text-center py-8 text-foreground-muted">
          <svg
            className="w-12 h-12 mx-auto mb-3 opacity-30"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"
            />
          </svg>
          <p className="text-sm">No open positions</p>
        </div>
      ) : (
        <div className="space-y-3">
          {positions.map((position) => (
            <div
              key={position.symbol}
              className="p-3 bg-background-tertiary/50 rounded-lg border border-border"
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="font-mono font-medium text-foreground">
                    {position.symbol}
                  </span>
                  <span
                    className={`badge ${
                      position.side === "LONG" ? "badge-success" : "badge-danger"
                    }`}
                  >
                    {position.side}
                  </span>
                </div>
                <span className="text-xs text-foreground-muted">
                  {formatDuration(position.entry_time)}
                </span>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div>
                  <span className="text-foreground-muted text-xs">Entry</span>
                  <p className="font-mono">
                    {position.entry_price.toLocaleString(undefined, {
                      minimumFractionDigits: 2,
                      maximumFractionDigits: 4,
                    })}
                  </p>
                </div>
                <div>
                  <span className="text-foreground-muted text-xs">Qty</span>
                  <p className="font-mono">{position.qty.toFixed(6)}</p>
                </div>
                <div>
                  <span className="text-foreground-muted text-xs">SL</span>
                  <p className="font-mono text-danger">
                    {position.sl_price
                      ? position.sl_price.toLocaleString(undefined, {
                          minimumFractionDigits: 2,
                          maximumFractionDigits: 4,
                        })
                      : "-"}
                  </p>
                </div>
                <div>
                  <span className="text-foreground-muted text-xs">TP</span>
                  <p className="font-mono text-success">
                    {position.tp_price
                      ? position.tp_price.toLocaleString(undefined, {
                          minimumFractionDigits: 2,
                          maximumFractionDigits: 4,
                        })
                      : "-"}
                  </p>
                </div>
              </div>

              <div className="mt-2 pt-2 border-t border-border/50 flex items-center justify-between text-xs text-foreground-muted">
                <span>Entered at {formatTime(position.entry_time)}</span>
                <span className="status-dot status-dot-warning" title="Monitoring SL/TP" />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
