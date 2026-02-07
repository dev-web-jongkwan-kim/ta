"use client";

import { useTradingEvents } from "@/contexts/TradingEventsContext";

export default function OpenPositionsLive() {
  const { positions, connected } = useTradingEvents();

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

  // Count positions by direction
  const longCount = positions.filter((p) => p.side === "LONG").length;
  const shortCount = positions.filter((p) => p.side === "SHORT").length;

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <h3 className="font-display font-semibold text-foreground">
            Open Positions
          </h3>
          <span
            className={`status-dot ${connected ? "status-dot-success" : "status-dot-danger"}`}
            title={connected ? "Live updates" : "Disconnected"}
          />
        </div>
        <div className="flex items-center gap-2">
          <span className="badge badge-success">{longCount} Long</span>
          <span className="badge badge-danger">{shortCount} Short</span>
          <span className="badge badge-neutral">{positions.length} Total</span>
        </div>
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
              className="p-3 bg-background-tertiary/50 rounded-lg border border-border animate-fade-in"
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
