"use client";

import { useEffect, useState } from "react";

interface Trade {
  trade_id: string;
  session_id: string | null;
  trade_group_id: string;
  symbol: string;
  side: "LONG" | "SHORT";
  entry_time: string;
  exit_time: string | null;
  entry_price: number;
  exit_price: number | null;
  qty: number;
  pnl: number | null;
  return_pct: number | null;
  hold_min: number | null;
  exit_reason: string | null;
  is_shadow: boolean;
}

interface TradesResponse {
  trades: Trade[];
  total: number;
  page: number;
  pages: number;
  limit: number;
}

interface TradeHistoryTableProps {
  sessionId?: string;
  symbol?: string;
}

export default function TradeHistoryTable({
  sessionId,
  symbol,
}: TradeHistoryTableProps) {
  const [data, setData] = useState<TradesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [flashTradeId, setFlashTradeId] = useState<string | null>(null);
  const limit = 20;

  const fetchTrades = async () => {
    try {
      const params = new URLSearchParams({
        page: page.toString(),
        limit: limit.toString(),
      });
      if (sessionId) params.append("session_id", sessionId);
      if (symbol) params.append("symbol", symbol);

      const res = await fetch(`/api/trading/trades?${params}`);
      const json = await res.json();
      setData(json);
    } catch (error) {
      console.error("Failed to fetch trades:", error);
    } finally {
      setLoading(false);
    }
  };

  // Initial fetch
  useEffect(() => {
    fetchTrades();
  }, [sessionId, symbol, page]);

  // SSE for real-time updates
  useEffect(() => {
    const eventSource = new EventSource("/api/trading/stream");

    eventSource.onmessage = (event) => {
      try {
        const eventData = JSON.parse(event.data);

        if (eventData.type === "position_closed") {
          // Refetch trades to get the new trade
          fetchTrades();

          // Flash animation for new trade
          setTimeout(() => {
            if (data?.trades?.[0]) {
              setFlashTradeId(data.trades[0].trade_id);
              setTimeout(() => setFlashTradeId(null), 500);
            }
          }, 100);
        }
      } catch {
        // Ignore parse errors (heartbeats)
      }
    };

    return () => {
      eventSource.close();
    };
  }, [sessionId, symbol, page]);

  const formatTime = (isoString: string): string => {
    const date = new Date(isoString);
    return date.toLocaleString("ko-KR", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const formatPrice = (price: number): string => {
    if (price >= 1000) return price.toFixed(1);
    if (price >= 1) return price.toFixed(2);
    return price.toFixed(4);
  };

  const formatPnl = (value: number): string => {
    const sign = value >= 0 ? "+" : "";
    return `${sign}${value.toFixed(2)}`;
  };

  const exportCSV = () => {
    if (!data?.trades.length) return;

    const headers = [
      "Symbol", "Side", "Entry Time", "Exit Time",
      "Entry Price", "Exit Price", "PnL", "Hold (min)", "Exit Reason"
    ];
    const rows = data.trades.map(t => [
      t.symbol,
      t.side,
      t.entry_time,
      t.exit_time || "",
      t.entry_price,
      t.exit_price || "",
      t.pnl || "",
      t.hold_min || "",
      t.exit_reason || ""
    ]);

    const csv = [headers, ...rows].map(row => row.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trades_${new Date().toISOString().split("T")[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (loading) {
    return (
      <div className="card">
        <div className="p-5 border-b border-border">
          <div className="h-5 bg-muted rounded w-32 animate-pulse" />
        </div>
        <div className="p-5">
          <div className="space-y-3">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="h-12 bg-muted rounded animate-pulse" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="p-5 border-b border-border flex items-center justify-between">
        <h3 className="font-display font-semibold text-foreground">
          Trade History
        </h3>
        <button
          onClick={exportCSV}
          disabled={!data?.trades.length}
          className="btn btn-sm btn-secondary"
        >
          <svg className="w-4 h-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          Export CSV
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="bg-muted/30">
              <th className="table-header text-left">Symbol</th>
              <th className="table-header text-left">Side</th>
              <th className="table-header text-left">Entry Time</th>
              <th className="table-header text-left">Exit Time</th>
              <th className="table-header text-right">Entry</th>
              <th className="table-header text-right">Exit</th>
              <th className="table-header text-right">PnL</th>
              <th className="table-header text-right">Hold</th>
            </tr>
          </thead>
          <tbody>
            {data?.trades.length === 0 ? (
              <tr>
                <td colSpan={8} className="table-cell text-center text-foreground-muted py-8">
                  No trades found
                </td>
              </tr>
            ) : (
              data?.trades.map((trade) => (
                <tr
                  key={trade.trade_id}
                  className={`table-row ${flashTradeId === trade.trade_id ? "flash-update" : ""}`}
                >
                  <td className="table-cell font-mono text-sm">
                    {trade.symbol}
                    {trade.is_shadow && (
                      <span className="ml-1 badge badge-neutral text-xs">Shadow</span>
                    )}
                  </td>
                  <td className="table-cell">
                    <span className={`badge ${
                      trade.side === "LONG" ? "badge-success" : "badge-danger"
                    }`}>
                      {trade.side}
                    </span>
                  </td>
                  <td className="table-cell text-sm font-mono text-foreground-secondary">
                    {formatTime(trade.entry_time)}
                  </td>
                  <td className="table-cell text-sm font-mono text-foreground-secondary">
                    {trade.exit_time ? formatTime(trade.exit_time) : "-"}
                  </td>
                  <td className="table-cell text-right font-mono text-sm">
                    {formatPrice(trade.entry_price)}
                  </td>
                  <td className="table-cell text-right font-mono text-sm">
                    {trade.exit_price ? formatPrice(trade.exit_price) : "-"}
                  </td>
                  <td className={`table-cell text-right font-mono text-sm font-medium ${
                    trade.pnl === null ? "text-foreground-muted" :
                    trade.pnl >= 0 ? "text-success" : "text-danger"
                  }`}>
                    {trade.pnl !== null ? formatPnl(trade.pnl) : "-"}
                  </td>
                  <td className="table-cell text-right text-sm text-foreground-secondary">
                    {trade.hold_min !== null ? `${trade.hold_min}m` : "-"}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {data && data.pages > 1 && (
        <div className="p-4 border-t border-border flex items-center justify-between">
          <span className="text-sm text-foreground-muted">
            Showing {(page - 1) * limit + 1}-{Math.min(page * limit, data.total)} of {data.total}
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page === 1}
              className="btn btn-sm btn-secondary disabled:opacity-50"
            >
              Prev
            </button>
            <span className="px-3 py-1 text-sm text-foreground-secondary">
              {page} / {data.pages}
            </span>
            <button
              onClick={() => setPage(p => Math.min(data.pages, p + 1))}
              disabled={page >= data.pages}
              className="btn btn-sm btn-secondary disabled:opacity-50"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
