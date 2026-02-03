"use client";

import { useEffect, useState } from "react";
import { getClientApiBase } from "@/lib/client-api";

interface Position {
  symbol: string;
  side: string;
  amt: number;
  entry_price: number;
  mark_price: number;
  unrealized_pnl: number;
  notional: number;
  leverage: number;
  event_type: string;
}

export default function PositionsTable() {
  const [positions, setPositions] = useState<Position[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const apiBase = getClientApiBase();
        const res = await fetch(`${apiBase}/api/positions/latest`);
        if (!res.ok) throw new Error("Failed to fetch");
        const data = await res.json();
        setPositions(data.filter((p: Position) => p.event_type === "OPEN" && Math.abs(p.amt) > 0));
      } catch (e) {
        console.error("Failed to fetch positions:", e);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="card p-5">
      <h3 className="font-display font-semibold text-foreground mb-4">Positions</h3>

      {loading ? (
        <div className="space-y-2">
          <div className="skeleton h-10 rounded" />
          <div className="skeleton h-10 rounded" />
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left border-b border-border">
                <th className="table-header pb-2">Symbol</th>
                <th className="table-header pb-2">Side</th>
                <th className="table-header pb-2 text-right">Size</th>
                <th className="table-header pb-2 text-right">Entry</th>
                <th className="table-header pb-2 text-right">PnL</th>
                <th className="table-header pb-2 text-right">Leverage</th>
              </tr>
            </thead>
            <tbody>
              {positions.length === 0 ? (
                <tr>
                  <td colSpan={6} className="py-6 text-center text-foreground-muted">
                    No open positions
                  </td>
                </tr>
              ) : (
                positions.map((pos) => (
                  <tr key={`${pos.symbol}-${pos.side}`} className="table-row">
                    <td className="py-2.5 font-mono font-medium text-foreground">{pos.symbol}</td>
                    <td className="py-2.5">
                      <span className={`badge ${pos.side?.toUpperCase() === "LONG" ? "badge-success" : "badge-danger"}`}>
                        {pos.side}
                      </span>
                    </td>
                    <td className="py-2.5 text-right font-mono text-foreground-secondary">
                      {Math.abs(pos.amt).toFixed(4)}
                    </td>
                    <td className="py-2.5 text-right font-mono text-foreground-secondary">
                      ${pos.entry_price?.toFixed(2)}
                    </td>
                    <td className={`py-2.5 text-right font-mono font-medium ${
                      pos.unrealized_pnl > 0 ? "text-success" :
                      pos.unrealized_pnl < 0 ? "text-danger" : "text-foreground-secondary"
                    }`}>
                      {pos.unrealized_pnl >= 0 ? "+" : ""}${pos.unrealized_pnl?.toFixed(2)}
                    </td>
                    <td className="py-2.5 text-right font-mono text-foreground-muted">
                      {pos.leverage}x
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
