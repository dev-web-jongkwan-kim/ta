import { fetchJSON } from "@/lib/api";

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

export default async function PositionsTable() {
  let positions: Position[] = [];
  try {
    const allPositions = await fetchJSON("/api/positions/latest");
    // Filter to show only OPEN positions with non-zero amount
    positions = allPositions.filter(
      (p: Position) => p.event_type === "OPEN" && Math.abs(p.amt) > 0
    );
  } catch (e) {
    console.error("Failed to fetch positions:", e);
  }

  const getSideColor = (side: string) => {
    return side?.toUpperCase() === "LONG" ? "text-emerald-600" : "text-red-600";
  };

  const getPnlColor = (pnl: number) => {
    if (pnl > 0) return "text-emerald-600";
    if (pnl < 0) return "text-red-600";
    return "text-slate";
  };

  return (
    <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
      <h3 className="font-display text-lg">Positions</h3>
      <table className="mt-4 w-full text-sm">
        <thead className="text-slate">
          <tr className="text-left">
            <th className="py-2">Symbol</th>
            <th className="py-2">Side</th>
            <th className="py-2">Size</th>
            <th className="py-2">Entry</th>
            <th className="py-2">PnL</th>
            <th className="py-2">Leverage</th>
          </tr>
        </thead>
        <tbody>
          {positions.length === 0 ? (
            <tr>
              <td colSpan={6} className="py-4 text-center text-slate">
                No open positions
              </td>
            </tr>
          ) : (
            positions.map((pos) => (
              <tr key={`${pos.symbol}-${pos.side}`} className="border-t border-ink/5">
                <td className="py-3 font-display">{pos.symbol}</td>
                <td className={`py-3 font-medium ${getSideColor(pos.side)}`}>
                  {pos.side}
                </td>
                <td className="py-3">{Math.abs(pos.amt).toFixed(4)}</td>
                <td className="py-3 text-slate">${pos.entry_price?.toFixed(2)}</td>
                <td className={`py-3 font-medium ${getPnlColor(pos.unrealized_pnl)}`}>
                  {pos.unrealized_pnl >= 0 ? "+" : ""}
                  ${pos.unrealized_pnl?.toFixed(2)}
                </td>
                <td className="py-3 text-slate">{pos.leverage}x</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
