export interface SignalRow {
  symbol: string;
  decision: string;
  ev: string;
  evValue?: number;
  block: string;
}

const getDecisionBadge = (decision: string) => {
  switch (decision?.toUpperCase()) {
    case "LONG":
      return "badge-success";
    case "SHORT":
      return "badge-danger";
    default:
      return "badge-neutral";
  }
};

export default function TopSignalsTable({ rows }: { rows: SignalRow[] }) {
  return (
    <div className="card p-5">
      <h3 className="font-display font-semibold text-foreground mb-4">Top Signals</h3>

      {rows.length === 0 ? (
        <div className="text-center text-foreground-muted text-sm py-4">
          No signals available
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left border-b border-border">
                <th className="table-header pb-2">Symbol</th>
                <th className="table-header pb-2">Signal</th>
                <th className="table-header pb-2 text-right">EV</th>
                <th className="table-header pb-2">Block</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.symbol} className="table-row">
                  <td className="py-2.5 font-mono font-medium text-foreground">{row.symbol}</td>
                  <td className="py-2.5">
                    <span className={`badge ${getDecisionBadge(row.decision)}`}>
                      {row.decision || "HOLD"}
                    </span>
                  </td>
                  <td className={`py-2.5 text-right font-mono ${
                    row.ev.startsWith("+") ? "text-success" :
                    row.ev.startsWith("-") ? "text-danger" :
                    "text-foreground-secondary"
                  }`}>
                    {row.ev}
                  </td>
                  <td className="py-2.5 text-foreground-muted text-xs truncate max-w-[100px]">
                    {row.block || "-"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
