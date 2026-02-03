export interface SignalRow {
  symbol: string;
  decision: string;
  ev: string;
  block: string;
}

export default function TopSignalsTable({ rows }: { rows: SignalRow[] }) {
  return (
    <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
      <h3 className="font-display text-lg">Top Signals</h3>
      <table className="mt-4 w-full text-sm">
        <thead className="text-slate">
          <tr className="text-left">
            <th className="py-2">Symbol</th>
            <th className="py-2">Decision</th>
            <th className="py-2">EV</th>
            <th className="py-2">Block</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.symbol} className="border-t border-ink/5">
              <td className="py-3 font-display">{row.symbol}</td>
              <td className="py-3">{row.decision}</td>
              <td className="py-3">{row.ev}</td>
              <td className="py-3 text-slate">{row.block}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
