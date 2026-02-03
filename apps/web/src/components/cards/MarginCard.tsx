interface MarginData {
  equity: string;
  used: string;
  available: string;
  ratio: string;
}

export default function MarginCard({ data }: { data: MarginData }) {
  return (
    <div className="card p-5">
      <h3 className="font-display font-semibold text-foreground mb-4">Margin Overview</h3>
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-background-tertiary/50 rounded-lg p-3">
          <div className="metric-label mb-1">Equity</div>
          <div className="font-mono text-lg font-semibold text-foreground">{data.equity}</div>
        </div>
        <div className="bg-background-tertiary/50 rounded-lg p-3">
          <div className="metric-label mb-1">Used</div>
          <div className="font-mono text-lg font-semibold text-foreground">{data.used}</div>
        </div>
        <div className="bg-background-tertiary/50 rounded-lg p-3">
          <div className="metric-label mb-1">Available</div>
          <div className="font-mono text-lg font-semibold text-success">{data.available}</div>
        </div>
        <div className="bg-background-tertiary/50 rounded-lg p-3">
          <div className="metric-label mb-1">Margin Ratio</div>
          <div className="font-mono text-lg font-semibold text-warning">{data.ratio}</div>
        </div>
      </div>
    </div>
  );
}
