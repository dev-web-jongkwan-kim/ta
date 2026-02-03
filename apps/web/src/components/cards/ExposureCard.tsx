interface ExposureRow {
  label: string;
  value: string;
  highlight?: "positive" | "negative";
}

export default function ExposureCard({ rows }: { rows: ExposureRow[] }) {
  return (
    <div className="card p-5">
      <h3 className="font-display font-semibold text-foreground mb-4">Portfolio Exposure</h3>
      <div className="space-y-2.5">
        {rows.map((row) => (
          <div
            key={row.label}
            className="flex items-center justify-between py-1.5 border-b border-border last:border-b-0"
          >
            <span className="text-sm text-foreground-muted">{row.label}</span>
            <span className={`font-mono text-sm font-medium ${
              row.highlight === "positive" ? "metric-positive" :
              row.highlight === "negative" ? "metric-negative" :
              "text-foreground"
            }`}>
              {row.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
