interface ExposureRow {
  label: string;
  value: string;
}

export default function ExposureCard({ rows }: { rows: ExposureRow[] }) {
  return (
    <div className="rounded-2xl bg-white/90 p-6 shadow-sm border border-ink/5">
      <h3 className="font-display text-lg">Portfolio Exposure</h3>
      <div className="mt-4 space-y-3">
        {rows.map((row) => (
          <div key={row.label} className="flex items-center justify-between border-b border-slate/10 pb-2 last:border-b-0 last:pb-0">
            <span className="text-sm text-slate">{row.label}</span>
            <span className="text-sm font-display text-ink">{row.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
