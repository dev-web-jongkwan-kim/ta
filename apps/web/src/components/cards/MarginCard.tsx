interface MarginData {
  equity: string;
  used: string;
  available: string;
  ratio: string;
}

export default function MarginCard({ data }: { data: MarginData }) {
  return (
    <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
      <h3 className="font-display text-lg">Margin Overview</h3>
      <div className="mt-4 grid grid-cols-2 gap-4">
        <div>
          <div className="text-xs text-slate uppercase">Equity</div>
          <div className="text-2xl font-display">{data.equity}</div>
        </div>
        <div>
          <div className="text-xs text-slate uppercase">Used</div>
          <div className="text-2xl font-display">{data.used}</div>
        </div>
        <div>
          <div className="text-xs text-slate uppercase">Available</div>
          <div className="text-2xl font-display">{data.available}</div>
        </div>
        <div>
          <div className="text-xs text-slate uppercase">Margin Ratio</div>
          <div className="text-2xl font-display text-ember">{data.ratio}</div>
        </div>
      </div>
    </div>
  );
}
