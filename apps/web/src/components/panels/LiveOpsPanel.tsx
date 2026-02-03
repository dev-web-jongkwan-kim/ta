export default function LiveOpsPanel() {
  return (
    <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
      <h3 className="font-display text-lg">Live Ops</h3>
      <div className="mt-4 text-sm text-slate">
        <div>Position: FLAT</div>
        <div>Open Orders: 0</div>
        <div>Shadow PnL: 0.0 USDT</div>
      </div>
    </div>
  );
}
