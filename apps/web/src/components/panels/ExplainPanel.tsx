export default function ExplainPanel() {
  return (
    <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
      <h3 className="font-display text-lg">Explainability</h3>
      <ul className="mt-4 space-y-2 text-sm text-slate">
        <li>ema_dist_atr: 0.23</li>
        <li>funding_z: -0.18</li>
        <li>vol_z: 0.15</li>
      </ul>
    </div>
  );
}
