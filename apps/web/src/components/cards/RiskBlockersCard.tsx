interface Blocker {
  reason: string;
  count: string;
}

export default function RiskBlockersCard({ blockers }: { blockers: Blocker[] }) {
  return (
    <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
      <h3 className="font-display text-lg">Risk Blockers</h3>
      <div className="mt-4 space-y-3">
        {blockers.map((blocker) => (
          <div key={blocker.reason} className="flex items-center justify-between">
            <span className="text-sm text-slate">{blocker.reason}</span>
            <span className="text-sm font-display text-ember">{blocker.count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
