export default function Topbar() {
  return (
    <div className="flex items-center justify-between px-8 py-4 border-b border-ink/10">
      <div>
        <div className="text-xl font-display">Realtime Ops</div>
        <div className="text-xs text-slate">Latency, drift, and execution safety</div>
      </div>
      <div className="flex items-center gap-3">
        <span className="px-3 py-1 rounded-full bg-moss/20 text-moss text-xs uppercase tracking-wide">Shadow</span>
        <span className="text-xs text-slate">Last update just now</span>
      </div>
    </div>
  );
}
