export default function SettingsPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-display">Settings</h1>
      <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
        <h3 className="font-display text-lg">Mode</h3>
        <div className="mt-4 flex gap-2">
          <button className="px-4 py-2 rounded-full bg-moss/20 text-moss text-xs uppercase">Shadow</button>
          <button className="px-4 py-2 rounded-full border border-ink/10 text-xs uppercase">Live</button>
          <button className="px-4 py-2 rounded-full border border-ink/10 text-xs uppercase">Off</button>
        </div>
      </div>
    </div>
  );
}
