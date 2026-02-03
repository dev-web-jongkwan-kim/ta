import TopSignalsTable, { SignalRow } from "@/components/cards/TopSignalsTable";
import { fetchJSON } from "@/lib/api";

type SignalDTO = {
  symbol: string;
  decision: string;
  ev_long?: number;
  ev_short?: number;
  block_reason_codes: string | string[] | null;
};

function parseBlockReasons(value: SignalDTO["block_reason_codes"]) {
  if (!value) {
    return [];
  }
  if (Array.isArray(value)) {
    return value;
  }
  try {
    return JSON.parse(value);
  } catch {
    return [];
  }
}

async function loadSignals(): Promise<SignalRow[]> {
  const data = await fetchJSON("/api/signals/latest?limit=50");
  return (data as SignalDTO[]).map((signal) => {
    const blockReasons = parseBlockReasons(signal.block_reason_codes);
    const evValue = signal.ev_long ?? signal.ev_short ?? 0;
    const evLabel = `${evValue >= 0 ? "+" : ""}${(evValue * 100).toFixed(2)}%`;
    return {
      symbol: signal.symbol,
      decision: signal.decision,
      ev: evLabel,
      block: blockReasons.length ? blockReasons.join(", ") : "",
    };
  });
}

export default async function SignalsPage() {
  const rows = await loadSignals();
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-display">Signals</h1>
          <p className="text-sm text-slate">Filter by EV, funding extremes, or blocked actions.</p>
        </div>
        <div className="flex gap-2">
          <button className="px-3 py-1 rounded-full border border-ink/10 text-xs">EV Top</button>
          <button className="px-3 py-1 rounded-full border border-ink/10 text-xs">Blocked</button>
        </div>
      </div>
      <TopSignalsTable rows={rows} />
    </div>
  );
}
