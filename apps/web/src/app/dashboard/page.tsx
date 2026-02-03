import DriftCard from "@/components/cards/DriftCard";
import ExposureCard from "@/components/cards/ExposureCard";
import MarginCard from "@/components/cards/MarginCard";
import ModeIndicator from "@/components/cards/ModeIndicator";
import RiskBlockersCard from "@/components/cards/RiskBlockersCard";
import SignalFilterSummary from "@/components/cards/SignalFilterSummary";
import SystemHealthCard from "@/components/cards/SystemHealthCard";
import { SignalRow, default as TopSignalsTable } from "@/components/cards/TopSignalsTable";
import { fetchJSON } from "@/lib/api";

type Exposures = {
  open_positions: number;
  total_notional: number;
  long_notional: number;
  short_notional: number;
  daily_pnl: number;
  daily_loss: number;
};

type Status = {
  collector_ok: boolean;
  userstream_ok: boolean;
  latest_candle_ts?: string;
  latest_feature_ts?: string;
  latest_signal_ts?: string;
  exposures: Exposures;
  mode: string;
};

type SignalDTO = {
  symbol: string;
  decision: string;
  ev_long: number | null;
  ev_short: number | null;
  block_reason_codes: string | string[];
};

type DriftDTO = {
  symbol: string;
  psi?: number;
  missing_rate?: number;
  latency_ms?: number;
};

type RiskEventDTO = {
  event_type: string;
  message?: string;
  severity?: number;
  symbol?: string;
};

const formatCurrency = (value: number) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(value);

const formatPercent = (value: number) => `${value >= 0 ? "+" : ""}${(value * 100).toFixed(2)}%`;

async function loadDashboardData() {
  const [status, rawSignals, drift, risk, account] = await Promise.all([
    fetchJSON("/api/status"),
    fetchJSON("/api/signals/latest?limit=50"),
    fetchJSON("/api/data-quality/summary"),
    fetchJSON("/api/risk/state?limit=20"),
    fetchJSON("/api/account/latest").catch(() => null),
  ]);
  return { status, rawSignals, drift, risk, account };
}

export default async function DashboardPage() {
  const { status, rawSignals, drift, risk, account } = await loadDashboardData();

  const healthMetrics = [
    { label: "Collector", value: status.collector_ok ? "OK" : "DOWN" },
    { label: "Userstream", value: status.userstream_ok ? "OK" : "DOWN" },
    {
      label: "Latest candle",
      value: status.latest_candle_ts ? status.latest_candle_ts.replace("T", " ").split(".")[0] : "n/a",
    },
    {
      label: "Latest signal",
      value: status.latest_signal_ts ? status.latest_signal_ts.replace("T", " ").split(".")[0] : "n/a",
    },
  ];

  const parseBlockReasons = (value: SignalDTO["block_reason_codes"]) => {
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
  };

  const signalRows: SignalRow[] = (rawSignals as SignalDTO[]).map((sig) => {
    const blockReasons =
      parseBlockReasons(sig.block_reason_codes);
    const evValue = sig.ev_long ?? sig.ev_short ?? 0;
    const evLabel = `${evValue >= 0 ? "+" : ""}${(evValue * 100).toFixed(2)}%`;
    return {
      symbol: sig.symbol,
      decision: sig.decision,
      ev: evLabel,
      block: blockReasons.length ? blockReasons.join(", ") : "",
    };
  });

  const driftMetrics = (drift as DriftDTO[])
    .slice(0, 3)
    .map((metric) => ({
      symbol: metric.symbol,
      psi: metric.psi?.toFixed(3) ?? "0.000",
      missing: `${((metric.missing_rate ?? 0) * 100).toFixed(2)}%`,
      latency: `${Math.round(metric.latency_ms ?? 0)}ms`,
    }));

  const reasonCounts = (risk as RiskEventDTO[]).reduce<Record<string, number>>((acc, event) => {
    acc[event.event_type] = (acc[event.event_type] ?? 0) + 1;
    return acc;
  }, {});
  const blockerSummary = Object.entries(reasonCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([reason, count]) => ({ reason, count: count.toString() }));

  const exposures = status.exposures;
  const exposureRows = [
    { label: "Open positions", value: exposures.open_positions.toString() },
    { label: "Total notional", value: formatCurrency(exposures.total_notional) },
    { label: "Directional notional", value: `${formatCurrency(exposures.long_notional)} / ${formatCurrency(exposures.short_notional)}` },
    { label: "Daily PnL", value: formatCurrency(exposures.daily_pnl) },
    { label: "Daily loss", value: formatCurrency(exposures.daily_loss) },
  ];

  // Margin data from account API
  const marginData = account ? {
    equity: formatCurrency(account.equity ?? 0),
    used: formatCurrency(account.used_margin ?? 0),
    available: formatCurrency(account.available_margin ?? 0),
    ratio: `${((account.margin_ratio ?? 0) * 100).toFixed(1)}%`,
  } : {
    equity: "-",
    used: "-",
    available: "-",
    ratio: "-",
  };

  return (
    <div className="space-y-8">
      <div className="grid gap-6 lg:grid-cols-4">
        <SystemHealthCard metrics={healthMetrics} />
        <ModeIndicator mode={status.mode} />
        <ExposureCard rows={exposureRows} />
        <RiskBlockersCard blockers={blockerSummary} />
      </div>
      <div className="grid gap-6 lg:grid-cols-4">
        <MarginCard data={marginData} />
        <DriftCard metrics={driftMetrics} />
        <SignalFilterSummary />
        <TopSignalsTable rows={signalRows.slice(0, 6)} />
      </div>
    </div>
  );
}
