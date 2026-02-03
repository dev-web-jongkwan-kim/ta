"use client";

import { useEffect, useState } from "react";
import { getClientApiBase } from "@/lib/client-api";
import DriftCard from "@/components/cards/DriftCard";
import ExposureCard from "@/components/cards/ExposureCard";
import MarginCard from "@/components/cards/MarginCard";
import ModeIndicator from "@/components/cards/ModeIndicator";
import RiskBlockersCard from "@/components/cards/RiskBlockersCard";
import SignalFilterSummary from "@/components/cards/SignalFilterSummary";
import SystemHealthCard from "@/components/cards/SystemHealthCard";
import WebSocketStatusCard from "@/components/cards/WebSocketStatusCard";
import { SignalRow, default as TopSignalsTable } from "@/components/cards/TopSignalsTable";
import TradingControl from "@/components/trading/TradingControl";

interface DashboardData {
  status: any;
  signals: any[];
  drift: any[];
  risk: any[];
  account: any;
}

const formatCurrency = (value: number) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(value);

export default function DashboardPage() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const apiBase = getClientApiBase();
        const [statusRes, signalsRes, driftRes, riskRes, accountRes] = await Promise.all([
          fetch(`${apiBase}/api/status`).then(r => r.ok ? r.json() : null).catch(() => null),
          fetch(`${apiBase}/api/signals/latest?limit=50`).then(r => r.ok ? r.json() : []).catch(() => []),
          fetch(`${apiBase}/api/data-quality/summary`).then(r => r.ok ? r.json() : []).catch(() => []),
          fetch(`${apiBase}/api/risk/state?limit=20`).then(r => r.ok ? r.json() : []).catch(() => []),
          fetch(`${apiBase}/api/account/latest`).then(r => r.ok ? r.json() : null).catch(() => null),
        ]);

        setData({
          status: statusRes || {
            collector_ok: false,
            userstream_ok: false,
            exposures: { open_positions: 0, total_notional: 0, long_notional: 0, short_notional: 0, daily_pnl: 0, daily_loss: 0 },
            mode: "shadow"
          },
          signals: signalsRes || [],
          drift: driftRes || [],
          risk: riskRes || [],
          account: accountRes,
        });
      } catch (e) {
        console.error("Failed to load dashboard:", e);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="card p-5">
              <div className="skeleton h-6 w-32 rounded mb-4" />
              <div className="skeleton h-20 rounded" />
            </div>
          ))}
        </div>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="card p-5">
              <div className="skeleton h-6 w-32 rounded mb-4" />
              <div className="skeleton h-20 rounded" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  const status = data!.status;
  const rawSignals = data!.signals;
  const drift = data!.drift;
  const risk = data!.risk;
  const account = data!.account;

  const healthMetrics = [
    { label: "Collector", value: status.collector_ok ? "OK" : "DOWN", ok: status.collector_ok },
    { label: "Userstream", value: status.userstream_ok ? "OK" : "DOWN", ok: status.userstream_ok },
    {
      label: "Latest candle",
      value: status.latest_candle_ts ? status.latest_candle_ts.replace("T", " ").split(".")[0] : "n/a",
    },
    {
      label: "Latest signal",
      value: status.latest_signal_ts ? status.latest_signal_ts.replace("T", " ").split(".")[0] : "n/a",
    },
  ];

  const parseBlockReasons = (value: any) => {
    if (!value) return [];
    if (Array.isArray(value)) return value;
    try {
      return JSON.parse(value);
    } catch {
      return [];
    }
  };

  const signalRows: SignalRow[] = rawSignals.map((sig: any) => {
    const blockReasons = parseBlockReasons(sig.block_reason_codes);
    const evValue = sig.ev_long ?? sig.ev_short ?? 0;
    const evLabel = `${evValue >= 0 ? "+" : ""}${(evValue * 100).toFixed(2)}%`;
    return {
      symbol: sig.symbol,
      decision: sig.decision,
      ev: evLabel,
      block: blockReasons.length ? blockReasons.join(", ") : "",
    };
  });

  const driftMetrics = drift.slice(0, 3).map((metric: any) => ({
    symbol: metric.symbol,
    psi: metric.psi?.toFixed(3) ?? "0.000",
    missing: `${((metric.missing_rate ?? 0) * 100).toFixed(2)}%`,
    latency: `${Math.round(metric.latency_ms ?? 0)}ms`,
  }));

  const reasonCounts = risk.reduce((acc: Record<string, number>, event: any) => {
    acc[event.event_type] = (acc[event.event_type] ?? 0) + 1;
    return acc;
  }, {});
  const blockerSummary = Object.entries(reasonCounts)
    .sort((a, b) => (b[1] as number) - (a[1] as number))
    .slice(0, 4)
    .map(([reason, count]) => ({ reason, count: count as number }));

  const exposures = status.exposures;
  const exposureRows: Array<{ label: string; value: string; highlight?: "positive" | "negative" }> = [
    { label: "Open positions", value: exposures.open_positions.toString() },
    { label: "Total notional", value: formatCurrency(exposures.total_notional) },
    { label: "Directional", value: `${formatCurrency(exposures.long_notional)} / ${formatCurrency(exposures.short_notional)}` },
    { label: "Daily PnL", value: formatCurrency(exposures.daily_pnl), highlight: exposures.daily_pnl >= 0 ? "positive" : "negative" },
  ];

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
    <div className="space-y-6">
      {/* Trading Control */}
      <TradingControl compact />

      {/* Top Row - Key Metrics */}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <SystemHealthCard metrics={healthMetrics} />
        <ModeIndicator mode={status.mode} />
        <ExposureCard rows={exposureRows} />
        <RiskBlockersCard blockers={blockerSummary} />
      </div>

      {/* Second Row - Details */}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MarginCard data={marginData} />
        <WebSocketStatusCard />
        <DriftCard metrics={driftMetrics} />
        <SignalFilterSummary />
      </div>

      {/* Third Row - Signals */}
      <div className="grid gap-4">
        <TopSignalsTable rows={signalRows.slice(0, 6)} />
      </div>
    </div>
  );
}
