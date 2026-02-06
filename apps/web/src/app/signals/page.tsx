"use client";

import { useEffect, useState } from "react";
import { getClientApiBase } from "@/lib/client-api";
import TopSignalsTable, { SignalRow } from "@/components/cards/TopSignalsTable";

type SignalDTO = {
  symbol: string;
  ts: string;
  decision: string;
  ev_long?: number;
  ev_short?: number;
  block_reason_codes: string | string[] | null;
};

type FilterType = "all" | "ev_top" | "blocked";

function parseBlockReasons(value: SignalDTO["block_reason_codes"]) {
  if (!value) return [];
  if (Array.isArray(value)) return value;
  try {
    return JSON.parse(value);
  } catch {
    return [];
  }
}

export default function SignalsPage() {
  const [signals, setSignals] = useState<SignalRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<FilterType>("all");
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [signalTime, setSignalTime] = useState<string | null>(null);

  useEffect(() => {
    const fetchSignals = async () => {
      try {
        const apiBase = getClientApiBase();
        const res = await fetch(`${apiBase}/api/signals/latest?limit=50`);
        if (!res.ok) throw new Error("Failed to fetch");
        const data: SignalDTO[] = await res.json();

        if (data.length > 0 && data[0].ts) {
          setSignalTime(data[0].ts);
        }

        const rows = data.map((signal) => {
          const blockReasons = parseBlockReasons(signal.block_reason_codes);
          const evValue = signal.ev_long ?? signal.ev_short ?? 0;
          const evLabel = `${evValue >= 0 ? "+" : ""}${(evValue * 100).toFixed(2)}%`;
          return {
            symbol: signal.symbol,
            decision: signal.decision,
            ev: evLabel,
            evValue,
            block: blockReasons.length ? blockReasons.join(", ") : "",
          };
        });

        setSignals(rows);
        setLastUpdated(new Date());
      } catch (e) {
        console.error("Failed to fetch signals:", e);
        setSignals([]);
      } finally {
        setLoading(false);
      }
    };

    fetchSignals();
    const interval = setInterval(fetchSignals, 10000);
    return () => clearInterval(interval);
  }, []);

  const filteredSignals = signals.filter((signal) => {
    if (filter === "ev_top") return (signal.evValue ?? 0) > 0.001;
    if (filter === "blocked") return signal.block.length > 0;
    return true;
  });

  const formatSignalTime = (ts: string) => {
    const date = new Date(ts);
    return date.toLocaleString("ko-KR", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-display font-semibold text-foreground">Signals</h1>
          <p className="text-sm text-foreground-muted">
            15분봉 종가 시점에 모델이 예측한 EV와 거래 결정을 표시합니다.
          </p>
          <div className="flex items-center gap-4 mt-1 text-xs text-foreground-muted">
            {signalTime && (
              <span>
                Signal Time: <span className="font-mono">{formatSignalTime(signalTime)}</span>
              </span>
            )}
            {lastUpdated && (
              <span>
                Updated: <span className="font-mono">{lastUpdated.toLocaleTimeString()}</span>
                <span className="ml-1">(10s)</span>
              </span>
            )}
          </div>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setFilter("all")}
            className={`btn btn-sm ${filter === "all" ? "btn-primary" : "btn-secondary"}`}
          >
            All
          </button>
          <button
            onClick={() => setFilter("ev_top")}
            className={`btn btn-sm ${filter === "ev_top" ? "btn-primary" : "btn-secondary"}`}
          >
            EV Top
          </button>
          <button
            onClick={() => setFilter("blocked")}
            className={`btn btn-sm ${filter === "blocked" ? "btn-primary" : "btn-secondary"}`}
          >
            Blocked
          </button>
        </div>
      </div>

      {loading ? (
        <div className="card p-5">
          <div className="space-y-3">
            <div className="skeleton h-10 rounded" />
            <div className="skeleton h-10 rounded" />
            <div className="skeleton h-10 rounded" />
            <div className="skeleton h-10 rounded" />
            <div className="skeleton h-10 rounded" />
          </div>
        </div>
      ) : (
        <TopSignalsTable rows={filteredSignals} />
      )}
    </div>
  );
}
