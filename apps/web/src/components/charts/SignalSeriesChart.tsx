"use client";

import { useEffect, useState } from "react";
import { getClientApiBase } from "@/lib/client-api";

interface Signal {
  ts: string;
  ev_long: number | null;
  ev_short: number | null;
  decision: string;
}

interface SignalSeriesChartProps {
  symbol?: string;
}

export default function SignalSeriesChart({ symbol }: SignalSeriesChartProps) {
  const [signals, setSignals] = useState<Signal[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const apiBase = getClientApiBase();
        const url = symbol
          ? `${apiBase}/api/symbol/${symbol}/series`
          : `${apiBase}/api/signals/latest?limit=100`;

        const res = await fetch(url);
        if (!res.ok) throw new Error("Failed to fetch");
        const data = await res.json();

        const signalList = symbol ? data.signals : data;
        if (signalList && signalList.length > 0) {
          const formatted = signalList
            .map((s: any) => ({
              ts: Array.isArray(s) ? s[1] : s.ts,
              ev_long: Array.isArray(s) ? s[3] : s.ev_long,
              ev_short: Array.isArray(s) ? s[4] : s.ev_short,
              decision: Array.isArray(s) ? s[13] : s.decision,
            }))
            .slice(0, 50)
            .reverse();
          setSignals(formatted);
        }
      } catch (e) {
        console.error("Failed to load signals:", e);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [symbol]);

  const evValues = signals.map((s) => Math.max(s.ev_long || 0, s.ev_short || 0));
  const minEV = evValues.length > 0 ? Math.min(...evValues, 0) : -0.01;
  const maxEV = evValues.length > 0 ? Math.max(...evValues, 0) : 0.01;
  const evRange = maxEV - minEV || 0.01;

  const getY = (ev: number) => {
    return 120 - ((ev - minEV) / evRange) * 100;
  };

  const zeroY = getY(0);

  return (
    <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
      <h3 className="font-display text-lg">Signal Series (EV)</h3>
      <div className="mt-4 h-40 rounded-xl bg-mist overflow-hidden">
        {loading ? (
          <div className="h-full flex items-center justify-center text-slate text-sm">
            Loading...
          </div>
        ) : signals.length === 0 ? (
          <div className="h-full flex items-center justify-center text-slate text-sm">
            No signals available
          </div>
        ) : (
          <svg width="100%" height="100%" viewBox="0 0 600 140" preserveAspectRatio="none">
            {/* Zero line */}
            <line
              x1="0"
              y1={zeroY}
              x2="600"
              y2={zeroY}
              stroke="#94a3b8"
              strokeWidth={1}
              strokeDasharray="4,4"
            />
            {/* EV bars */}
            {signals.map((signal, i) => {
              const x = (i / signals.length) * 580 + 10;
              const width = Math.max(4, 580 / signals.length - 2);
              const ev = Math.max(signal.ev_long || 0, signal.ev_short || 0);
              const isPositive = ev >= 0;
              const barY = isPositive ? getY(ev) : zeroY;
              const barHeight = Math.abs(getY(ev) - zeroY);

              let color = "#94a3b8";
              if (signal.decision === "LONG") color = "#10b981";
              else if (signal.decision === "SHORT") color = "#ef4444";

              return (
                <rect
                  key={i}
                  x={x}
                  y={barY}
                  width={width}
                  height={Math.max(1, barHeight)}
                  fill={color}
                  opacity={0.8}
                />
              );
            })}
          </svg>
        )}
      </div>
      <div className="mt-2 flex justify-between text-xs text-slate">
        <span>Min: {(minEV * 100).toFixed(3)}%</span>
        <span>Max: {(maxEV * 100).toFixed(3)}%</span>
      </div>
    </div>
  );
}
