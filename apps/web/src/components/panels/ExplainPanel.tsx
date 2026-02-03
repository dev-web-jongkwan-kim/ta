"use client";

import { useEffect, useState } from "react";
import { getClientApiBase } from "@/lib/client-api";
import { InfoTooltip, TERM_DEFINITIONS } from "@/components/ui/Tooltip";

interface Signal {
  symbol: string;
  ts: string;
  er_long: number;
  er_short: number;
  decision: string;
  explain: Record<string, number> | null;
}

export default function ExplainPanel() {
  const [signal, setSignal] = useState<Signal | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const apiBase = getClientApiBase();
        const res = await fetch(`${apiBase}/api/signals/latest?limit=1`);
        if (!res.ok) throw new Error("Failed to fetch");
        const data: Signal[] = await res.json();
        if (data.length > 0) {
          setSignal(data[0]);
        }
      } catch (e) {
        console.error("Failed to load signal:", e);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  // explain 필드가 JSON인 경우 파싱
  const getExplainData = (): Record<string, number> | null => {
    if (!signal?.explain) return null;
    if (typeof signal.explain === "string") {
      try {
        return JSON.parse(signal.explain);
      } catch {
        return null;
      }
    }
    return signal.explain;
  };

  const explainData = getExplainData();
  const sortedFeatures = explainData
    ? Object.entries(explainData)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .slice(0, 5)
    : [];

  const getFeatureColor = (value: number) => {
    if (value > 0) return "text-emerald-600";
    if (value < 0) return "text-rose-500";
    return "text-slate";
  };

  return (
    <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
      <h3 className="font-display text-lg">Explainability</h3>
      <p className="text-xs text-slate mt-1">모델 예측에 영향을 준 주요 피처</p>

      {loading ? (
        <div className="mt-4 text-sm text-slate">Loading...</div>
      ) : !signal ? (
        <div className="mt-4 text-sm text-slate">
          No signals available yet.
          <div className="text-xs mt-1">
            시스템이 실행되면 신호가 여기에 표시됩니다.
          </div>
        </div>
      ) : (
        <div className="mt-4 space-y-3">
          {/* 신호 요약 */}
          <div className="flex items-center justify-between text-sm pb-2 border-b border-ink/10">
            <span className="font-display">{signal.symbol}</span>
            <span
              className={`px-2 py-0.5 rounded text-xs ${
                signal.decision === "LONG"
                  ? "bg-emerald-100 text-emerald-700"
                  : signal.decision === "SHORT"
                  ? "bg-rose-100 text-rose-700"
                  : "bg-slate/10 text-slate"
              }`}
            >
              {signal.decision || "HOLD"}
            </span>
          </div>

          {/* 피처 기여도 */}
          {sortedFeatures.length > 0 ? (
            <ul className="space-y-2 text-sm">
              {sortedFeatures.map(([feature, value]) => (
                <li key={feature} className="flex items-center justify-between">
                  <span className="text-slate flex items-center">
                    {feature}
                    {TERM_DEFINITIONS[feature] && <InfoTooltip term={feature} />}
                  </span>
                  <span className={`font-mono ${getFeatureColor(value)}`}>
                    {value > 0 ? "+" : ""}
                    {value.toFixed(3)}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <div className="text-sm text-slate">
              No feature contributions available.
            </div>
          )}

          {/* 예측값 */}
          <div className="pt-2 border-t border-ink/10 grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-slate">ER Long</span>
              <InfoTooltip term="er_long" />
              <div className={`font-mono ${getFeatureColor(signal.er_long || 0)}`}>
                {signal.er_long ? `${(signal.er_long * 100).toFixed(3)}%` : "-"}
              </div>
            </div>
            <div>
              <span className="text-slate">ER Short</span>
              <InfoTooltip term="er_short" />
              <div className={`font-mono ${getFeatureColor(-(signal.er_short || 0))}`}>
                {signal.er_short ? `${(signal.er_short * 100).toFixed(3)}%` : "-"}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
