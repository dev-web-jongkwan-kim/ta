"use client";

import { useEffect, useState } from "react";
import { getClientApiBase } from "@/lib/client-api";

interface RLDecision {
  id: number;
  ts: string;
  symbol: string;
  model_id: string;
  action: number;
  action_name: string;
  confidence: number;
  action_probs: number[] | null;
  value_estimate: number | null;
  model_predictions: {
    er_long?: number;
    q05_long?: number;
    e_mae_long?: number;
    e_hold_long?: number;
  } | null;
  position_before: number;
  position_after: number;
  executed: boolean;
  pnl_result: number | null;
}

interface DecisionStats {
  period_hours: number;
  symbol: string | null;
  total_decisions: number;
  total_executed: number;
  by_action: Record<string, {
    count: number;
    avg_confidence: number;
    executed_count: number;
    avg_pnl: number | null;
  }>;
}

export default function RLDecisionsTable() {
  const [decisions, setDecisions] = useState<RLDecision[]>([]);
  const [stats, setStats] = useState<DecisionStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedSymbol, setSelectedSymbol] = useState<string>("");
  const [symbols, setSymbols] = useState<string[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const apiBase = getClientApiBase();

        // Fetch decisions
        const url = selectedSymbol
          ? `${apiBase}/api/rl/decisions?symbol=${selectedSymbol}&limit=50`
          : `${apiBase}/api/rl/decisions?limit=50`;
        const res = await fetch(url);
        const data = await res.json();
        setDecisions(data);

        // Extract unique symbols
        const uniqueSymbols = [...new Set(data.map((d: RLDecision) => d.symbol))];
        setSymbols(uniqueSymbols as string[]);

        // Fetch stats
        const statsUrl = selectedSymbol
          ? `${apiBase}/api/rl/decisions/stats?symbol=${selectedSymbol}`
          : `${apiBase}/api/rl/decisions/stats`;
        const statsRes = await fetch(statsUrl);
        setStats(await statsRes.json());

      } catch (e) {
        console.error("Failed to fetch RL decisions:", e);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [selectedSymbol]);

  const formatTime = (ts: string) => {
    const date = new Date(ts);
    return date.toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  };

  const getActionColor = (action: string) => {
    switch (action) {
      case "LONG": return "text-emerald-600 bg-emerald-50";
      case "SHORT": return "text-rose-600 bg-rose-50";
      case "CLOSE": return "text-amber-600 bg-amber-50";
      default: return "text-gray-600 bg-gray-50";
    }
  };

  if (loading) {
    return (
      <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
        <h3 className="font-display text-lg">RL Agent Decisions</h3>
        <div className="mt-4 text-sm text-slate">Loading...</div>
      </div>
    );
  }

  return (
    <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
      <div className="flex items-center justify-between">
        <h3 className="font-display text-lg">RL Agent Decisions</h3>
        <select
          value={selectedSymbol}
          onChange={(e) => setSelectedSymbol(e.target.value)}
          className="text-sm border rounded-lg px-3 py-1"
        >
          <option value="">All Symbols</option>
          {symbols.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>

      {/* Stats Summary */}
      {stats && (
        <div className="mt-4 grid grid-cols-5 gap-3">
          <div className="bg-mist rounded-lg p-3 text-center">
            <div className="text-xs text-slate">Total (24h)</div>
            <div className="font-display text-xl">{stats.total_decisions}</div>
          </div>
          <div className="bg-mist rounded-lg p-3 text-center">
            <div className="text-xs text-slate">Executed</div>
            <div className="font-display text-xl text-purple-600">{stats.total_executed}</div>
          </div>
          {["LONG", "SHORT", "HOLD"].map((action) => (
            <div key={action} className={`rounded-lg p-3 text-center ${
              action === "LONG" ? "bg-emerald-50" :
              action === "SHORT" ? "bg-rose-50" :
              "bg-gray-50"
            }`}>
              <div className="text-xs text-slate">{action}</div>
              <div className="font-display text-xl">
                {stats.by_action[action]?.count ?? 0}
              </div>
              <div className="text-xs text-slate">
                {stats.by_action[action]?.avg_confidence
                  ? `${(stats.by_action[action].avg_confidence * 100).toFixed(0)}% conf`
                  : "-"}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Decisions Table */}
      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs text-slate border-b">
              <th className="py-2 pr-2">Time</th>
              <th className="py-2 pr-2">Symbol</th>
              <th className="py-2 pr-2">Action</th>
              <th className="py-2 pr-2">Confidence</th>
              <th className="py-2 pr-2">Position</th>
              <th className="py-2 pr-2">Model Input</th>
              <th className="py-2 pr-2">Executed</th>
              <th className="py-2">PnL</th>
            </tr>
          </thead>
          <tbody>
            {decisions.length === 0 ? (
              <tr>
                <td colSpan={8} className="py-8 text-center text-slate">
                  No RL decisions recorded yet
                </td>
              </tr>
            ) : (
              decisions.map((d) => (
                <tr key={d.id} className="border-b border-ink/5 hover:bg-ink/5">
                  <td className="py-2 pr-2 font-mono text-xs">{formatTime(d.ts)}</td>
                  <td className="py-2 pr-2">{d.symbol}</td>
                  <td className="py-2 pr-2">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getActionColor(d.action_name)}`}>
                      {d.action_name}
                    </span>
                  </td>
                  <td className="py-2 pr-2">
                    <div className="flex items-center gap-1">
                      <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-purple-500"
                          style={{ width: `${(d.confidence ?? 0) * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-slate">
                        {((d.confidence ?? 0) * 100).toFixed(0)}%
                      </span>
                    </div>
                  </td>
                  <td className="py-2 pr-2 text-xs">
                    {d.position_before.toFixed(2)} → {d.position_after.toFixed(2)}
                  </td>
                  <td className="py-2 pr-2">
                    {d.model_predictions && (
                      <div className="text-xs space-x-2">
                        <span className={d.model_predictions.er_long && d.model_predictions.er_long > 0 ? "text-emerald-600" : "text-rose-500"}>
                          ER:{(d.model_predictions.er_long ?? 0).toFixed(4)}
                        </span>
                        <span className="text-slate">
                          Q05:{(d.model_predictions.q05_long ?? 0).toFixed(4)}
                        </span>
                      </div>
                    )}
                  </td>
                  <td className="py-2 pr-2">
                    {d.executed ? (
                      <span className="text-emerald-600">✓</span>
                    ) : (
                      <span className="text-slate">-</span>
                    )}
                  </td>
                  <td className="py-2">
                    {d.pnl_result !== null ? (
                      <span className={d.pnl_result >= 0 ? "text-emerald-600" : "text-rose-500"}>
                        ${d.pnl_result.toFixed(2)}
                      </span>
                    ) : (
                      <span className="text-slate">-</span>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Action Probability Visualization */}
      {decisions.length > 0 && decisions[0].action_probs && (
        <div className="mt-4 pt-4 border-t border-ink/10">
          <div className="text-xs font-medium text-slate uppercase mb-2">
            Latest Action Probabilities
          </div>
          <div className="flex gap-2">
            {["HOLD", "LONG", "SHORT", "CLOSE"].map((action, idx) => {
              const prob = decisions[0].action_probs?.[idx] ?? 0;
              return (
                <div key={action} className="flex-1">
                  <div className="flex justify-between text-xs mb-1">
                    <span>{action}</span>
                    <span>{(prob * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${
                        action === "LONG" ? "bg-emerald-500" :
                        action === "SHORT" ? "bg-rose-500" :
                        action === "CLOSE" ? "bg-amber-500" :
                        "bg-gray-400"
                      }`}
                      style={{ width: `${prob * 100}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
