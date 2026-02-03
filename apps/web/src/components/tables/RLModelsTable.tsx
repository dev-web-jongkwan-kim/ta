"use client";

import { useEffect, useState } from "react";
import { getClientApiBase } from "@/lib/client-api";

interface RLModel {
  model_id: string;
  symbol: string;
  algorithm: string;
  train_start: string;
  train_end: string;
  metrics: {
    total_reward?: number;
    final_balance?: number;
    total_pnl?: number;
    total_trades?: number;
    winning_trades?: number;
    win_rate?: number;
    profit_factor?: number;
    sharpe_ratio?: number;
    max_drawdown?: number;
    action_distribution?: Record<string, number>;
  } | null;
  model_path: string;
  is_production: boolean;
  created_at: string;
}

export default function RLModelsTable() {
  const [models, setModels] = useState<RLModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const apiBase = getClientApiBase();
        const res = await fetch(`${apiBase}/api/rl/models`);
        const data = await res.json();
        setModels(data);
      } catch (e) {
        console.error("Failed to fetch RL models:", e);
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
    const interval = setInterval(fetchModels, 30000);
    return () => clearInterval(interval);
  }, []);

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString("ko-KR", { year: "numeric", month: "short", day: "numeric" });
  };

  const handlePromote = async (modelId: string) => {
    try {
      const apiBase = getClientApiBase();
      await fetch(`${apiBase}/api/rl/models/${modelId}/promote`, { method: "POST" });
      // Refresh models
      const res = await fetch(`${apiBase}/api/rl/models`);
      setModels(await res.json());
    } catch (e) {
      console.error("Failed to promote model:", e);
    }
  };

  if (loading) {
    return (
      <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
        <h3 className="font-display text-lg">RL Models</h3>
        <div className="mt-4 text-sm text-slate">Loading...</div>
      </div>
    );
  }

  return (
    <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
      <h3 className="font-display text-lg">RL Models</h3>

      <div className="mt-4 space-y-4">
        {models.length === 0 ? (
          <div className="text-center text-slate py-4">No RL models trained yet</div>
        ) : (
          models.map((model) => {
            const isExpanded = expandedId === model.model_id;
            const metrics = model.metrics;

            return (
              <div
                key={model.model_id}
                className={`border rounded-xl overflow-hidden ${
                  model.is_production ? "border-purple-300 bg-purple-50/50" : "border-ink/10"
                }`}
              >
                {/* Header */}
                <div
                  className="p-4 cursor-pointer hover:bg-ink/5"
                  onClick={() => setExpandedId(isExpanded ? null : model.model_id)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <code className="text-xs bg-ink/5 px-2 py-1 rounded">
                        {model.model_id}
                      </code>
                      <span className="px-2 py-1 rounded-full text-xs bg-blue-100 text-blue-700">
                        {model.symbol}
                      </span>
                      {model.is_production && (
                        <span className="px-2 py-1 rounded-full text-xs bg-purple-100 text-purple-700">
                          PRODUCTION
                        </span>
                      )}
                      <span className="text-xs text-slate">{model.algorithm}</span>
                    </div>
                    <div className="flex items-center gap-4 text-sm">
                      <span className="text-slate">PF: </span>
                      <span className={`font-medium ${
                        (metrics?.profit_factor ?? 0) >= 1 ? "text-emerald-600" : "text-rose-500"
                      }`}>
                        {metrics?.profit_factor?.toFixed(2) ?? "-"}
                      </span>
                      <span className="text-slate">Win: </span>
                      <span className="font-medium">
                        {metrics?.win_rate ? `${(metrics.win_rate * 100).toFixed(1)}%` : "-"}
                      </span>
                      <span className="text-xl text-slate">{isExpanded ? "−" : "+"}</span>
                    </div>
                  </div>

                  {/* Training Info Summary */}
                  <div className="mt-2 flex gap-4 text-xs text-slate">
                    <span>Train: {formatDate(model.train_start)} ~ {formatDate(model.train_end)}</span>
                    <span>Trades: {metrics?.total_trades?.toLocaleString() ?? "-"}</span>
                  </div>
                </div>

                {/* Expanded Details */}
                {isExpanded && (
                  <div className="border-t border-ink/10 p-4 bg-white/50 space-y-4">
                    {/* Performance Metrics */}
                    <div>
                      <div className="text-xs font-medium text-slate uppercase mb-2">Performance Metrics</div>
                      <div className="grid grid-cols-5 gap-3 text-sm">
                        <div className="bg-mist rounded-lg p-3 text-center">
                          <div className="text-xs text-slate">Profit Factor</div>
                          <div className={`font-display text-lg ${
                            (metrics?.profit_factor ?? 0) >= 1 ? "text-emerald-600" : "text-rose-500"
                          }`}>
                            {metrics?.profit_factor?.toFixed(2) ?? "-"}
                          </div>
                        </div>
                        <div className="bg-mist rounded-lg p-3 text-center">
                          <div className="text-xs text-slate">Win Rate</div>
                          <div className="font-display text-lg">
                            {metrics?.win_rate ? `${(metrics.win_rate * 100).toFixed(1)}%` : "-"}
                          </div>
                        </div>
                        <div className="bg-mist rounded-lg p-3 text-center">
                          <div className="text-xs text-slate">Sharpe Ratio</div>
                          <div className={`font-display text-lg ${
                            (metrics?.sharpe_ratio ?? 0) >= 1 ? "text-emerald-600" : "text-rose-500"
                          }`}>
                            {metrics?.sharpe_ratio?.toFixed(2) ?? "-"}
                          </div>
                        </div>
                        <div className="bg-mist rounded-lg p-3 text-center">
                          <div className="text-xs text-slate">Max Drawdown</div>
                          <div className="font-display text-lg text-rose-500">
                            {metrics?.max_drawdown ? `${(metrics.max_drawdown * 100).toFixed(1)}%` : "-"}
                          </div>
                        </div>
                        <div className="bg-mist rounded-lg p-3 text-center">
                          <div className="text-xs text-slate">Total PnL</div>
                          <div className={`font-display text-lg ${
                            (metrics?.total_pnl ?? 0) >= 0 ? "text-emerald-600" : "text-rose-500"
                          }`}>
                            ${metrics?.total_pnl?.toFixed(2) ?? "-"}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Action Distribution */}
                    {metrics?.action_distribution && (
                      <div>
                        <div className="text-xs font-medium text-slate uppercase mb-2">Action Distribution</div>
                        <div className="grid grid-cols-4 gap-3">
                          {Object.entries(metrics.action_distribution).map(([action, count]) => (
                            <div key={action} className={`rounded-lg p-3 text-center ${
                              action === "HOLD" ? "bg-gray-100" :
                              action === "LONG" ? "bg-emerald-50" :
                              action === "SHORT" ? "bg-rose-50" :
                              "bg-amber-50"
                            }`}>
                              <div className="text-xs text-slate">{action}</div>
                              <div className="font-display text-lg">{count.toLocaleString()}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Actions */}
                    <div className="flex justify-end pt-2 border-t border-ink/10">
                      {!model.is_production && (
                        <button
                          onClick={() => handlePromote(model.model_id)}
                          className="px-4 py-2 bg-purple-600 text-white rounded-lg text-sm hover:bg-purple-700"
                        >
                          Promote to Production
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
