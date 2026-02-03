"use client";

import { useEffect, useState } from "react";
import { getClientApiBase } from "@/lib/client-api";

interface ModelMetrics {
  trade?: {
    profit_factor?: number;
    expectancy?: number;
    win_rate?: number;
    n_trades?: number;
    max_drawdown?: number;
  };
  regime?: Record<string, {
    profit_factor?: number;
    win_rate?: number;
    n_trades?: number;
  }>;
  er_long?: { rmse?: number; mae?: number };
  q05_long?: { rmse?: number; mae?: number };
  e_mae_long?: { rmse?: number; mae?: number };
  cost_breakdown?: {
    fee_per_trade?: number;
    slippage_per_trade?: number;
  };
}

interface Model {
  model_id: string;
  algo: string;
  is_production: boolean;
  train_start: string;
  train_end: string;
  feature_schema_version: number;
  label_spec_hash: string;
  metrics: ModelMetrics | null;
  created_at: string;
}

export default function ModelsDetailTable() {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const apiBase = getClientApiBase();
        const res = await fetch(`${apiBase}/api/models`);
        const data = await res.json();
        setModels(data);
      } catch (e) {
        console.error("Failed to fetch models:", e);
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, []);

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString("ko-KR", { year: "numeric", month: "short", day: "numeric" });
  };

  const calculateTrainDays = (start: string, end: string) => {
    const startDate = new Date(start);
    const endDate = new Date(end);
    const diffTime = Math.abs(endDate.getTime() - startDate.getTime());
    return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  };

  if (loading) {
    return (
      <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
        <h3 className="font-display text-lg">Models</h3>
        <div className="mt-4 text-sm text-slate">Loading...</div>
      </div>
    );
  }

  return (
    <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
      <h3 className="font-display text-lg">Models</h3>

      <div className="mt-4 space-y-4">
        {models.length === 0 ? (
          <div className="text-center text-slate py-4">No models found</div>
        ) : (
          models.map((model) => {
            const isExpanded = expandedId === model.model_id;
            const trainDays = calculateTrainDays(model.train_start, model.train_end);
            const metrics = model.metrics;

            return (
              <div
                key={model.model_id}
                className={`border rounded-xl overflow-hidden ${
                  model.is_production ? "border-emerald-300 bg-emerald-50/50" : "border-ink/10"
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
                        {model.model_id.slice(0, 8)}
                      </code>
                      {model.is_production && (
                        <span className="px-2 py-1 rounded-full text-xs bg-emerald-100 text-emerald-700">
                          PRODUCTION
                        </span>
                      )}
                      <span className="text-xs text-slate">{model.algo.toUpperCase()}</span>
                    </div>
                    <div className="flex items-center gap-4 text-sm">
                      <span className="text-slate">PF: </span>
                      <span className={`font-medium ${
                        (metrics?.trade?.profit_factor ?? 0) >= 1 ? "text-emerald-600" : "text-rose-500"
                      }`}>
                        {metrics?.trade?.profit_factor?.toFixed(2) ?? "-"}
                      </span>
                      <span className="text-slate">Win: </span>
                      <span className="font-medium">
                        {metrics?.trade?.win_rate ? `${(metrics.trade.win_rate * 100).toFixed(1)}%` : "-"}
                      </span>
                      <span className="text-xl text-slate">{isExpanded ? "−" : "+"}</span>
                    </div>
                  </div>

                  {/* Training Info Summary */}
                  <div className="mt-2 flex gap-4 text-xs text-slate">
                    <span>Train: {formatDate(model.train_start)} ~ {formatDate(model.train_end)}</span>
                    <span>({trainDays}일)</span>
                    <span>Samples: {metrics?.trade?.n_trades?.toLocaleString() ?? "-"}</span>
                  </div>
                </div>

                {/* Expanded Details */}
                {isExpanded && (
                  <div className="border-t border-ink/10 p-4 bg-white/50 space-y-4">
                    {/* Trade Metrics */}
                    <div>
                      <div className="text-xs font-medium text-slate uppercase mb-2">Trade Metrics</div>
                      <div className="grid grid-cols-5 gap-3 text-sm">
                        <div className="bg-mist rounded-lg p-3 text-center">
                          <div className="text-xs text-slate">Profit Factor</div>
                          <div className={`font-display text-lg ${
                            (metrics?.trade?.profit_factor ?? 0) >= 1 ? "text-emerald-600" : "text-rose-500"
                          }`}>
                            {metrics?.trade?.profit_factor?.toFixed(2) ?? "-"}
                          </div>
                        </div>
                        <div className="bg-mist rounded-lg p-3 text-center">
                          <div className="text-xs text-slate">Win Rate</div>
                          <div className="font-display text-lg">
                            {metrics?.trade?.win_rate ? `${(metrics.trade.win_rate * 100).toFixed(1)}%` : "-"}
                          </div>
                        </div>
                        <div className="bg-mist rounded-lg p-3 text-center">
                          <div className="text-xs text-slate">Expectancy</div>
                          <div className={`font-display text-lg ${
                            (metrics?.trade?.expectancy ?? 0) >= 0 ? "text-emerald-600" : "text-rose-500"
                          }`}>
                            {metrics?.trade?.expectancy ? `${(metrics.trade.expectancy * 100).toFixed(3)}%` : "-"}
                          </div>
                        </div>
                        <div className="bg-mist rounded-lg p-3 text-center">
                          <div className="text-xs text-slate">Max DD</div>
                          <div className="font-display text-lg text-rose-500">
                            {metrics?.trade?.max_drawdown?.toFixed(2) ?? "-"}
                          </div>
                        </div>
                        <div className="bg-mist rounded-lg p-3 text-center">
                          <div className="text-xs text-slate">N Trades</div>
                          <div className="font-display text-lg">
                            {metrics?.trade?.n_trades?.toLocaleString() ?? "-"}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Target Metrics */}
                    <div>
                      <div className="text-xs font-medium text-slate uppercase mb-2">Model Targets (RMSE)</div>
                      <div className="grid grid-cols-4 gap-3 text-sm">
                        <div className="bg-blue-50 rounded-lg p-3 text-center">
                          <div className="text-xs text-slate">er_long</div>
                          <div className="font-medium">{metrics?.er_long?.rmse?.toFixed(4) ?? "-"}</div>
                        </div>
                        <div className="bg-purple-50 rounded-lg p-3 text-center">
                          <div className="text-xs text-slate">q05_long</div>
                          <div className="font-medium">{metrics?.q05_long?.rmse?.toFixed(4) ?? "-"}</div>
                        </div>
                        <div className="bg-amber-50 rounded-lg p-3 text-center">
                          <div className="text-xs text-slate">e_mae_long</div>
                          <div className="font-medium">{metrics?.e_mae_long?.rmse?.toFixed(4) ?? "-"}</div>
                        </div>
                        <div className="bg-rose-50 rounded-lg p-3 text-center">
                          <div className="text-xs text-slate">Cost/Trade</div>
                          <div className="font-medium">
                            {metrics?.cost_breakdown
                              ? `${((metrics.cost_breakdown.fee_per_trade ?? 0) * 100 + (metrics.cost_breakdown.slippage_per_trade ?? 0) * 100).toFixed(3)}%`
                              : "-"}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Regime Performance */}
                    {metrics?.regime && Object.keys(metrics.regime).length > 0 && (
                      <div>
                        <div className="text-xs font-medium text-slate uppercase mb-2">BTC Regime Performance</div>
                        <div className="grid grid-cols-2 gap-3">
                          {Object.entries(metrics.regime).map(([regime, data]) => (
                            <div key={regime} className="bg-mist rounded-lg p-3">
                              <div className="flex items-center justify-between mb-2">
                                <span className={`text-xs px-2 py-1 rounded ${
                                  regime === "1" ? "bg-emerald-100 text-emerald-700" : "bg-rose-100 text-rose-700"
                                }`}>
                                  {regime === "1" ? "Bullish" : "Bearish"}
                                </span>
                                <span className="text-xs text-slate">
                                  {data.n_trades?.toLocaleString()} trades
                                </span>
                              </div>
                              <div className="flex justify-between text-sm">
                                <span>PF: <strong>{data.profit_factor?.toFixed(2) ?? "-"}</strong></span>
                                <span>Win: <strong>{data.win_rate ? `${(data.win_rate * 100).toFixed(1)}%` : "-"}</strong></span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Meta Info */}
                    <div className="text-xs text-slate space-y-1 pt-2 border-t border-ink/10">
                      <div>Label Spec: <code>{model.label_spec_hash}</code></div>
                      <div>Feature Schema: v{model.feature_schema_version}</div>
                      <div>Created: {new Date(model.created_at).toLocaleString("ko-KR")}</div>
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
