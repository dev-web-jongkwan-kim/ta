"use client";

import { useEffect, useState } from "react";
import { getClientApiBase } from "@/lib/client-api";
import { InfoTooltip } from "@/components/ui/Tooltip";

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
    return date.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });
  };

  const calculateTrainDays = (start: string, end: string) => {
    const startDate = new Date(start);
    const endDate = new Date(end);
    const diffTime = Math.abs(endDate.getTime() - startDate.getTime());
    return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  };

  if (loading) {
    return (
      <div className="card p-5">
        <h3 className="font-display font-semibold text-foreground mb-4">Models</h3>
        <div className="space-y-3">
          <div className="skeleton h-20 rounded" />
          <div className="skeleton h-20 rounded" />
        </div>
      </div>
    );
  }

  return (
    <div className="card p-5">
      <h3 className="font-display font-semibold text-foreground mb-4">Models</h3>

      <div className="space-y-3">
        {models.length === 0 ? (
          <div className="text-center text-foreground-muted py-6">No models found</div>
        ) : (
          models.map((model) => {
            const isExpanded = expandedId === model.model_id;
            const trainDays = calculateTrainDays(model.train_start, model.train_end);
            const metrics = model.metrics;

            return (
              <div
                key={model.model_id}
                className={`border rounded-lg overflow-hidden transition-colors ${
                  model.is_production ? "border-success bg-success-muted" : "border-border"
                }`}
              >
                {/* Header */}
                <div
                  className="p-4 cursor-pointer hover:bg-background-tertiary/30 transition-colors"
                  onClick={() => setExpandedId(isExpanded ? null : model.model_id)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <code className="text-xs bg-background-tertiary px-2 py-1 rounded font-mono">
                        {model.model_id.slice(0, 8)}
                      </code>
                      {model.is_production && (
                        <span className="badge badge-success">PRODUCTION</span>
                      )}
                      <span className="text-xs text-foreground-muted">{model.algo.toUpperCase()}</span>
                    </div>
                    <div className="flex items-center gap-4 text-sm">
                      <span className="text-foreground-muted">PF:</span>
                      <span className={`font-mono font-medium ${
                        (metrics?.trade?.profit_factor ?? 0) >= 1 ? "text-success" : "text-danger"
                      }`}>
                        {metrics?.trade?.profit_factor?.toFixed(2) ?? "-"}
                      </span>
                      <span className="text-foreground-muted">Win:</span>
                      <span className="font-mono font-medium text-foreground">
                        {metrics?.trade?.win_rate ? `${(metrics.trade.win_rate * 100).toFixed(1)}%` : "-"}
                      </span>
                      <span className="text-xl text-foreground-muted">{isExpanded ? "−" : "+"}</span>
                    </div>
                  </div>

                  {/* Training Info Summary */}
                  <div className="mt-2 flex gap-4 text-xs text-foreground-muted">
                    <span>Train: {formatDate(model.train_start)} ~ {formatDate(model.train_end)}</span>
                    <span>({trainDays} days)</span>
                    <span>Samples: {metrics?.trade?.n_trades?.toLocaleString() ?? "-"}</span>
                  </div>
                </div>

                {/* Expanded Details */}
                {isExpanded && (
                  <div className="border-t border-border p-4 bg-background space-y-4">
                    {/* Trade Metrics */}
                    <div>
                      <div className="text-xs font-medium text-foreground-muted uppercase mb-2">Trade Metrics</div>
                      <div className="grid grid-cols-5 gap-3 text-sm">
                        <div className="bg-background-tertiary rounded-lg p-3 text-center">
                          <div className="text-xs text-foreground-muted flex items-center justify-center gap-1">
                            Profit Factor <InfoTooltip term="PF" />
                          </div>
                          <div className={`font-mono text-lg font-semibold ${
                            (metrics?.trade?.profit_factor ?? 0) >= 1 ? "text-success" : "text-danger"
                          }`}>
                            {metrics?.trade?.profit_factor?.toFixed(2) ?? "-"}
                          </div>
                        </div>
                        <div className="bg-background-tertiary rounded-lg p-3 text-center">
                          <div className="text-xs text-foreground-muted flex items-center justify-center gap-1">
                            Win Rate <InfoTooltip term="WinRate" />
                          </div>
                          <div className="font-mono text-lg font-semibold text-foreground">
                            {metrics?.trade?.win_rate ? `${(metrics.trade.win_rate * 100).toFixed(1)}%` : "-"}
                          </div>
                        </div>
                        <div className="bg-background-tertiary rounded-lg p-3 text-center">
                          <div className="text-xs text-foreground-muted flex items-center justify-center gap-1">
                            Expectancy <InfoTooltip term="Expectancy" />
                          </div>
                          <div className={`font-mono text-lg font-semibold ${
                            (metrics?.trade?.expectancy ?? 0) >= 0 ? "text-success" : "text-danger"
                          }`}>
                            {metrics?.trade?.expectancy ? `${(metrics.trade.expectancy * 100).toFixed(3)}%` : "-"}
                          </div>
                        </div>
                        <div className="bg-background-tertiary rounded-lg p-3 text-center">
                          <div className="text-xs text-foreground-muted flex items-center justify-center gap-1">
                            Max DD <InfoTooltip term="MaxDD" />
                          </div>
                          <div className="font-mono text-lg font-semibold text-danger">
                            {metrics?.trade?.max_drawdown?.toFixed(2) ?? "-"}
                          </div>
                        </div>
                        <div className="bg-background-tertiary rounded-lg p-3 text-center">
                          <div className="text-xs text-foreground-muted">N Trades</div>
                          <div className="font-mono text-lg font-semibold text-foreground">
                            {metrics?.trade?.n_trades?.toLocaleString() ?? "-"}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Model Targets */}
                    <div>
                      <div className="text-xs font-medium text-foreground-muted uppercase mb-2">Model Targets (RMSE)</div>
                      <div className="grid grid-cols-4 gap-3 text-sm">
                        <div className="bg-accent-muted rounded-lg p-3 text-center">
                          <div className="text-xs text-foreground-muted flex items-center justify-center gap-1">
                            er_long <InfoTooltip term="er_long" />
                          </div>
                          <div className="font-mono font-medium text-foreground">{metrics?.er_long?.rmse?.toFixed(4) ?? "-"}</div>
                        </div>
                        <div className="bg-purple-500/10 rounded-lg p-3 text-center">
                          <div className="text-xs text-foreground-muted flex items-center justify-center gap-1">
                            q05_long <InfoTooltip term="q05_long" />
                          </div>
                          <div className="font-mono font-medium text-foreground">{metrics?.q05_long?.rmse?.toFixed(4) ?? "-"}</div>
                        </div>
                        <div className="bg-warning-muted rounded-lg p-3 text-center">
                          <div className="text-xs text-foreground-muted flex items-center justify-center gap-1">
                            e_mae_long <InfoTooltip term="e_mae_long" />
                          </div>
                          <div className="font-mono font-medium text-foreground">{metrics?.e_mae_long?.rmse?.toFixed(4) ?? "-"}</div>
                        </div>
                        <div className="bg-danger-muted rounded-lg p-3 text-center">
                          <div className="text-xs text-foreground-muted">Cost/Trade</div>
                          <div className="font-mono font-medium text-foreground">
                            {metrics?.cost_breakdown
                              ? `${((metrics.cost_breakdown.fee_per_trade ?? 0) * 100 + (metrics.cost_breakdown.slippage_per_trade ?? 0) * 100).toFixed(3)}%`
                              : "-"}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Meta Info */}
                    <div className="text-xs text-foreground-muted space-y-1 pt-2 border-t border-border">
                      <div>Label Spec: <code className="font-mono">{model.label_spec_hash}</code></div>
                      <div>Feature Schema: v{model.feature_schema_version}</div>
                      <div>Created: {new Date(model.created_at).toLocaleString()}</div>
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
