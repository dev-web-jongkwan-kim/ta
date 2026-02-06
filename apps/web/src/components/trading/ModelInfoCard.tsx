"use client";

import { useEffect, useState } from "react";
import { getClientApiBase } from "@/lib/client-api";

interface FilteredMetrics {
  profit_factor: number;
  expectancy: number;
  max_drawdown: number;
  turnover: number;
  tail_loss: number;
}

interface ModelMetrics {
  trade?: {
    profit_factor: number;
    expectancy: number;
    max_drawdown: number;
    turnover: number;
    tail_loss: number;
  };
  regime?: Record<string, {
    profit_factor: number;
    expectancy: number;
    max_drawdown: number;
  }>;
  by_symbol?: Record<string, {
    profit_factor: number;
    expectancy: number;
    max_drawdown: number;
  }>;
  filtered_long?: Record<string, FilteredMetrics>;
  filtered_short?: Record<string, FilteredMetrics>;
}

interface ModelInfo {
  model_id: string;
  algo: string;
  created_at: string;
  train_start: string;
  train_end: string;
  metrics: ModelMetrics;
}

export default function ModelInfoCard() {
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    const fetchModel = async () => {
      try {
        const apiBase = getClientApiBase();
        const res = await fetch(`${apiBase}/api/models/production`);
        if (!res.ok) throw new Error("Failed to fetch");
        const data = await res.json();
        if (data.model_id) {
          setModel(data);
        }
      } catch (e) {
        console.error("Failed to fetch production model:", e);
      } finally {
        setLoading(false);
      }
    };

    fetchModel();
  }, []);

  if (loading) {
    return (
      <div className="card p-5">
        <div className="animate-pulse">
          <div className="h-5 bg-muted rounded w-40 mb-4" />
          <div className="grid grid-cols-4 gap-4">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-16 bg-muted rounded" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (!model) {
    return (
      <div className="card p-5">
        <p className="text-foreground-muted">No production model set</p>
      </div>
    );
  }

  const tradeMetrics = model.metrics?.trade;
  const pf = tradeMetrics?.profit_factor ?? 0;
  const expectancy = tradeMetrics?.expectancy ?? 0;
  const maxDD = tradeMetrics?.max_drawdown ?? 0;
  const turnover = tradeMetrics?.turnover ?? 0;
  const tailLoss = tradeMetrics?.tail_loss ?? 0;

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString("ko-KR", { year: "numeric", month: "short", day: "numeric" });
  };

  const getSymbolMetrics = () => {
    if (!model.metrics?.by_symbol) return [];
    return Object.entries(model.metrics.by_symbol)
      .map(([symbol, metrics]) => ({
        symbol,
        pf: metrics.profit_factor,
        expectancy: metrics.expectancy,
        maxDD: metrics.max_drawdown,
      }))
      .sort((a, b) => b.pf - a.pf);
  };

  const symbolMetrics = getSymbolMetrics();

  return (
    <div className="card p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="font-display font-semibold text-foreground">Production Model</h3>
          <p className="text-xs text-foreground-muted mt-1">
            {model.algo.toUpperCase()} | {formatDate(model.train_start)} ~ {formatDate(model.train_end)}
          </p>
        </div>
        <div className="text-xs font-mono text-foreground-muted">
          {model.model_id.slice(0, 8)}...
        </div>
      </div>

      {/* Main Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 mb-4">
        <div className="bg-muted/50 rounded-lg p-3">
          <p className="text-xs text-foreground-muted mb-1">Profit Factor</p>
          <p className={`text-xl font-mono font-semibold ${pf >= 1 ? "text-success" : "text-danger"}`}>
            {pf.toFixed(3)}
          </p>
        </div>

        <div className="bg-muted/50 rounded-lg p-3">
          <p className="text-xs text-foreground-muted mb-1">Expectancy</p>
          <p className={`text-xl font-mono font-semibold ${expectancy >= 0 ? "text-success" : "text-danger"}`}>
            {(expectancy * 100).toFixed(3)}%
          </p>
        </div>

        <div className="bg-muted/50 rounded-lg p-3">
          <p className="text-xs text-foreground-muted mb-1">Max Drawdown</p>
          <p className="text-xl font-mono font-semibold text-danger">
            {maxDD.toFixed(2)}
          </p>
        </div>

        <div className="bg-muted/50 rounded-lg p-3">
          <p className="text-xs text-foreground-muted mb-1">Tail Loss (5%)</p>
          <p className="text-xl font-mono font-semibold text-foreground">
            {(tailLoss * 100).toFixed(2)}%
          </p>
        </div>

        <div className="bg-muted/50 rounded-lg p-3">
          <p className="text-xs text-foreground-muted mb-1">Turnover</p>
          <p className="text-xl font-mono font-semibold text-foreground">
            {turnover.toLocaleString()}
          </p>
        </div>
      </div>

      {/* Regime Metrics */}
      {model.metrics?.regime && (
        <div className="border-t border-border pt-3 mb-3">
          <p className="text-xs text-foreground-muted mb-2">By Regime</p>
          <div className="flex gap-4 text-sm">
            {Object.entries(model.metrics.regime).map(([regime, metrics]) => (
              <div key={regime} className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${regime === "0" ? "bg-blue-500" : "bg-purple-500"}`} />
                <span className="text-foreground-secondary">
                  Regime {regime}: PF <span className={`font-mono ${metrics.profit_factor >= 1 ? "text-success" : "text-danger"}`}>
                    {metrics.profit_factor.toFixed(2)}
                  </span>
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ER Filter Performance */}
      {(model.metrics?.filtered_long || model.metrics?.filtered_short) && (
        <div className="border-t border-border pt-3 mb-3">
          <p className="text-xs text-foreground-muted mb-2">ER Filter Performance</p>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-foreground-muted">
                  <th className="text-left py-1 px-2">Filter</th>
                  <th className="text-right py-1 px-2">Long PF</th>
                  <th className="text-right py-1 px-2">Long Exp</th>
                  <th className="text-right py-1 px-2">Short PF</th>
                  <th className="text-right py-1 px-2">Short Exp</th>
                  <th className="text-right py-1 px-2">Trades</th>
                </tr>
              </thead>
              <tbody>
                {["er>0", "er>0.0005", "er>0.001", "er>0.0015", "er>0.002"].map((filter) => {
                  const longMetrics = model.metrics?.filtered_long?.[filter];
                  const shortMetrics = model.metrics?.filtered_short?.[filter];
                  if (!longMetrics && !shortMetrics) return null;

                  const longPF = longMetrics?.profit_factor ?? 0;
                  const shortPF = shortMetrics?.profit_factor ?? 0;
                  const longExp = longMetrics?.expectancy ?? 0;
                  const shortExp = shortMetrics?.expectancy ?? 0;
                  const totalTrades = (longMetrics?.turnover ?? 0) + (shortMetrics?.turnover ?? 0);

                  return (
                    <tr key={filter} className="border-t border-border/50">
                      <td className="py-1.5 px-2 font-mono">{filter}</td>
                      <td className={`text-right py-1.5 px-2 font-mono font-semibold ${longPF >= 1.5 ? "text-success" : longPF >= 1 ? "text-foreground" : "text-danger"}`}>
                        {longPF.toFixed(2)}
                      </td>
                      <td className={`text-right py-1.5 px-2 font-mono ${longExp >= 0 ? "text-success" : "text-danger"}`}>
                        {(longExp * 100).toFixed(2)}%
                      </td>
                      <td className={`text-right py-1.5 px-2 font-mono font-semibold ${shortPF >= 1.5 ? "text-success" : shortPF >= 1 ? "text-foreground" : "text-danger"}`}>
                        {shortPF.toFixed(2)}
                      </td>
                      <td className={`text-right py-1.5 px-2 font-mono ${shortExp >= 0 ? "text-success" : "text-danger"}`}>
                        {(shortExp * 100).toFixed(2)}%
                      </td>
                      <td className="text-right py-1.5 px-2 font-mono text-foreground-muted">
                        {totalTrades.toLocaleString()}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Symbol Performance Toggle */}
      {symbolMetrics.length > 0 && (
        <div className="border-t border-border pt-3">
          <button
            onClick={() => setExpanded(!expanded)}
            className="flex items-center gap-2 text-xs text-foreground-muted hover:text-foreground"
          >
            <svg
              className={`w-4 h-4 transition-transform ${expanded ? "rotate-180" : ""}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
            Symbol Performance ({symbolMetrics.length} symbols)
          </button>

          {expanded && (
            <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-2">
              {symbolMetrics.map(({ symbol, pf, expectancy }) => (
                <div
                  key={symbol}
                  className="flex items-center justify-between bg-background-tertiary/50 rounded px-2 py-1.5 text-xs"
                >
                  <span className="font-mono text-foreground">{symbol.replace("USDT", "")}</span>
                  <span className={`font-mono ${pf >= 1 ? "text-success" : "text-danger"}`}>
                    {pf.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
