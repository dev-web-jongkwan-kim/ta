"use client";

import { useState, ReactNode } from "react";

interface TooltipProps {
  content: string;
  children: ReactNode;
}

export default function Tooltip({ content, children }: TooltipProps) {
  const [show, setShow] = useState(false);

  return (
    <span
      className="relative inline-flex items-center cursor-help"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      {show && (
        <span className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 text-xs bg-foreground text-background rounded-lg shadow-lg whitespace-normal max-w-[250px] text-center">
          {content}
          <span className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-foreground" />
        </span>
      )}
    </span>
  );
}

// Term Dictionary
export const TERM_DEFINITIONS: Record<string, string> = {
  // Data Quality
  PSI: "Population Stability Index - Measures data distribution change. <0.1 good, >0.2 warning",
  Missing: "Missing Rate - Percentage of missing/null data points",
  Latency: "Data Latency - Time delay from real-time",
  Outlier: "Outlier Count - Data points outside normal range",

  // Risk
  MARGIN_LIMIT: "Margin Limit - Insufficient available margin",
  MAX_POSITIONS: "Max Positions - Position limit reached",
  DATA_STALE: "Data Stale - Real-time data too old",
  DAILY_LOSS: "Daily Loss - Daily loss limit reached",
  CIRCUIT_BREAKER: "Circuit Breaker - Trading halted due to volatility",

  // Trading Metrics
  PF: "Profit Factor - Gross Profit / Gross Loss. >1 profitable, >2 excellent",
  Expectancy: "Expectancy - Average return per trade (%)",
  WinRate: "Win Rate - Percentage of profitable trades",
  MaxDD: "Max Drawdown - Maximum peak-to-trough decline",
  Turnover: "Turnover - Total number of trades",

  // Model Targets
  er_long: "Expected Return Long - Predicted long position return",
  er_short: "Expected Return Short - Predicted short position return",
  q05_long: "5% Quantile Long - Worst 5% loss prediction for long",
  e_mae_long: "Expected MAE - Maximum adverse excursion prediction",
  e_hold_long: "Expected Hold Time - Predicted position hold time (min)",

  // Features
  ema_dist_atr: "EMA Distance - Price distance from EMA in ATR units",
  funding_z: "Funding Z-score - Funding rate deviation from mean",
  vol_z: "Volatility Z-score - Current volatility deviation",
  rsi: "RSI - Relative Strength Index. >70 overbought, <30 oversold",
  oi_change: "OI Change - Open interest rate of change",

  // Position Status
  FLAT: "Flat - No open positions",
  LONG: "Long - Betting on price increase (buy)",
  SHORT: "Short - Betting on price decrease (sell)",
  Shadow: "Shadow Mode - Simulation only, no real trades",
};

// Inline tooltip with question mark icon
export function InfoTooltip({ term }: { term: string }) {
  const definition = TERM_DEFINITIONS[term] || term;
  return (
    <Tooltip content={definition}>
      <span className="inline-flex items-center justify-center w-4 h-4 ml-1 text-[10px] text-foreground-muted border border-border rounded-full hover:bg-background-tertiary transition-colors">
        ?
      </span>
    </Tooltip>
  );
}
