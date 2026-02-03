"use client";

import { useState } from "react";

type TradingMode = "off" | "shadow" | "live";

interface TradingControlProps {
  compact?: boolean;
}

export default function TradingControl({ compact = false }: TradingControlProps) {
  const [mode, setMode] = useState<TradingMode>("off");
  const [isStarting, setIsStarting] = useState(false);

  const handleStart = async (newMode: TradingMode) => {
    if (newMode === mode) {
      // Stop trading
      setMode("off");
      return;
    }

    setIsStarting(true);
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 500));
    setMode(newMode);
    setIsStarting(false);
  };

  if (compact) {
    return (
      <div className="card p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className={`status-dot ${
              mode === "live" ? "status-dot-success" :
              mode === "shadow" ? "status-dot-warning" :
              "status-dot-muted"
            }`} />
            <span className="font-medium text-foreground">
              {mode === "off" ? "Trading Off" :
               mode === "shadow" ? "Shadow Mode" :
               "Live Trading"}
            </span>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => handleStart("shadow")}
              disabled={isStarting}
              className={`btn btn-sm ${mode === "shadow" ? "btn-primary" : "btn-secondary"}`}
            >
              Shadow
            </button>
            <button
              onClick={() => handleStart("live")}
              disabled={isStarting}
              className={`btn btn-sm ${mode === "live" ? "bg-success text-white hover:bg-success/90" : "btn-secondary"}`}
            >
              Live
            </button>
            {mode !== "off" && (
              <button
                onClick={() => setMode("off")}
                disabled={isStarting}
                className="btn btn-sm btn-danger"
              >
                Stop
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="card p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-display font-semibold text-foreground">Trading Control</h3>
        <div className="flex items-center gap-2">
          <span className={`status-dot ${
            mode === "live" ? "status-dot-success" :
            mode === "shadow" ? "status-dot-warning" :
            "status-dot-muted"
          }`} />
          <span className={`text-sm font-medium ${
            mode === "live" ? "text-success" :
            mode === "shadow" ? "text-warning" :
            "text-foreground-muted"
          }`}>
            {mode === "off" ? "Stopped" :
             mode === "shadow" ? "Shadow" :
             "Live"}
          </span>
        </div>
      </div>

      <div className="flex flex-wrap gap-3">
        <button
          onClick={() => handleStart("shadow")}
          disabled={isStarting}
          className={`btn flex-1 ${
            mode === "shadow"
              ? "bg-warning text-white hover:bg-warning/90"
              : "btn-secondary"
          }`}
        >
          <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
          </svg>
          {mode === "shadow" ? "Shadow Running" : "Start Shadow"}
        </button>

        <button
          onClick={() => handleStart("live")}
          disabled={isStarting}
          className={`btn flex-1 ${
            mode === "live"
              ? "bg-success text-white hover:bg-success/90"
              : "btn-secondary"
          }`}
        >
          <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          {mode === "live" ? "Live Running" : "Start Live"}
        </button>

        {mode !== "off" && (
          <button
            onClick={() => setMode("off")}
            disabled={isStarting}
            className="btn btn-danger"
          >
            <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
            </svg>
            Stop
          </button>
        )}
      </div>

      <p className="text-xs text-foreground-muted mt-3">
        {mode === "off" && "Select a mode to start trading."}
        {mode === "shadow" && "Shadow mode: Simulating trades without real orders."}
        {mode === "live" && "Live mode: Real orders are being placed."}
      </p>
    </div>
  );
}
