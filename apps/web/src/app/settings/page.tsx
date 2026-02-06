"use client";

import { useState } from "react";

export default function SettingsPage() {
  const [mode, setMode] = useState<"shadow" | "live" | "off">("shadow");

  return (
    <div className="space-y-6">
      {/* Trading Control */}
      <div className="card p-5">
        <h3 className="font-display font-semibold text-foreground mb-4">Trading Control</h3>
        <div className="flex flex-wrap gap-3">
          <button
            onClick={() => setMode("shadow")}
            className={`btn ${mode === "shadow" ? "btn-primary" : "btn-secondary"}`}
          >
            <span className="status-dot status-dot-warning mr-2" />
            Shadow Mode
          </button>
          <button
            onClick={() => setMode("live")}
            className={`btn ${mode === "live" ? "bg-success text-white" : "btn-secondary"}`}
          >
            <span className="status-dot status-dot-success mr-2" />
            Live Mode
          </button>
          <button
            onClick={() => setMode("off")}
            className={`btn ${mode === "off" ? "btn-danger" : "btn-secondary"}`}
          >
            <span className="status-dot status-dot-danger mr-2" />
            Off
          </button>
        </div>
        <p className="text-xs text-foreground-muted mt-3">
          {mode === "shadow" && "Simulation only - no real orders will be placed."}
          {mode === "live" && "Real orders will be placed. Use with caution."}
          {mode === "off" && "Trading is disabled."}
        </p>
      </div>

      {/* API Configuration */}
      <div className="card p-5">
        <h3 className="font-display font-semibold text-foreground mb-4">API Configuration</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm text-foreground-muted mb-1">API Endpoint</label>
            <input
              type="text"
              className="input"
              defaultValue="http://localhost:7101"
              disabled
            />
          </div>
          <div>
            <label className="block text-sm text-foreground-muted mb-1">Refresh Interval</label>
            <select className="input">
              <option value="5000">5 seconds</option>
              <option value="10000">10 seconds</option>
              <option value="30000">30 seconds</option>
            </select>
          </div>
        </div>
      </div>

      {/* Risk Limits */}
      <div className="card p-5">
        <h3 className="font-display font-semibold text-foreground mb-4">Risk Limits</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-foreground-muted mb-1">Max Positions</label>
            <input type="number" className="input" defaultValue={5} />
          </div>
          <div>
            <label className="block text-sm text-foreground-muted mb-1">Daily Loss Limit (%)</label>
            <input type="number" className="input" defaultValue={2} step={0.1} />
          </div>
          <div>
            <label className="block text-sm text-foreground-muted mb-1">Max Leverage</label>
            <input type="number" className="input" defaultValue={3} />
          </div>
          <div>
            <label className="block text-sm text-foreground-muted mb-1">Position Size (%)</label>
            <input type="number" className="input" defaultValue={5} step={0.5} />
          </div>
        </div>
        <button className="btn btn-primary mt-4">Save Changes</button>
      </div>
    </div>
  );
}
