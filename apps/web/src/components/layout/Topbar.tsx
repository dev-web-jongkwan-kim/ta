"use client";

import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { useWebSocketStatus } from "@/contexts/WebSocketStatusContext";

const pageInfo: Record<string, { title: string; subtitle: string }> = {
  "/dashboard": { title: "Dashboard", subtitle: "System overview and key metrics" },
  "/signals": { title: "Signals", subtitle: "Model predictions and trading signals" },
  "/positions": { title: "Positions", subtitle: "Current open positions" },
  "/orders": { title: "Orders", subtitle: "Order history and execution" },
  "/training": { title: "Training", subtitle: "Model training and performance" },
  "/risk": { title: "Risk", subtitle: "Risk management and blockers" },
  "/data-quality": { title: "Data Quality", subtitle: "Data drift and feature stability" },
  "/settings": { title: "Settings", subtitle: "System configuration" },
};

export default function Topbar() {
  const pathname = usePathname();
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const { status: wsStatus } = useWebSocketStatus();

  const info = pageInfo[pathname] || { title: "TA Ops", subtitle: "Trading system" };

  useEffect(() => {
    const interval = setInterval(() => {
      setLastUpdate(new Date());
    }, 60000);
    return () => clearInterval(interval);
  }, []);

  const formatTime = () => {
    const now = new Date();
    const diff = Math.floor((now.getTime() - lastUpdate.getTime()) / 1000);
    if (diff < 60) return "just now";
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    return `${Math.floor(diff / 3600)}h ago`;
  };

  const getWsStatusDisplay = () => {
    if (!wsStatus) {
      return { text: "WS Unknown", color: "text-foreground-muted", dot: "status-dot-muted" };
    }

    if (wsStatus.connected) {
      const stale = wsStatus.last_message_ago_sec !== null && wsStatus.last_message_ago_sec > 10;
      if (stale) {
        return { text: "WS Stale", color: "text-warning", dot: "status-dot-warning" };
      }
      return { text: "WS Connected", color: "text-success", dot: "status-dot-success" };
    }

    if (wsStatus.reconnect_count > 0) {
      return { text: `WS Reconnecting (${wsStatus.reconnect_count})`, color: "text-warning", dot: "status-dot-warning" };
    }

    return { text: "WS Disconnected", color: "text-danger", dot: "status-dot-danger" };
  };

  const wsDisplay = getWsStatusDisplay();

  return (
    <header className="h-16 flex items-center justify-between px-6 border-b border-border bg-background/80 backdrop-blur-sm sticky top-0 z-10">
      <div>
        <h1 className="text-lg font-display font-semibold text-foreground">{info.title}</h1>
        <p className="text-xs text-foreground-muted">{info.subtitle}</p>
      </div>

      <div className="flex items-center gap-4">
        {/* WebSocket Status */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-border">
          <span className={`status-dot ${wsDisplay.dot}`} />
          <span className={`text-xs font-medium ${wsDisplay.color}`}>{wsDisplay.text}</span>
        </div>

        {/* Status Badge */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-success-muted">
          <span className="status-dot status-dot-success" />
          <span className="text-xs font-medium text-success">Shadow</span>
        </div>

        {/* Last Update */}
        <div className="text-xs text-foreground-muted">
          Updated {formatTime()}
        </div>

        {/* Mobile Menu Button */}
        <button className="md:hidden p-2 rounded-lg hover:bg-background-tertiary">
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
      </div>
    </header>
  );
}
