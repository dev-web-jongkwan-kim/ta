"use client";

import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from "react";

export interface TradingStats {
  started_at: string | null;
  running_time_sec: number;
  total_trades: number;
  wins: number;
  losses: number;
  total_pnl: number;
  gross_profit: number;
  gross_loss: number;
  win_rate: number | null;
  profit_factor: number | null;
  avg_hold_min: number | null;
  best_trade: number | null;
  worst_trade: number | null;
  mode?: string;
  current_equity?: number;
}

export interface Position {
  symbol: string;
  side: "LONG" | "SHORT";
  qty: number;
  entry_price: number;
  entry_time: string;
  sl_price: number | null;
  tp_price: number | null;
  trade_group_id: string | null;
}

export interface Trade {
  trade_id: string;
  symbol: string;
  side: string;
  entry_price: number;
  exit_price: number | null;
  qty: number;
  pnl: number | null;
  return_pct: number | null;
  hold_min: number | null;
  exit_reason: string | null;
  entry_time: string;
  exit_time: string | null;
}

export interface TradingEvent {
  type: "position_opened" | "position_closed" | "stats_update" | "trade_completed";
  symbol?: string;
  side?: string;
  qty?: number;
  entry_price?: number;
  exit_price?: number;
  entry_time?: string;
  exit_time?: string;
  sl_price?: number;
  tp_price?: number;
  pnl?: number;
  exit_reason?: string;
  stats?: TradingStats;
  trade?: Trade;
}

interface TradingEventsContextValue {
  connected: boolean;
  positions: Position[];
  stats: TradingStats | null;
  lastEvent: TradingEvent | null;
  refreshStats: () => Promise<void>;
  refreshPositions: () => Promise<void>;
  refreshTrades: () => Promise<Trade[]>;
}

const TradingEventsContext = createContext<TradingEventsContextValue | null>(null);

export function useTradingEvents() {
  const context = useContext(TradingEventsContext);
  if (!context) {
    throw new Error("useTradingEvents must be used within TradingEventsProvider");
  }
  return context;
}

interface TradingEventsProviderProps {
  children: React.ReactNode;
}

export function TradingEventsProvider({ children }: TradingEventsProviderProps) {
  const [connected, setConnected] = useState(false);
  const [positions, setPositions] = useState<Position[]>([]);
  const [stats, setStats] = useState<TradingStats | null>(null);
  const [lastEvent, setLastEvent] = useState<TradingEvent | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch("/api/trading/stats");
      const data = await res.json();
      setStats(data);
    } catch (error) {
      console.error("Failed to fetch trading stats:", error);
    }
  }, []);

  const fetchPositions = useCallback(async () => {
    try {
      const res = await fetch("/api/trading/positions");
      const data = await res.json();
      setPositions(data.positions || []);
    } catch (error) {
      console.error("Failed to fetch positions:", error);
    }
  }, []);

  const fetchTrades = useCallback(async (): Promise<Trade[]> => {
    try {
      const res = await fetch("/api/trading/trades?limit=50");
      const data = await res.json();
      return data.trades || [];
    } catch (error) {
      console.error("Failed to fetch trades:", error);
      return [];
    }
  }, []);

  const connect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const eventSource = new EventSource("/api/trading/stream");
    eventSourceRef.current = eventSource;

    eventSource.onopen = () => {
      setConnected(true);
      console.log("Trading SSE connected");
    };

    eventSource.onmessage = (event) => {
      try {
        const data: TradingEvent = JSON.parse(event.data);
        setLastEvent(data);

        switch (data.type) {
          case "position_opened":
            setPositions((prev) => {
              if (prev.some((p) => p.symbol === data.symbol)) {
                return prev;
              }
              return [
                ...prev,
                {
                  symbol: data.symbol!,
                  side: data.side as "LONG" | "SHORT",
                  qty: data.qty!,
                  entry_price: data.entry_price!,
                  entry_time: data.entry_time!,
                  sl_price: data.sl_price ?? null,
                  tp_price: data.tp_price ?? null,
                  trade_group_id: null,
                },
              ];
            });
            break;

          case "position_closed":
            setPositions((prev) => prev.filter((p) => p.symbol !== data.symbol));
            // Also refresh stats when position closes
            fetchStats();
            break;

          case "stats_update":
            if (data.stats) {
              setStats(data.stats);
            } else {
              fetchStats();
            }
            break;

          case "trade_completed":
            // Refresh stats when trade completes
            fetchStats();
            break;
        }
      } catch {
        // Ignore parse errors (heartbeats)
      }
    };

    eventSource.onerror = () => {
      setConnected(false);
      eventSource.close();
      eventSourceRef.current = null;

      // Reconnect after 3 seconds
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      reconnectTimeoutRef.current = setTimeout(() => {
        console.log("Reconnecting to trading SSE...");
        connect();
      }, 3000);
    };
  }, [fetchStats]);

  // Initial connection and data fetch
  useEffect(() => {
    fetchStats();
    fetchPositions();
    connect();

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [connect, fetchStats, fetchPositions]);

  const value: TradingEventsContextValue = {
    connected,
    positions,
    stats,
    lastEvent,
    refreshStats: fetchStats,
    refreshPositions: fetchPositions,
    refreshTrades: fetchTrades,
  };

  return (
    <TradingEventsContext.Provider value={value}>
      {children}
    </TradingEventsContext.Provider>
  );
}
