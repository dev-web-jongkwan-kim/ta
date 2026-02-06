"use client";

import { createContext, useContext, useEffect, useState, useRef, ReactNode } from "react";
import { getClientApiBase } from "@/lib/client-api";

export interface WebsocketStatus {
  connected: boolean;
  uptime_sec: number;
  last_message_ago_sec: number | null;
  reconnect_count: number;
  streams_active: number;
  streams_total: number;
  total_messages: number;
  message_counts: Record<string, number>;
}

interface WebSocketStatusContextType {
  status: WebsocketStatus | null;
  loading: boolean;
  refresh: () => void;
}

const WebSocketStatusContext = createContext<WebSocketStatusContextType | undefined>(undefined);

// Polling interval - 5 seconds is sufficient for status monitoring
const POLL_INTERVAL_MS = 5000;

export function WebSocketStatusProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<WebsocketStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const isMounted = useRef(true);

  const fetchStatus = async () => {
    try {
      const apiBase = getClientApiBase();
      const res = await fetch(`${apiBase}/api/websocket/status`);
      if (res.ok && isMounted.current) {
        const data = await res.json();
        setStatus(data);
        setLoading(false);
      }
    } catch (e) {
      console.error("Failed to fetch WS status:", e);
      if (isMounted.current) {
        setLoading(false);
      }
    }
  };

  useEffect(() => {
    isMounted.current = true;
    fetchStatus();
    const interval = setInterval(fetchStatus, POLL_INTERVAL_MS);

    return () => {
      isMounted.current = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <WebSocketStatusContext.Provider value={{ status, loading, refresh: fetchStatus }}>
      {children}
    </WebSocketStatusContext.Provider>
  );
}

export function useWebSocketStatus() {
  const context = useContext(WebSocketStatusContext);
  if (context === undefined) {
    throw new Error("useWebSocketStatus must be used within a WebSocketStatusProvider");
  }
  return context;
}
