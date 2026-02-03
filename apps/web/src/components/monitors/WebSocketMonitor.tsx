"use client";

import { useEffect, useRef } from "react";
import { getClientApiBase } from "@/lib/client-api";
import { useToast } from "@/contexts/ToastContext";

interface WebsocketStatus {
  connected: boolean;
  reconnect_count: number;
}

export default function WebSocketMonitor() {
  const { addToast } = useToast();
  const previousStatus = useRef<WebsocketStatus | null>(null);
  const hasShownInitialConnect = useRef(false);

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const apiBase = getClientApiBase();
        const res = await fetch(`${apiBase}/api/websocket/status`);
        if (res.ok) {
          const status: WebsocketStatus = await res.json();

          // On first load, don't show toast if already connected
          if (!hasShownInitialConnect.current && status.connected) {
            hasShownInitialConnect.current = true;
            previousStatus.current = status;
            return;
          }

          // Check for state changes
          if (previousStatus.current) {
            // Disconnected
            if (previousStatus.current.connected && !status.connected) {
              addToast("WebSocket disconnected", "error");
            }

            // Reconnected
            if (!previousStatus.current.connected && status.connected) {
              addToast("WebSocket reconnected", "success");
            }

            // Reconnecting
            if (
              !status.connected &&
              status.reconnect_count > 0 &&
              status.reconnect_count !== previousStatus.current.reconnect_count
            ) {
              addToast(`Reconnecting... (attempt ${status.reconnect_count})`, "warning");
            }
          }

          previousStatus.current = status;
          hasShownInitialConnect.current = true;
        }
      } catch (e) {
        // Silently fail - don't spam toasts on network errors
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 3000);
    return () => clearInterval(interval);
  }, [addToast]);

  return null; // This component doesn't render anything
}
