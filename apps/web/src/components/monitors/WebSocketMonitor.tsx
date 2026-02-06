"use client";

import { useEffect, useRef } from "react";
import { useToast } from "@/contexts/ToastContext";
import { useWebSocketStatus, WebsocketStatus } from "@/contexts/WebSocketStatusContext";

export default function WebSocketMonitor() {
  const { addToast } = useToast();
  const { status } = useWebSocketStatus();
  const previousStatus = useRef<WebsocketStatus | null>(null);
  const hasShownInitialConnect = useRef(false);

  useEffect(() => {
    if (!status) return;

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
  }, [status, addToast]);

  return null; // This component doesn't render anything
}
