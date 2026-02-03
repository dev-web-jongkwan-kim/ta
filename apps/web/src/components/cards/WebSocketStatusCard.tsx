"use client";

import { useEffect, useState } from "react";
import { getClientApiBase } from "@/lib/client-api";

interface WebsocketStatus {
  connected: boolean;
  uptime_sec: number;
  last_message_ago_sec: number | null;
  reconnect_count: number;
  streams_active: number;
  streams_total: number;
  total_messages: number;
  message_counts: Record<string, number>;
}

export default function WebSocketStatusCard() {
  const [status, setStatus] = useState<WebsocketStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const apiBase = getClientApiBase();
        const res = await fetch(`${apiBase}/api/websocket/status`);
        if (res.ok) {
          const data = await res.json();
          setStatus(data);
        }
      } catch (e) {
        console.error("Failed to fetch WS status:", e);
      } finally {
        setLoading(false);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  const formatUptime = (seconds: number) => {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  const formatMessagesPerSec = () => {
    if (!status) return "0";
    const recent = status.last_message_ago_sec;
    if (recent === null || recent > 60) return "0";
    // Rough estimate based on total messages and uptime
    if (status.uptime_sec > 0) {
      const rate = status.total_messages / status.uptime_sec;
      return rate.toFixed(1);
    }
    return "0";
  };

  if (loading) {
    return (
      <div className="card p-5">
        <h3 className="font-display font-semibold text-foreground mb-4">WebSocket Status</h3>
        <div className="space-y-3">
          <div className="skeleton h-8 rounded" />
          <div className="skeleton h-8 rounded" />
          <div className="skeleton h-8 rounded" />
        </div>
      </div>
    );
  }

  if (!status) {
    return (
      <div className="card p-5">
        <h3 className="font-display font-semibold text-foreground mb-4">WebSocket Status</h3>
        <div className="text-center text-foreground-muted py-4">
          No status available
        </div>
      </div>
    );
  }

  const getStatusColor = () => {
    if (!status.connected) return "text-danger";
    if (status.last_message_ago_sec !== null && status.last_message_ago_sec > 10) return "text-warning";
    return "text-success";
  };

  const getStatusDot = () => {
    if (!status.connected) return "status-dot-danger";
    if (status.last_message_ago_sec !== null && status.last_message_ago_sec > 10) return "status-dot-warning";
    return "status-dot-success";
  };

  const getStatusText = () => {
    if (!status.connected) {
      return status.reconnect_count > 0 ? `Reconnecting (attempt ${status.reconnect_count})` : "Disconnected";
    }
    if (status.last_message_ago_sec !== null && status.last_message_ago_sec > 10) {
      return "Connected (stale data)";
    }
    return "Connected";
  };

  // Top 5 stream types by message count
  const topStreams = Object.entries(status.message_counts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  return (
    <div className="card p-5">
      <h3 className="font-display font-semibold text-foreground mb-4">WebSocket Status</h3>

      {/* Main Status */}
      <div className="flex items-center gap-3 mb-4 pb-4 border-b border-border">
        <span className={`status-dot ${getStatusDot()}`} />
        <div className="flex-1">
          <div className={`font-medium ${getStatusColor()}`}>{getStatusText()}</div>
          {status.connected && (
            <div className="text-xs text-foreground-muted">
              Uptime: {formatUptime(status.uptime_sec)}
            </div>
          )}
        </div>
      </div>

      {/* Metrics */}
      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-sm text-foreground-muted">Streams</span>
          <span className="font-mono text-sm font-medium text-foreground">
            {status.streams_active}/{status.streams_total}
          </span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-sm text-foreground-muted">Messages/sec</span>
          <span className="font-mono text-sm font-medium text-foreground">
            {formatMessagesPerSec()}
          </span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-sm text-foreground-muted">Total messages</span>
          <span className="font-mono text-sm font-medium text-foreground">
            {status.total_messages.toLocaleString()}
          </span>
        </div>

        {status.last_message_ago_sec !== null && (
          <div className="flex justify-between items-center">
            <span className="text-sm text-foreground-muted">Last message</span>
            <span className="font-mono text-sm font-medium text-foreground">
              {status.last_message_ago_sec < 1 ? "<1s ago" : `${status.last_message_ago_sec.toFixed(1)}s ago`}
            </span>
          </div>
        )}
      </div>

      {/* Top Streams */}
      {topStreams.length > 0 && (
        <>
          <div className="mt-4 pt-4 border-t border-border">
            <div className="text-xs font-medium text-foreground-muted uppercase mb-2">
              Top Streams
            </div>
            <div className="space-y-2">
              {topStreams.map(([streamType, count]) => (
                <div key={streamType} className="flex justify-between items-center">
                  <span className="text-xs text-foreground-muted font-mono">{streamType}</span>
                  <span className="text-xs font-medium text-foreground">
                    {count.toLocaleString()}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
