"use client";

import { useEffect, useState } from "react";
import { getClientApiBase } from "@/lib/client-api";

interface Order {
  order_id: string;
  symbol: string;
  side: string;
  type: string;
  status: string;
  qty: number;
  price: number | null;
  created_at: string;
}

export default function OrdersTable() {
  const [orders, setOrders] = useState<Order[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const apiBase = getClientApiBase();
        const res = await fetch(`${apiBase}/api/orders`);
        if (!res.ok) throw new Error("Failed to fetch");
        const data = await res.json();
        setOrders(data);
      } catch (e) {
        console.error("Failed to fetch orders:", e);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const getStatusBadge = (status: string) => {
    switch (status?.toUpperCase()) {
      case "FILLED":
        return "badge-success";
      case "NEW":
      case "OPEN":
        return "bg-accent-muted text-accent";
      case "CANCELED":
        return "badge-neutral";
      case "REJECTED":
        return "badge-danger";
      default:
        return "badge-neutral";
    }
  };

  return (
    <div className="card p-5">
      <h3 className="font-display font-semibold text-foreground mb-4">Orders</h3>

      {loading ? (
        <div className="space-y-2">
          <div className="skeleton h-10 rounded" />
          <div className="skeleton h-10 rounded" />
          <div className="skeleton h-10 rounded" />
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left border-b border-border">
                <th className="table-header pb-2">Order ID</th>
                <th className="table-header pb-2">Symbol</th>
                <th className="table-header pb-2">Side</th>
                <th className="table-header pb-2">Type</th>
                <th className="table-header pb-2 text-right">Qty</th>
                <th className="table-header pb-2">Status</th>
              </tr>
            </thead>
            <tbody>
              {orders.length === 0 ? (
                <tr>
                  <td colSpan={6} className="py-6 text-center text-foreground-muted">
                    No orders found
                  </td>
                </tr>
              ) : (
                orders.slice(0, 20).map((order) => (
                  <tr key={order.order_id} className="table-row">
                    <td className="py-2.5 font-mono text-xs text-foreground-muted">
                      {order.order_id.slice(0, 8)}...
                    </td>
                    <td className="py-2.5 font-mono font-medium text-foreground">{order.symbol}</td>
                    <td className="py-2.5">
                      <span className={`badge ${order.side?.toUpperCase() === "BUY" ? "badge-success" : "badge-danger"}`}>
                        {order.side}
                      </span>
                    </td>
                    <td className="py-2.5 text-foreground-secondary">{order.type}</td>
                    <td className="py-2.5 text-right font-mono text-foreground-secondary">{order.qty}</td>
                    <td className="py-2.5">
                      <span className={`badge ${getStatusBadge(order.status)}`}>
                        {order.status}
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
