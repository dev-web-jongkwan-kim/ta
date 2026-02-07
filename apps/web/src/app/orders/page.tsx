"use client";

import { TradingEventsProvider } from "@/contexts/TradingEventsContext";
import OrdersTable from "@/components/tables/OrdersTable";
import TradingControl from "@/components/trading/TradingControl";
import TradingSummaryLive from "@/components/trading/TradingSummaryLive";
import TradeHistoryTableLive from "@/components/trading/TradeHistoryTableLive";
import OpenPositionsLive from "@/components/trading/OpenPositionsLive";
import ModelInfoCard from "@/components/trading/ModelInfoCard";

export default function OrdersPage() {
  return (
    <TradingEventsProvider>
      <div className="space-y-6">
        <h1 className="text-2xl font-display">Trading & Orders</h1>

        {/* Trading Control */}
        <TradingControl />

        {/* Production Model Info */}
        <ModelInfoCard />

        {/* Trading Summary - Real-time via SSE */}
        <TradingSummaryLive />

        {/* Open Positions - Real-time via SSE */}
        <OpenPositionsLive />

        {/* Closed Trades History - Real-time via SSE */}
        <TradeHistoryTableLive />

        {/* Orders Table */}
        <div className="card">
          <div className="p-5 border-b border-border">
            <h3 className="font-display font-semibold text-foreground">Orders</h3>
          </div>
          <OrdersTable />
        </div>
      </div>
    </TradingEventsProvider>
  );
}
