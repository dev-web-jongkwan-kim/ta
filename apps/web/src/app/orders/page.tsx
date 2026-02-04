import OrdersTable from "@/components/tables/OrdersTable";
import TradingControl from "@/components/trading/TradingControl";
import TradingSummary from "@/components/trading/TradingSummary";
import TradeHistoryTable from "@/components/trading/TradeHistoryTable";

export default function OrdersPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-display">Trading & Orders</h1>

      {/* Trading Control */}
      <TradingControl />

      {/* Trading Summary */}
      <TradingSummary />

      {/* Trade History */}
      <TradeHistoryTable />

      {/* Orders Table */}
      <div className="card">
        <div className="p-5 border-b border-border">
          <h3 className="font-display font-semibold text-foreground">Orders</h3>
        </div>
        <OrdersTable />
      </div>
    </div>
  );
}
