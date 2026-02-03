import PriceChart from "@/components/charts/PriceChart";
import SignalSeriesChart from "@/components/charts/SignalSeriesChart";
import ExplainPanel from "@/components/panels/ExplainPanel";
import LiveOpsPanel from "@/components/panels/LiveOpsPanel";

export default function SymbolPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-display">BTCUSDT</h1>
        <p className="text-sm text-slate">Mark vs last, SL/TP, liquidation lines.</p>
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        <PriceChart />
        <SignalSeriesChart />
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        <ExplainPanel />
        <LiveOpsPanel />
      </div>
    </div>
  );
}
