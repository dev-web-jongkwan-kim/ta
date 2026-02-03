"use client";

import { useParams } from "next/navigation";
import PriceChart from "@/components/charts/PriceChart";
import SignalSeriesChart from "@/components/charts/SignalSeriesChart";
import ExplainPanel from "@/components/panels/ExplainPanel";
import LiveOpsPanel from "@/components/panels/LiveOpsPanel";
import { InfoTooltip } from "@/components/ui/Tooltip";

export default function SymbolPage() {
  const params = useParams();
  const symbol = (params.symbol as string) || "BTCUSDT";

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-display">{symbol}</h1>
        <p className="text-sm text-slate">
          실시간 가격, 신호 시리즈, 모델 설명
          <InfoTooltip term="er_long" />
        </p>
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        <PriceChart symbol={symbol} />
        <SignalSeriesChart symbol={symbol} />
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        <ExplainPanel />
        <LiveOpsPanel />
      </div>
    </div>
  );
}
