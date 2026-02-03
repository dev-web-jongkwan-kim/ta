"use client";

import { useEffect, useState } from "react";
import { getClientApiBase } from "@/lib/client-api";

interface CostData {
  fees: number;
  slippage: number;
  funding: number;
}

export default function CostBreakdownChart() {
  const [costs, setCosts] = useState<CostData>({ fees: 0, slippage: 0, funding: 0 });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const apiBase = getClientApiBase();

        // 모델에서 cost breakdown 가져오기
        const res = await fetch(`${apiBase}/api/models`);
        if (!res.ok) throw new Error("Failed to fetch");
        const models = await res.json();

        // production 모델의 metrics에서 cost_breakdown 추출
        const prodModel = models.find((m: any) => m.is_production);
        if (prodModel?.metrics?.cost_breakdown) {
          const cb = prodModel.metrics.cost_breakdown;
          setCosts({
            fees: (cb.fee_per_trade || 0.0004) * 100,
            slippage: (cb.slippage_per_trade || 0.0003) * 100,
            funding: (cb.funding_per_trade || 0) * 100,
          });
        } else {
          // 기본값
          setCosts({
            fees: 0.04,
            slippage: 0.03,
            funding: 0.01,
          });
        }
      } catch (e) {
        console.error("Failed to load costs:", e);
        setCosts({ fees: 0.04, slippage: 0.03, funding: 0.01 });
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const total = costs.fees + costs.slippage + costs.funding;
  const maxCost = Math.max(costs.fees, costs.slippage, costs.funding, 0.01);

  const getWidth = (value: number) => {
    return (value / maxCost) * 100;
  };

  return (
    <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
      <h3 className="font-display text-lg">Cost Breakdown</h3>
      <div className="mt-4 space-y-4">
        {loading ? (
          <div className="h-32 flex items-center justify-center text-slate text-sm">
            Loading...
          </div>
        ) : (
          <>
            {/* Fees */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate">Fees</span>
                <span className="font-medium">{costs.fees.toFixed(3)}%</span>
              </div>
              <div className="h-4 bg-mist rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 rounded-full"
                  style={{ width: `${getWidth(costs.fees)}%` }}
                />
              </div>
            </div>

            {/* Slippage */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate">Slippage</span>
                <span className="font-medium">{costs.slippage.toFixed(3)}%</span>
              </div>
              <div className="h-4 bg-mist rounded-full overflow-hidden">
                <div
                  className="h-full bg-amber-500 rounded-full"
                  style={{ width: `${getWidth(costs.slippage)}%` }}
                />
              </div>
            </div>

            {/* Funding */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate">Funding</span>
                <span className="font-medium">{costs.funding.toFixed(3)}%</span>
              </div>
              <div className="h-4 bg-mist rounded-full overflow-hidden">
                <div
                  className="h-full bg-purple-500 rounded-full"
                  style={{ width: `${getWidth(costs.funding)}%` }}
                />
              </div>
            </div>

            {/* Total */}
            <div className="pt-2 border-t border-ink/10">
              <div className="flex justify-between text-sm">
                <span className="font-medium">Total Cost per Trade</span>
                <span className="font-display text-lg">{total.toFixed(3)}%</span>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
