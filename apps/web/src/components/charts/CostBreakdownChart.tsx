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
        const res = await fetch(`${apiBase}/api/models`);
        if (!res.ok) throw new Error("Failed to fetch");
        const models = await res.json();

        const prodModel = models.find((m: any) => m.is_production);
        if (prodModel?.metrics?.cost_breakdown) {
          const cb = prodModel.metrics.cost_breakdown;
          setCosts({
            fees: (cb.fee_per_trade || 0.0004) * 100,
            slippage: (cb.slippage_per_trade || 0.0003) * 100,
            funding: (cb.funding_per_trade || 0) * 100,
          });
        } else {
          setCosts({ fees: 0.04, slippage: 0.03, funding: 0.01 });
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
  const getWidth = (value: number) => (value / maxCost) * 100;

  return (
    <div className="card p-5">
      <h3 className="font-display font-semibold text-foreground mb-4">Cost Breakdown</h3>

      {loading ? (
        <div className="space-y-4">
          <div className="skeleton h-8 rounded" />
          <div className="skeleton h-8 rounded" />
          <div className="skeleton h-8 rounded" />
        </div>
      ) : (
        <div className="space-y-4">
          {/* Fees */}
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-foreground-muted">Fees</span>
              <span className="font-mono text-foreground">{costs.fees.toFixed(3)}%</span>
            </div>
            <div className="h-3 bg-background-tertiary rounded-full overflow-hidden">
              <div
                className="h-full bg-accent rounded-full transition-all duration-300"
                style={{ width: `${getWidth(costs.fees)}%` }}
              />
            </div>
          </div>

          {/* Slippage */}
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-foreground-muted">Slippage</span>
              <span className="font-mono text-foreground">{costs.slippage.toFixed(3)}%</span>
            </div>
            <div className="h-3 bg-background-tertiary rounded-full overflow-hidden">
              <div
                className="h-full bg-warning rounded-full transition-all duration-300"
                style={{ width: `${getWidth(costs.slippage)}%` }}
              />
            </div>
          </div>

          {/* Funding */}
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-foreground-muted">Funding</span>
              <span className="font-mono text-foreground">{costs.funding.toFixed(3)}%</span>
            </div>
            <div className="h-3 bg-background-tertiary rounded-full overflow-hidden">
              <div
                className="h-full bg-purple-500 rounded-full transition-all duration-300"
                style={{ width: `${getWidth(costs.funding)}%` }}
              />
            </div>
          </div>

          {/* Total */}
          <div className="pt-3 border-t border-border">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium text-foreground">Total Cost per Trade</span>
              <span className="font-mono text-xl font-semibold text-foreground">{total.toFixed(3)}%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
