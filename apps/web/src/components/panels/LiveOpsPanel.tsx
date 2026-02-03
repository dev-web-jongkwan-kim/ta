"use client";

import { useEffect, useState } from "react";
import { getClientApiBase } from "@/lib/client-api";
import { InfoTooltip } from "@/components/ui/Tooltip";

interface Position {
  symbol: string;
  side: string;
  amt: number;
  entry_price: number;
  mark_price: number;
  unrealized_pnl: number;
  leverage: number;
}

interface Order {
  symbol: string;
  side: string;
  type: string;
  status: string;
  qty: number;
  price: number;
}

interface Account {
  equity: number;
  unrealized_pnl: number;
  available_margin: number;
}

export default function LiveOpsPanel() {
  const [positions, setPositions] = useState<Position[]>([]);
  const [orders, setOrders] = useState<Order[]>([]);
  const [account, setAccount] = useState<Account | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const apiBase = getClientApiBase();

        const [posRes, ordRes, accRes] = await Promise.all([
          fetch(`${apiBase}/api/positions/latest`),
          fetch(`${apiBase}/api/orders`),
          fetch(`${apiBase}/api/account/latest`),
        ]);

        if (posRes.ok) {
          const posData = await posRes.json();
          // 실제 포지션만 필터 (amt > 0)
          setPositions(posData.filter((p: Position) => Math.abs(p.amt) > 0));
        }

        if (ordRes.ok) {
          const ordData = await ordRes.json();
          // 열린 주문만 필터
          setOrders(ordData.filter((o: Order) => o.status === "NEW" || o.status === "PARTIALLY_FILLED"));
        }

        if (accRes.ok) {
          const accData = await accRes.json();
          if (accData && accData.equity) {
            setAccount(accData);
          }
        }
      } catch (e) {
        console.error("Failed to load live ops:", e);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const getPositionStatus = () => {
    if (positions.length === 0) return "FLAT";
    const longPos = positions.find((p) => p.side === "LONG" || p.amt > 0);
    const shortPos = positions.find((p) => p.side === "SHORT" || p.amt < 0);
    if (longPos && shortPos) return "MIXED";
    if (longPos) return "LONG";
    if (shortPos) return "SHORT";
    return "FLAT";
  };

  const totalPnl = positions.reduce((sum, p) => sum + (p.unrealized_pnl || 0), 0);
  const positionStatus = getPositionStatus();

  const getStatusColor = (status: string) => {
    switch (status) {
      case "LONG":
        return "text-emerald-600";
      case "SHORT":
        return "text-rose-500";
      case "MIXED":
        return "text-amber-500";
      default:
        return "text-slate";
    }
  };

  return (
    <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
      <h3 className="font-display text-lg">Live Ops</h3>
      <p className="text-xs text-slate mt-1">실시간 포지션 및 주문 현황</p>

      {loading ? (
        <div className="mt-4 text-sm text-slate">Loading...</div>
      ) : (
        <div className="mt-4 space-y-3">
          {/* 포지션 상태 */}
          <div className="flex items-center justify-between">
            <span className="text-sm text-slate flex items-center">
              Position <InfoTooltip term={positionStatus} />
            </span>
            <span className={`font-display ${getStatusColor(positionStatus)}`}>
              {positionStatus}
            </span>
          </div>

          {/* 열린 주문 수 */}
          <div className="flex items-center justify-between">
            <span className="text-sm text-slate">Open Orders</span>
            <span className="font-display">{orders.length}</span>
          </div>

          {/* 미실현 PnL */}
          <div className="flex items-center justify-between">
            <span className="text-sm text-slate">Unrealized PnL</span>
            <span
              className={`font-display ${
                totalPnl > 0 ? "text-emerald-600" : totalPnl < 0 ? "text-rose-500" : ""
              }`}
            >
              {totalPnl >= 0 ? "+" : ""}
              {totalPnl.toFixed(2)} USDT
            </span>
          </div>

          {/* 계좌 정보 */}
          {account && (
            <div className="pt-2 border-t border-ink/10 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate">Equity</span>
                <span>{account.equity?.toFixed(2) || "-"} USDT</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate">Available</span>
                <span>{account.available_margin?.toFixed(2) || "-"} USDT</span>
              </div>
            </div>
          )}

          {/* 활성 포지션 목록 */}
          {positions.length > 0 && (
            <div className="pt-2 border-t border-ink/10">
              <div className="text-xs text-slate mb-2">Active Positions</div>
              <div className="space-y-1">
                {positions.slice(0, 3).map((pos, idx) => (
                  <div key={idx} className="flex items-center justify-between text-xs">
                    <span className="font-medium">{pos.symbol}</span>
                    <span
                      className={
                        pos.side === "LONG" || pos.amt > 0
                          ? "text-emerald-600"
                          : "text-rose-500"
                      }
                    >
                      {pos.side || (pos.amt > 0 ? "LONG" : "SHORT")} {Math.abs(pos.amt)}
                    </span>
                  </div>
                ))}
                {positions.length > 3 && (
                  <div className="text-xs text-slate">
                    +{positions.length - 3} more
                  </div>
                )}
              </div>
            </div>
          )}

          {/* 데이터 없음 메시지 */}
          {!account && positions.length === 0 && orders.length === 0 && (
            <div className="text-xs text-slate pt-2 border-t border-ink/10">
              No active trading data.
              <div className="mt-1">시스템이 실행되면 여기에 표시됩니다.</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
