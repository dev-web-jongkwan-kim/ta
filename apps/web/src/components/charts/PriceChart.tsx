"use client";

import { useEffect, useRef, useState } from "react";
import { getClientApiBase } from "@/lib/client-api";

interface CandleData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

interface PriceChartProps {
  symbol?: string;
}

export default function PriceChart({ symbol = "BTCUSDT" }: PriceChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const [candles, setCandles] = useState<CandleData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const apiBase = getClientApiBase();
        const res = await fetch(`${apiBase}/api/symbol/${symbol}/series`);
        if (!res.ok) throw new Error("Failed to fetch");
        const data = await res.json();

        if (data.candles && data.candles.length > 0) {
          const formatted = data.candles
            .map((c: any[]) => ({
              time: c[1]?.split("+")[0]?.replace("T", " ") || "",
              open: c[2],
              high: c[3],
              low: c[4],
              close: c[5],
            }))
            .reverse()
            .slice(-100);
          setCandles(formatted);
        }
      } catch (e) {
        setError("Failed to load chart data");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [symbol]);

  // Simple candlestick visualization without external library
  const minPrice = candles.length > 0 ? Math.min(...candles.map((c) => c.low)) : 0;
  const maxPrice = candles.length > 0 ? Math.max(...candles.map((c) => c.high)) : 100;
  const priceRange = maxPrice - minPrice || 1;

  const getY = (price: number) => {
    return 180 - ((price - minPrice) / priceRange) * 160;
  };

  return (
    <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
      <div className="flex justify-between items-center">
        <h3 className="font-display text-lg">Price Action</h3>
        <span className="text-sm text-slate">{symbol}</span>
      </div>
      <div ref={chartContainerRef} className="mt-4 h-48 rounded-xl bg-mist overflow-hidden">
        {loading ? (
          <div className="h-full flex items-center justify-center text-slate text-sm">
            Loading...
          </div>
        ) : error ? (
          <div className="h-full flex items-center justify-center text-red-500 text-sm">
            {error}
          </div>
        ) : candles.length === 0 ? (
          <div className="h-full flex items-center justify-center text-slate text-sm">
            No data available
          </div>
        ) : (
          <svg width="100%" height="100%" viewBox="0 0 800 200" preserveAspectRatio="none">
            {candles.map((candle, i) => {
              const x = (i / candles.length) * 780 + 10;
              const width = Math.max(4, 780 / candles.length - 2);
              const isGreen = candle.close >= candle.open;
              const bodyTop = getY(Math.max(candle.open, candle.close));
              const bodyBottom = getY(Math.min(candle.open, candle.close));
              const bodyHeight = Math.max(1, bodyBottom - bodyTop);

              return (
                <g key={i}>
                  {/* Wick */}
                  <line
                    x1={x + width / 2}
                    y1={getY(candle.high)}
                    x2={x + width / 2}
                    y2={getY(candle.low)}
                    stroke={isGreen ? "#10b981" : "#ef4444"}
                    strokeWidth={1}
                  />
                  {/* Body */}
                  <rect
                    x={x}
                    y={bodyTop}
                    width={width}
                    height={bodyHeight}
                    fill={isGreen ? "#10b981" : "#ef4444"}
                  />
                </g>
              );
            })}
          </svg>
        )}
      </div>
      {candles.length > 0 && (
        <div className="mt-2 flex justify-between text-xs text-slate">
          <span>L: ${minPrice.toFixed(2)}</span>
          <span>H: ${maxPrice.toFixed(2)}</span>
        </div>
      )}
    </div>
  );
}
