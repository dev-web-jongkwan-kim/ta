import DriftCard from "@/components/cards/DriftCard";

export default function DataQualityPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-display">Data Quality</h1>
      <DriftCard
        metrics={[
          { symbol: "BTCUSDT", psi: "0.12", missing: "0.01%", latency: "12s" },
          { symbol: "ETHUSDT", psi: "0.08", missing: "0.00%", latency: "9s" },
        ]}
      />
    </div>
  );
}
