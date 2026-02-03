import RiskBlockersCard from "@/components/cards/RiskBlockersCard";

export default function RiskPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-display">Risk Guardrails</h1>
      <RiskBlockersCard
        blockers={[
          { reason: "MARGIN_LIMIT", count: "0" },
          { reason: "MAX_POSITIONS", count: "1" },
          { reason: "DATA_STALE", count: "0" },
        ]}
      />
    </div>
  );
}
