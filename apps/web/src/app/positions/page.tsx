import CostBreakdownChart from "@/components/charts/CostBreakdownChart";
import PositionsTable from "@/components/tables/PositionsTable";

export default function PositionsPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-display">Positions</h1>
      <div className="grid gap-6 lg:grid-cols-2">
        <PositionsTable />
        <CostBreakdownChart />
      </div>
    </div>
  );
}
