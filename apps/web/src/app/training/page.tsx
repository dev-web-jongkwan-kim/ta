import ModelsDetailTable from "@/components/tables/ModelsDetailTable";
import SignalFilterSummary from "@/components/cards/SignalFilterSummary";
import TrainingJobsTable from "@/components/tables/TrainingJobsTable";
import RLModelsTable from "@/components/tables/RLModelsTable";
import RLDecisionsTable from "@/components/tables/RLDecisionsTable";

export default function TrainingPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-display">Training & Models</h1>

      {/* Signal Filter Status */}
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <TrainingJobsTable />
        </div>
        <SignalFilterSummary />
      </div>

      {/* Supervised Learning Models */}
      <div>
        <h2 className="text-lg font-display mb-4 text-slate">Supervised Learning Models (LightGBM)</h2>
        <ModelsDetailTable />
      </div>

      {/* RL Section */}
      <div className="pt-6 border-t border-ink/10">
        <h2 className="text-lg font-display mb-4 text-slate">Reinforcement Learning Agent</h2>
        <div className="grid gap-6 lg:grid-cols-2">
          <RLModelsTable />
          <RLDecisionsTable />
        </div>
      </div>
    </div>
  );
}
