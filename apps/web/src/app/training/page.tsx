import ModelsDetailTable from "@/components/tables/ModelsDetailTable";
import SignalFilterSummary from "@/components/cards/SignalFilterSummary";
import TrainingJobsTable from "@/components/tables/TrainingJobsTable";

export default function TrainingPage() {
  return (
    <div className="space-y-6">
      {/* Signal Filter Status */}
      <div className="grid gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <TrainingJobsTable />
        </div>
        <SignalFilterSummary />
      </div>

      {/* Supervised Learning Models */}
      <div>
        <h2 className="text-sm font-medium text-foreground-muted uppercase tracking-wider mb-4">
          Supervised Learning Models (LightGBM)
        </h2>
        <ModelsDetailTable />
      </div>
    </div>
  );
}
