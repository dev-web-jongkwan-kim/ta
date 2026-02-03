import { fetchJSON } from "@/lib/api";

interface Model {
  model_id: string;
  algo: string;
  is_production: boolean;
  metrics: {
    trade?: {
      profit_factor?: number;
      expectancy?: number;
    };
  } | null;
  created_at: string;
}

export default async function ModelsTable() {
  let models: Model[] = [];
  try {
    models = await fetchJSON("/api/models");
  } catch (e) {
    console.error("Failed to fetch models:", e);
  }

  return (
    <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
      <h3 className="font-display text-lg">Models</h3>
      <table className="mt-4 w-full text-sm">
        <thead className="text-slate">
          <tr className="text-left">
            <th className="py-2">Model</th>
            <th className="py-2">Status</th>
            <th className="py-2">PF</th>
            <th className="py-2">Expectancy</th>
          </tr>
        </thead>
        <tbody>
          {models.length === 0 ? (
            <tr>
              <td colSpan={4} className="py-4 text-center text-slate">
                No models found
              </td>
            </tr>
          ) : (
            models.map((model) => (
              <tr key={model.model_id} className="border-t border-ink/5">
                <td className="py-3 font-display text-xs">{model.model_id.slice(0, 8)}...</td>
                <td className="py-3">
                  <span
                    className={`px-2 py-1 rounded-full text-xs ${
                      model.is_production
                        ? "bg-emerald-100 text-emerald-700"
                        : "bg-gray-100 text-gray-600"
                    }`}
                  >
                    {model.is_production ? "production" : "staging"}
                  </span>
                </td>
                <td className="py-3 text-slate">
                  {model.metrics?.trade?.profit_factor?.toFixed(2) ?? "-"}
                </td>
                <td className="py-3 text-slate">
                  {model.metrics?.trade?.expectancy
                    ? (model.metrics.trade.expectancy * 100).toFixed(3) + "%"
                    : "-"}
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
