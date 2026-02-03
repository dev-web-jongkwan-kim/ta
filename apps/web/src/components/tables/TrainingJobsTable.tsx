import { fetchJSON } from "@/lib/api";

interface TrainingJob {
  job_id: string;
  status: string;
  progress: number | null;
  created_at: string;
  started_at: string | null;
  ended_at: string | null;
  error: string | null;
}

export default async function TrainingJobsTable() {
  let jobs: TrainingJob[] = [];
  try {
    jobs = await fetchJSON("/api/training/jobs");
  } catch (e) {
    console.error("Failed to fetch training jobs:", e);
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-emerald-100 text-emerald-700";
      case "running":
        return "bg-blue-100 text-blue-700";
      case "failed":
        return "bg-red-100 text-red-700";
      case "queued":
        return "bg-yellow-100 text-yellow-700";
      default:
        return "bg-gray-100 text-gray-600";
    }
  };

  return (
    <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
      <h3 className="font-display text-lg">Training Jobs</h3>
      <table className="mt-4 w-full text-sm">
        <thead className="text-slate">
          <tr className="text-left">
            <th className="py-2">Job</th>
            <th className="py-2">Status</th>
            <th className="py-2">Created</th>
            <th className="py-2">Error</th>
          </tr>
        </thead>
        <tbody>
          {jobs.length === 0 ? (
            <tr>
              <td colSpan={4} className="py-4 text-center text-slate">
                No training jobs found
              </td>
            </tr>
          ) : (
            jobs.slice(0, 10).map((job) => (
              <tr key={job.job_id} className="border-t border-ink/5">
                <td className="py-3 font-display text-xs">{job.job_id.slice(0, 8)}...</td>
                <td className="py-3">
                  <span className={`px-2 py-1 rounded-full text-xs ${getStatusColor(job.status)}`}>
                    {job.status}
                  </span>
                </td>
                <td className="py-3 text-slate text-xs">
                  {new Date(job.created_at).toLocaleString()}
                </td>
                <td className="py-3 text-red-500 text-xs truncate max-w-[150px]">
                  {job.error || "-"}
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
