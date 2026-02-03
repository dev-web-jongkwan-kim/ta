"use client";

import { useEffect, useState } from "react";
import { getClientApiBase } from "@/lib/client-api";

interface TrainingJob {
  job_id: string;
  status: string;
  progress: number | null;
  created_at: string;
  started_at: string | null;
  ended_at: string | null;
  error: string | null;
}

export default function TrainingJobsTable() {
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const apiBase = getClientApiBase();
        const res = await fetch(`${apiBase}/api/training/jobs`);
        if (!res.ok) throw new Error("Failed to fetch");
        const data = await res.json();
        setJobs(data);
      } catch (e) {
        console.error("Failed to fetch training jobs:", e);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return "badge-success";
      case "running":
        return "bg-accent-muted text-accent";
      case "failed":
        return "badge-danger";
      case "queued":
        return "badge-warning";
      default:
        return "badge-neutral";
    }
  };

  return (
    <div className="card p-5">
      <h3 className="font-display font-semibold text-foreground mb-4">Training Jobs</h3>

      {loading ? (
        <div className="space-y-2">
          <div className="skeleton h-10 rounded" />
          <div className="skeleton h-10 rounded" />
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left border-b border-border">
                <th className="table-header pb-2">Job</th>
                <th className="table-header pb-2">Status</th>
                <th className="table-header pb-2">Created</th>
                <th className="table-header pb-2">Error</th>
              </tr>
            </thead>
            <tbody>
              {jobs.length === 0 ? (
                <tr>
                  <td colSpan={4} className="py-6 text-center text-foreground-muted">
                    No training jobs found
                  </td>
                </tr>
              ) : (
                jobs.slice(0, 10).map((job) => (
                  <tr key={job.job_id} className="table-row">
                    <td className="py-2.5 font-mono text-xs text-foreground">
                      {job.job_id.slice(0, 8)}...
                    </td>
                    <td className="py-2.5">
                      <span className={`badge ${getStatusBadge(job.status)}`}>
                        {job.status}
                      </span>
                    </td>
                    <td className="py-2.5 text-foreground-muted text-xs">
                      {new Date(job.created_at).toLocaleString()}
                    </td>
                    <td className="py-2.5 text-danger text-xs truncate max-w-[150px]">
                      {job.error || "-"}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
