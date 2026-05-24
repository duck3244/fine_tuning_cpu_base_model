import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, JobInfo, ModelInfo } from "../api/client";

export default function Dashboard() {
  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.listJobs(), api.listModels()])
      .then(([j, m]) => {
        setJobs(j);
        setModels(m);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <div className="space-y-8">
      <h1 className="text-2xl font-semibold">대시보드</h1>
      {error && <div className="rounded bg-red-50 p-4 text-red-700">{error}</div>}

      <section>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-lg font-medium">최근 학습 작업</h2>
          <Link to="/train" className="text-sm text-blue-600 hover:underline">
            새 학습 시작 →
          </Link>
        </div>
        {jobs.length === 0 ? (
          <p className="text-sm text-slate-500">아직 학습 작업이 없습니다.</p>
        ) : (
          <ul className="divide-y rounded border bg-white">
            {jobs.slice(0, 5).map((j) => (
              <li key={j.job_id} className="flex items-center justify-between px-4 py-3 text-sm">
                <span className="font-mono">{j.job_id}</span>
                <span className="text-slate-500">{j.started_at}</span>
                <StatusBadge status={j.status} />
              </li>
            ))}
          </ul>
        )}
      </section>

      <section>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-lg font-medium">저장된 모델</h2>
          <Link to="/models" className="text-sm text-blue-600 hover:underline">
            전체 보기 →
          </Link>
        </div>
        {models.length === 0 ? (
          <p className="text-sm text-slate-500">저장된 모델이 없습니다.</p>
        ) : (
          <ul className="divide-y rounded border bg-white">
            {models.slice(0, 5).map((m) => (
              <li key={m.model_id} className="flex items-center justify-between px-4 py-3 text-sm">
                <span className="font-mono">{m.model_id}</span>
                <span className="text-slate-500">{(m.size_bytes / 1024 / 1024).toFixed(1)} MB</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function StatusBadge({ status }: { status: JobInfo["status"] }) {
  const styles: Record<JobInfo["status"], string> = {
    running: "bg-blue-100 text-blue-700",
    completed: "bg-green-100 text-green-700",
    failed: "bg-red-100 text-red-700",
    stopped: "bg-slate-200 text-slate-700",
  };
  return (
    <span className={`rounded px-2 py-0.5 text-xs font-medium ${styles[status]}`}>{status}</span>
  );
}
