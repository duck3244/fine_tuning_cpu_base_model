import { useEffect, useRef, useState } from "react";
import { DatasetInfo, JobInfo, api, streamJobLog } from "../api/client";

export default function Train() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [datasetId, setDatasetId] = useState("");
  const [maxSteps, setMaxSteps] = useState(100);
  const [batchSize, setBatchSize] = useState(1);
  const [lr, setLr] = useState(5e-5);
  const [epochs, setEpochs] = useState(1);
  const [job, setJob] = useState<JobInfo | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const stopStreamRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    api.listDatasets().then(setDatasets).catch((e) => setError(e.message));
  }, []);

  useEffect(() => () => stopStreamRef.current?.(), []);

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const info = await api.uploadDataset(file);
      setDatasets((d) => [...d, info]);
      setDatasetId(info.dataset_id);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function start() {
    setError(null);
    setLogs([]);
    try {
      const j = await api.startTrain({
        dataset_id: datasetId,
        max_steps: maxSteps,
        batch_size: batchSize,
        learning_rate: lr,
        epochs,
      });
      setJob(j);
      stopStreamRef.current = streamJobLog(
        j.job_id,
        (line) => setLogs((prev) => [...prev, line]),
        (status) => api.getJob(j.job_id).then((updated) => setJob({ ...updated, status: status as JobInfo["status"] })),
      );
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function stop() {
    if (!job) return;
    await api.stopJob(job.job_id).catch((e) => setError(e.message));
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">학습</h1>
      {error && <div className="rounded bg-red-50 p-4 text-red-700">{error}</div>}

      <section className="space-y-4 rounded border bg-white p-6">
        <div>
          <label className="mb-1 block text-sm font-medium">데이터셋</label>
          <div className="flex gap-2">
            <select
              value={datasetId}
              onChange={(e) => setDatasetId(e.target.value)}
              className="flex-1 rounded border border-slate-300 px-3 py-2 text-sm"
            >
              <option value="">선택...</option>
              {datasets.map((d) => (
                <option key={d.dataset_id} value={d.dataset_id}>
                  {d.filename} ({d.rows} rows)
                </option>
              ))}
            </select>
            <label className="cursor-pointer rounded border border-slate-300 px-3 py-2 text-sm hover:bg-slate-50">
              CSV 업로드
              <input type="file" accept=".csv" className="hidden" onChange={handleUpload} />
            </label>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Field label="Max steps" value={maxSteps} onChange={setMaxSteps} />
          <Field label="Batch size" value={batchSize} onChange={setBatchSize} />
          <Field label="Learning rate" value={lr} onChange={setLr} step={1e-5} />
          <Field label="Epochs" value={epochs} onChange={setEpochs} />
        </div>

        <div className="flex gap-2">
          <button
            disabled={!datasetId || job?.status === "running"}
            onClick={start}
            className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            학습 시작
          </button>
          {job?.status === "running" && (
            <button
              onClick={stop}
              className="rounded border border-red-300 bg-white px-4 py-2 text-sm font-medium text-red-700 hover:bg-red-50"
            >
              중단
            </button>
          )}
        </div>
      </section>

      {job && (
        <section className="rounded border bg-white p-6">
          <div className="mb-3 flex items-center justify-between text-sm">
            <span className="font-mono">{job.job_id}</span>
            <span className="text-slate-500">{job.status}</span>
          </div>
          <pre className="max-h-96 overflow-auto rounded bg-slate-900 p-4 text-xs text-slate-100">
            {logs.join("\n") || "(로그 대기 중...)"}
          </pre>
        </section>
      )}
    </div>
  );
}

function Field({
  label,
  value,
  onChange,
  step,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-sm font-medium">{label}</span>
      <input
        type="number"
        value={value}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full rounded border border-slate-300 px-3 py-2 text-sm"
      />
    </label>
  );
}
