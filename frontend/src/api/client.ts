export interface DatasetInfo {
  dataset_id: string;
  filename: string;
  rows: number;
  size_bytes: number;
}

export interface ModelInfo {
  model_id: string;
  path: string;
  created_at: string;
  size_bytes: number;
}

export interface TrainRequest {
  dataset_id: string;
  model_name?: string;
  max_steps?: number;
  batch_size?: number;
  learning_rate?: number;
  epochs?: number;
}

export interface JobInfo {
  job_id: string;
  status: "running" | "completed" | "failed" | "stopped";
  started_at: string;
  finished_at?: string | null;
  exit_code?: number | null;
  output_dir?: string | null;
  request?: TrainRequest | null;
}

export interface InferResponse {
  question: string;
  answer: string;
  elapsed_ms: number;
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  listDatasets: () => request<DatasetInfo[]>("/api/datasets"),
  uploadDataset: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return request<DatasetInfo>("/api/datasets/upload", { method: "POST", body: fd });
  },
  deleteDataset: (id: string) =>
    request<{ deleted: string }>(`/api/datasets/${id}`, { method: "DELETE" }),

  listModels: () => request<ModelInfo[]>("/api/models"),
  deleteModel: (id: string) =>
    request<{ deleted: string }>(`/api/models/${id}`, { method: "DELETE" }),

  startTrain: (req: TrainRequest) =>
    request<JobInfo>("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }),
  listJobs: () => request<JobInfo[]>("/api/train/jobs"),
  getJob: (id: string) => request<JobInfo>(`/api/train/jobs/${id}`),
  stopJob: (id: string) =>
    request<{ stopped: string }>(`/api/train/jobs/${id}/stop`, { method: "POST" }),

  infer: (model_id: string, question: string) =>
    request<InferResponse>("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id, question }),
    }),
};

export function streamJobLog(
  jobId: string,
  onLine: (line: string) => void,
  onEnd?: (status: string) => void,
): () => void {
  const es = new EventSource(`/api/train/jobs/${jobId}/stream`);
  es.onmessage = (e) => onLine(e.data);
  es.addEventListener("end", (e: MessageEvent) => {
    onEnd?.(e.data);
    es.close();
  });
  es.onerror = () => es.close();
  return () => es.close();
}
