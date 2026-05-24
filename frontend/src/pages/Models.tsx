import { useEffect, useState } from "react";
import { ModelInfo, api } from "../api/client";

export default function Models() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  function refresh() {
    api.listModels().then(setModels).catch((e) => setError(e.message));
  }

  useEffect(refresh, []);

  async function remove(id: string) {
    if (!confirm(`모델 ${id}를 삭제할까요?`)) return;
    try {
      await api.deleteModel(id);
      refresh();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">모델</h1>
      {error && <div className="rounded bg-red-50 p-4 text-red-700">{error}</div>}

      {models.length === 0 ? (
        <p className="text-sm text-slate-500">저장된 모델이 없습니다.</p>
      ) : (
        <table className="w-full overflow-hidden rounded border bg-white text-sm">
          <thead className="bg-slate-100 text-left">
            <tr>
              <th className="px-4 py-2">Model ID</th>
              <th className="px-4 py-2">Created</th>
              <th className="px-4 py-2">Size</th>
              <th className="px-4 py-2"></th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {models.map((m) => (
              <tr key={m.model_id}>
                <td className="px-4 py-2 font-mono">{m.model_id}</td>
                <td className="px-4 py-2 text-slate-500">{m.created_at}</td>
                <td className="px-4 py-2 text-slate-500">
                  {(m.size_bytes / 1024 / 1024).toFixed(1)} MB
                </td>
                <td className="px-4 py-2 text-right">
                  <button
                    onClick={() => remove(m.model_id)}
                    className="text-red-600 hover:underline"
                  >
                    삭제
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
