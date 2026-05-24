import { useEffect, useState } from "react";
import { ModelInfo, api } from "../api/client";

export default function Evaluate() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelId, setModelId] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listModels().then((m) => {
      setModels(m);
      if (m.length > 0) setModelId(m[0].model_id);
    }).catch((e) => setError(e.message));
  }, []);

  async function ask() {
    if (!modelId || !question.trim()) return;
    setLoading(true);
    setError(null);
    setAnswer("");
    try {
      const res = await api.infer(modelId, question);
      setAnswer(res.answer);
      setElapsed(res.elapsed_ms);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">평가</h1>
      {error && <div className="rounded bg-red-50 p-4 text-red-700">{error}</div>}

      <section className="space-y-4 rounded border bg-white p-6">
        <label className="block">
          <span className="mb-1 block text-sm font-medium">모델</span>
          <select
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            className="w-full rounded border border-slate-300 px-3 py-2 text-sm"
          >
            {models.length === 0 && <option value="">(없음)</option>}
            {models.map((m) => (
              <option key={m.model_id} value={m.model_id}>
                {m.model_id}
              </option>
            ))}
          </select>
        </label>

        <label className="block">
          <span className="mb-1 block text-sm font-medium">질문</span>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            rows={3}
            className="w-full rounded border border-slate-300 px-3 py-2 text-sm"
            placeholder="예: 전세권이란 무엇인가요?"
          />
        </label>

        <button
          onClick={ask}
          disabled={loading || !modelId || !question.trim()}
          className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
        >
          {loading ? "생성 중..." : "답변 받기"}
        </button>
      </section>

      {answer && (
        <section className="rounded border bg-white p-6">
          <div className="mb-2 text-xs text-slate-500">{elapsed} ms</div>
          <p className="whitespace-pre-wrap text-sm leading-relaxed">{answer}</p>
        </section>
      )}
    </div>
  );
}
