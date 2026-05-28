/**
 * QueryPanel.tsx – Natural-language Q&A powered by GraphRAG.
 */

import React, { useState } from "react";
import { postQuery, QueryResponse } from "../api/client";

interface Props {
  onHighlight: (ids: string[]) => void;
  onJumpToNode: (id: string) => void;
}

const EXAMPLE_QUESTIONS = [
  "How does preprocessing affect ASR performance?",
  "What evaluation metrics are used for Whisper fine-tuning?",
  "How does synthetic audio generation improve CER?",
  "What is the LLM post-processing pipeline?",
];

const QueryPanel: React.FC<Props> = ({ onHighlight, onJumpToNode }) => {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (q?: string) => {
    const text = q || question;
    if (!text.trim()) return;
    setQuestion(text);
    setLoading(true);
    setError("");
    setResponse(null);
    try {
      const res = await postQuery(text);
      setResponse(res);
      onHighlight(res.nodes_used);
    } catch (e: any) {
      setError(e?.response?.data?.detail || "Query failed. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <p className="section-heading">Ask the Thesis</p>

      {/* Example questions */}
      <div style={{ marginBottom: "var(--space-3)", display: "flex",
        flexWrap: "wrap", gap: "var(--space-2)" }}>
        {EXAMPLE_QUESTIONS.map((q) => (
          <button key={q} className="btn btn-ghost"
            style={{ fontSize: "var(--text-xs)", padding: "2px 8px" }}
            onClick={() => handleSubmit(q)}>
            {q.length > 30 ? q.substring(0, 29) + "…" : q}
          </button>
        ))}
      </div>

      {/* Input */}
      <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
        <textarea
          className="input-field"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about the thesis…"
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSubmit();
            }
          }}
        />
        <button
          className="btn btn-primary"
          onClick={() => handleSubmit()}
          disabled={loading || !question.trim()}
          style={{ alignSelf: "flex-end" }}
        >
          {loading ? "Thinking…" : "Ask ↵"}
        </button>
      </div>

      {error && (
        <div style={{ color: "var(--color-error)", fontSize: "var(--text-xs)",
          marginTop: "var(--space-3)" }}>
          {error}
        </div>
      )}

      {/* Loading skeleton */}
      {loading && (
        <div style={{ marginTop: "var(--space-4)" }}>
          {[200, 180, 220, 160, 140].map((w, i) => (
            <div key={i} className="loading-shimmer"
              style={{ height: 12, width: w, marginBottom: 10 }} />
          ))}
        </div>
      )}

      {/* Answer */}
      {response && !loading && (
        <div style={{ marginTop: "var(--space-4)" }}>
          <div className="card" style={{ fontSize: "var(--text-sm)",
            lineHeight: 1.7, whiteSpace: "pre-wrap" }}>
            {response.answer}
          </div>

          {/* Nodes used */}
          {response.nodes_used.length > 0 && (
            <div style={{ marginTop: "var(--space-4)" }}>
              <p className="section-heading">Nodes Used ({response.nodes_used.length})</p>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--space-2)" }}>
                {response.nodes_used.slice(0, 20).map((id) => (
                  <button
                    key={id}
                    className="badge badge-concept"
                    style={{ cursor: "pointer", border: "none" }}
                    onClick={() => onJumpToNode(id)}
                    title={id}
                  >
                    {id.replace(/^(chapter|section|concept|method|metric)_/, "")
                       .replace(/_/g, " ").substring(0, 20)}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Chunks used */}
          {response.chunks_used.length > 0 && (
            <div style={{ marginTop: "var(--space-4)" }}>
              <p className="section-heading">
                Sources ({response.chunks_used.length})
              </p>
              {response.chunks_used.map((chunk) => (
                <div key={chunk.chunk_id} className="card"
                  style={{ fontSize: "var(--text-xs)", lineHeight: 1.6,
                    marginBottom: "var(--space-2)" }}>
                  <p style={{ color: "var(--color-text-muted)",
                    marginBottom: "var(--space-1)", fontWeight: 600 }}>
                    {chunk.section || chunk.chapter} · p.{chunk.page}
                  </p>
                  <p>{chunk.text.substring(0, 220)}…</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default QueryPanel;