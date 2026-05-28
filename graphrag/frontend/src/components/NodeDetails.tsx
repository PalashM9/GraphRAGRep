/**
 * NodeDetails.tsx – Shows selected node information + LLM explanation.
 */

import React from "react";
import { GraphNode, NodeDetail } from "../api/client";

interface Props {
  node: GraphNode | null;
  detail: NodeDetail | null;
  loading: boolean;
  onJumpToNode: (id: string) => void;
}

const NodeDetails: React.FC<Props> = ({ node, detail, loading, onJumpToNode }) => {
  if (!node) {
    return (
      <div className="empty-state">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="1.5">
          <circle cx="12" cy="12" r="10"/>
          <circle cx="12" cy="10" r="3"/>
          <path d="M6.2 20C6.8 17.2 9.2 15 12 15s5.2 2.2 5.8 5"/>
        </svg>
        <p>Click any node in the graph to see details.</p>
      </div>
    );
  }

  const typeClass = `badge badge-${node.type?.toLowerCase()}`;

  return (
    <div>
      {/* Node header */}
      <div style={{ marginBottom: "var(--space-4)" }}>
        <span className={typeClass}>{node.type}</span>
        <h2 style={{ fontSize: "var(--text-base)", fontWeight: 700,
          marginTop: "var(--space-2)", lineHeight: 1.3 }}>
          {node.label}
        </h2>
        {node.text_snippet && (
          <p style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)",
            marginTop: "var(--space-2)", lineHeight: 1.5 }}>
            {node.text_snippet.substring(0, 180)}…
          </p>
        )}
      </div>

      {/* LLM Explanation */}
      <div style={{ marginBottom: "var(--space-6)" }}>
        <p className="section-heading">AI Explanation</p>
        {loading ? (
          <div>
            {[120, 200, 160].map((w, i) => (
              <div key={i} className="loading-shimmer"
                style={{ height: 14, width: w, marginBottom: 8 }} />
            ))}
          </div>
        ) : detail?.explanation ? (
          <div className="card" style={{ fontSize: "var(--text-sm)",
            lineHeight: 1.7, color: "var(--color-text)" }}>
            {detail.explanation}
          </div>
        ) : (
          <p style={{ color: "var(--color-text-muted)", fontSize: "var(--text-sm)" }}>
            No explanation available.
          </p>
        )}
      </div>

      {/* Neighbours grouped by edge type */}
      {detail?.neighbors && detail.neighbors.length > 0 && (
        <div style={{ marginBottom: "var(--space-6)" }}>
          <p className="section-heading">Connections</p>
          {groupByRel(detail.neighbors).map(([rel, items]) => (
            <div key={rel} style={{ marginBottom: "var(--space-3)" }}>
              <p style={{ fontSize: "var(--text-xs)", fontWeight: 600,
                color: "var(--color-text-muted)", marginBottom: "var(--space-1)",
                textTransform: "uppercase", letterSpacing: "0.05em" }}>
                {rel}
              </p>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--space-2)" }}>
                {items.map((nb) => (
                  <button
                    key={nb.node_id}
                    className={`badge badge-${nb.type.toLowerCase()}`}
                    onClick={() => onJumpToNode(nb.node_id)}
                    style={{ cursor: "pointer", border: "none" }}
                    title={nb.label}
                  >
                    {nb.label.length > 24 ? nb.label.substring(0, 23) + "…" : nb.label}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Relevant chunks */}
      {detail?.chunks && detail.chunks.length > 0 && (
        <div>
          <p className="section-heading">Relevant Excerpts</p>
          {detail.chunks.map((chunk) => (
            <div key={chunk.chunk_id} className="card"
              style={{ fontSize: "var(--text-xs)", lineHeight: 1.6 }}>
              <p style={{ color: "var(--color-text-muted)", marginBottom: "var(--space-1)" }}>
                {chunk.section} · p.{chunk.page}
              </p>
              <p>{chunk.text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

function groupByRel(
  neighbors: NodeDetail["neighbors"]
): [string, NodeDetail["neighbors"]][] {
  const map = new Map<string, NodeDetail["neighbors"]>();
  for (const nb of neighbors) {
    const key = nb.direction === "out" ? `→ ${nb.rel}` : `← ${nb.rel}`;
    if (!map.has(key)) map.set(key, []);
    map.get(key)!.push(nb);
  }
  return Array.from(map.entries());
}

export default NodeDetails;