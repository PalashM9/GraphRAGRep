/**
 * PathExplorer.tsx – Find and explain paths between two graph nodes.
 */

import React, { useState } from "react";
import { fetchPath, GraphData, GraphNode, PathResponse } from "../api/client";

interface Props {
  graphData: GraphData;
  onHighlight: (ids: string[]) => void;
  onJumpToNode: (id: string) => void;
}

const PathExplorer: React.FC<Props> = ({ graphData, onHighlight, onJumpToNode }) => {
  const [fromSearch, setFromSearch] = useState("");
  const [toSearch, setToSearch] = useState("");
  const [fromNode, setFromNode] = useState<GraphNode | null>(null);
  const [toNode, setToNode] = useState<GraphNode | null>(null);
  const [fromSuggestions, setFromSuggestions] = useState<GraphNode[]>([]);
  const [toSuggestions, setToSuggestions] = useState<GraphNode[]>([]);
  const [pathResult, setPathResult] = useState<PathResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const NON_CHUNK = graphData.nodes.filter((n) => n.type !== "Chunk");

  const search = (query: string, setter: (nodes: GraphNode[]) => void) => {
    if (!query.trim()) { setter([]); return; }
    const q = query.toLowerCase();
    const matches = NON_CHUNK.filter(
      (n) => n.label?.toLowerCase().includes(q) || n.type?.toLowerCase().includes(q)
    ).slice(0, 8);
    setter(matches);
  };

  const handleFindPath = async () => {
    if (!fromNode || !toNode) return;
    setLoading(true);
    setError("");
    setPathResult(null);
    try {
      const res = await fetchPath(fromNode.id, toNode.id);
      setPathResult(res);
      onHighlight(res.path);
    } catch (e: any) {
      setError(e?.response?.data?.detail || "Path not found.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <p className="section-heading">Path Explorer</p>
      <p style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)",
        marginBottom: "var(--space-4)" }}>
        Find how two concepts are connected through the thesis.
      </p>

      {/* From node */}
      <NodePicker
        label="From"
        value={fromSearch}
        onChange={(v) => { setFromSearch(v); search(v, setFromSuggestions); }}
        suggestions={fromSuggestions}
        selected={fromNode}
        onSelect={(n) => { setFromNode(n); setFromSearch(n.label); setFromSuggestions([]); }}
        onClear={() => { setFromNode(null); setFromSearch(""); }}
      />

      {/* To node */}
      <NodePicker
        label="To"
        value={toSearch}
        onChange={(v) => { setToSearch(v); search(v, setToSuggestions); }}
        suggestions={toSuggestions}
        selected={toNode}
        onSelect={(n) => { setToNode(n); setToSearch(n.label); setToSuggestions([]); }}
        onClear={() => { setToNode(null); setToSearch(""); }}
      />

      <button
        className="btn btn-primary"
        onClick={handleFindPath}
        disabled={!fromNode || !toNode || loading}
        style={{ marginTop: "var(--space-3)", width: "100%",
          justifyContent: "center" }}
      >
        {loading ? "Finding path…" : "Find Path →"}
      </button>

      {error && (
        <p style={{ color: "var(--color-error)", fontSize: "var(--text-xs)",
          marginTop: "var(--space-3)" }}>
          {error}
        </p>
      )}

      {pathResult && !loading && (
        <div style={{ marginTop: "var(--space-4)" }}>
          {/* Visual path */}
          <p className="section-heading">Path ({pathResult.path.length} nodes)</p>
          <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center",
            gap: "var(--space-1)", marginBottom: "var(--space-4)" }}>
            {pathResult.nodes.map((n, i) => (
              <React.Fragment key={n.id}>
                <button
                  className={`badge badge-${n.type?.toLowerCase()}`}
                  style={{ cursor: "pointer", border: "none" }}
                  onClick={() => onJumpToNode(n.id)}
                >
                  {n.label?.length > 20 ? n.label.substring(0, 19) + "…" : n.label}
                </button>
                {i < pathResult.nodes.length - 1 && (
                  <span style={{ color: "var(--color-text-faint)", fontSize: "var(--text-xs)" }}>
                    {pathResult.edges[i]?.rel || "→"}
                  </span>
                )}
              </React.Fragment>
            ))}
          </div>

          {/* LLM Explanation */}
          <p className="section-heading">Explanation</p>
          <div className="card" style={{ fontSize: "var(--text-sm)", lineHeight: 1.7 }}>
            {pathResult.explanation}
          </div>
        </div>
      )}
    </div>
  );
};

interface NodePickerProps {
  label: string;
  value: string;
  onChange: (v: string) => void;
  suggestions: GraphNode[];
  selected: GraphNode | null;
  onSelect: (n: GraphNode) => void;
  onClear: () => void;
}

const NodePicker: React.FC<NodePickerProps> = ({
  label, value, onChange, suggestions, selected, onSelect, onClear,
}) => (
  <div style={{ marginBottom: "var(--space-3)", position: "relative" }}>
    <label style={{ fontSize: "var(--text-xs)", fontWeight: 600,
      color: "var(--color-text-muted)", display: "block",
      marginBottom: "var(--space-1)" }}>
      {label}
    </label>
    <div style={{ display: "flex", gap: "var(--space-2)" }}>
      <input
        className="input-field"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={`Search ${label.toLowerCase()} node…`}
        style={{ flex: 1 }}
      />
      {selected && (
        <button className="btn btn-ghost" onClick={onClear}
          style={{ padding: "var(--space-2)", flexShrink: 0 }}>
          ✕
        </button>
      )}
    </div>
    {selected && (
      <div style={{ marginTop: "var(--space-1)" }}>
        <span className={`badge badge-${selected.type?.toLowerCase()}`}>
          {selected.label}
        </span>
      </div>
    )}
    {suggestions.length > 0 && !selected && (
      <div style={{
        position: "absolute", top: "100%", left: 0, right: 0, zIndex: 20,
        background: "var(--color-surface-2)",
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-md)",
        boxShadow: "var(--shadow-md)",
        maxHeight: 200, overflowY: "auto",
      }}>
        {suggestions.map((n) => (
          <div
            key={n.id}
            onClick={() => onSelect(n)}
            style={{
              padding: "var(--space-2) var(--space-3)",
              cursor: "pointer",
              display: "flex", alignItems: "center", gap: "var(--space-2)",
              fontSize: "var(--text-xs)",
              transition: "background var(--transition)",
            }}
            onMouseEnter={(e) =>
              (e.currentTarget.style.background = "var(--color-surface-offset)")}
            onMouseLeave={(e) =>
              (e.currentTarget.style.background = "transparent")}
          >
            <span className={`badge badge-${n.type?.toLowerCase()}`}
              style={{ flexShrink: 0 }}>
              {n.type}
            </span>
            <span>{n.label}</span>
          </div>
        ))}
      </div>
    )}
  </div>
);

export default PathExplorer;