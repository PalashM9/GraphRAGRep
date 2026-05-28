
/**
 * App.tsx – Root layout: ChapterSidebar | GraphView | Right Panel (tabs)
 */

import React, { useCallback, useEffect, useRef, useState } from "react";
import ChapterSidebar from "./components/ChapterSidebar";
import GraphView from "./components/GraphView";
import NodeDetails from "./components/NodeDetails";
import PathExplorer from "./components/PathExplorer";
import QueryPanel from "./components/QueryPanel";
import { fetchGraph, fetchNode, GraphData, GraphNode, NodeDetail } from "./api/client";
import "./App.css";

type RightTab = "node" | "query" | "path";

const App: React.FC = () => {
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] });
  const [loading, setLoading] = useState(true);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [nodeDetail, setNodeDetail] = useState<NodeDetail | null>(null);
  const [nodeLoading, setNodeLoading] = useState(false);
  const [highlightNodes, setHighlightNodes] = useState<Set<string>>(new Set());
  const [rightTab, setRightTab] = useState<RightTab>("node");
  const focusNodeRef = useRef<((id: string) => void) | null>(null);

  // ── Load full graph on mount ─────────────────────────────────────────────
  useEffect(() => {
    fetchGraph()
      .then((data) => {
        // Filter out Chunk nodes for initial render (too many)
        const visible: GraphData = {
          nodes: data.nodes.filter((n) => n.type !== "Chunk"),
          edges: data.edges.filter(
            (e) =>
              !e.source.startsWith("chunk_") && !e.target.startsWith("chunk_")
          ),
        };
        setGraphData(visible);
      })
      .finally(() => setLoading(false));
  }, []);

  // ── Node selection ────────────────────────────────────────────────────────
  const handleSelectNode = useCallback((node: GraphNode) => {
    setSelectedNode(node);
    setRightTab("node");
    setNodeLoading(true);
    setNodeDetail(null);
    fetchNode(node.id)
      .then(setNodeDetail)
      .catch(console.error)
      .finally(() => setNodeLoading(false));
  }, []);

  // ── Highlight nodes (from query or path results) ─────────────────────────
  const handleHighlight = useCallback((ids: string[]) => {
    setHighlightNodes(new Set(ids));
  }, []);

  // ── Jump to node from sidebar or query results ───────────────────────────
  const handleJumpToNode = useCallback((id: string) => {
    const node = graphData.nodes.find((n) => n.id === id);
    if (node) handleSelectNode(node);
    if (focusNodeRef.current) focusNodeRef.current(id);
  }, [graphData.nodes, handleSelectNode]);

  return (
    <div className="app-root">
      {/* ── Header ──────────────────────────────────────────────────── */}
      <header className="app-header">
        <div className="header-logo">
          <svg
            width="28" height="28" viewBox="0 0 28 28"
            fill="none" aria-label="ThesisGraphRAG logo"
          >
            <circle cx="14" cy="14" r="13" stroke="#4f98a3" strokeWidth="2" />
            <circle cx="14" cy="8" r="3" fill="#4f98a3" />
            <circle cx="7" cy="20" r="3" fill="#6daa45" />
            <circle cx="21" cy="20" r="3" fill="#a86fdf" />
            <line x1="14" y1="11" x2="7" y2="17" stroke="#4f98a3" strokeWidth="1.5" />
            <line x1="14" y1="11" x2="21" y2="17" stroke="#4f98a3" strokeWidth="1.5" />
            <line x1="10" y1="20" x2="18" y2="20" stroke="#4f98a3" strokeWidth="1.5" />
          </svg>
          <span className="header-title">ASR Domain Thesis</span>
        </div>
        <div className="header-subtitle">
          Knowledge Graph · Vector Retrieval · LLM Answering
        </div>
        {loading && <div className="header-loading">Loading graph…</div>}
      </header>

      {/* ── Three-column layout ──────────────────────────────────────── */}
      <div className="app-body">
        {/* Left: Chapter sidebar */}
        <aside className="sidebar-left">
          <ChapterSidebar
            onSelectNode={handleJumpToNode}
            selectedNodeId={selectedNode?.id}
          />
        </aside>

        {/* Centre: Graph canvas */}
        <main className="graph-area">
          <GraphView
            graphData={graphData}
            onSelectNode={handleSelectNode}
            selectedNodeId={selectedNode?.id}
            highlightNodes={highlightNodes}
            focusNodeRef={focusNodeRef}
          />
        </main>

        {/* Right: Tabbed panel */}
        <aside className="sidebar-right">
          <div className="right-tabs">
            {(["node", "query", "path"] as RightTab[]).map((tab) => (
              <button
                key={tab}
                className={`tab-btn ${rightTab === tab ? "active" : ""}`}
                onClick={() => setRightTab(tab)}
              >
                {tab === "node" ? "Node" : tab === "query" ? "Q&A" : "Path"}
              </button>
            ))}
          </div>
          <div className="right-panel-content">
            {rightTab === "node" && (
              <NodeDetails
                node={selectedNode}
                detail={nodeDetail}
                loading={nodeLoading}
                onJumpToNode={handleJumpToNode}
              />
            )}
            {rightTab === "query" && (
              <QueryPanel
                onHighlight={handleHighlight}
                onJumpToNode={handleJumpToNode}
              />
            )}
            {rightTab === "path" && (
              <PathExplorer
                graphData={graphData}
                onHighlight={handleHighlight}
                onJumpToNode={handleJumpToNode}
              />
            )}
          </div>
        </aside>
      </div>
    </div>
  );
};

export default App;