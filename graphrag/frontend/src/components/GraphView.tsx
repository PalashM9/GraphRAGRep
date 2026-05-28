/**
 * GraphView.tsx – Stable force-directed graph.
 *
 * Fixes applied:
 *  1. Removed d3-force require() — was causing OOM crash.
 *     Spacing handled by charge strength -300 instead.
 *  2. posMap ref captures x/y on every paint tick → reliable sidebar zoom.
 *  3. ResizeObserver sets exact canvas dimensions → no off-screen nodes.
 *  4. zoomToFit fires once after engine stops → good initial zoom level.
 *  5. Sidebar focus retries up to 15× until node positions are ready.
 *  6. Node click zooms in immediately using captured position.
 */

import React, {
  MutableRefObject,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import ForceGraph2D, { ForceGraphMethods } from "react-force-graph-2d";
import { GraphData, GraphNode } from "../api/client";

interface Props {
  graphData: GraphData;
  onSelectNode: (node: GraphNode) => void;
  selectedNodeId?: string;
  highlightNodes: Set<string>;
  focusNodeRef: MutableRefObject<((id: string) => void) | null>;
}

const NODE_COLORS: Record<string, string> = {
  Chapter: "#01696f",
  Section: "#7a39bb",
  Concept: "#437a22",
  Method:  "#964219",
  Metric:  "#d19900",
  Chunk:   "#bab9b4",
};

const NODE_RADIUS: Record<string, number> = {
  Chapter: 12,
  Section: 8,
  Concept: 6,
  Method:  6,
  Metric:  5,
  Chunk:   3,
};

const GraphView: React.FC<Props> = ({
  graphData,
  onSelectNode,
  selectedNodeId,
  highlightNodes,
  focusNodeRef,
}) => {
  const fgRef        = useRef<ForceGraphMethods>();
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ w: 800, h: 600 });

  // Stores last-known {x,y} for every node — updated on every paint tick
  const posMap = useRef<Record<string, { x: number; y: number }>>({});

  // ── Measure container so canvas fills exactly ─────────────────────────────
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setDimensions({ w: Math.floor(width), h: Math.floor(height) });
    });
    obs.observe(el);
    setDimensions({ w: el.clientWidth || 800, h: el.clientHeight || 600 });
    return () => obs.disconnect();
  }, []);

  // ── d3 force tuning — NO external d3 import, uses built-in forces ─────────
  useEffect(() => {
    if (!fgRef.current) return;
    // High repulsion = nodes pushed far apart, spacious layout
    fgRef.current.d3Force("charge")?.strength(-300);
    // Longer link distance = breathing room between connected nodes
    fgRef.current.d3Force("link")?.distance(90).strength(0.3);
    // Very gentle center pull so the graph doesn't drift off canvas
    fgRef.current.d3Force("center")?.strength(0.04);
  }, [graphData]);

  // ── Fit all nodes into view once simulation settles ───────────────────────
  const fittedRef = useRef(false);
  const handleEngineStop = useCallback(() => {
    if (fittedRef.current) return;
    fittedRef.current = true;
    // Small delay lets final positions stabilise before fitting
    setTimeout(() => {
      fgRef.current?.zoomToFit(500, 80);
    }, 150);
  }, []);

  // ── Capture node positions when user drags ────────────────────────────────
  const handleNodeDragEnd = useCallback((node: any) => {
    if (node.x != null) posMap.current[node.id] = { x: node.x, y: node.y };
  }, []);

  // ── Paint each node; spy on position to keep posMap current ──────────────
  const paintNode = useCallback(
    (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      // Always capture latest position
      if (node.x != null) posMap.current[node.id] = { x: node.x, y: node.y };

      const isSelected    = node.id === selectedNodeId;
      const isHighlighted = highlightNodes.has(node.id);
      const type          = (node.type as string) || "Chunk";
      const color         = NODE_COLORS[type] || "#888";
      const radius        = (NODE_RADIUS[type] || 5) * (isSelected ? 1.4 : 1);

      // Glow ring for selected / highlighted nodes
      if (isHighlighted || isSelected) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius + 5, 0, 2 * Math.PI);
        ctx.fillStyle = isSelected
          ? "rgba(1,105,111,0.22)"
          : "rgba(232,175,52,0.28)";
        ctx.fill();
      }

      // Main circle
      ctx.beginPath();
      ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();

      if (isSelected) {
        ctx.strokeStyle = "#fff";
        ctx.lineWidth   = 2;
        ctx.stroke();
      }

      // Labels: always for Chapter nodes; others appear when zoomed in (>2×)
      const showLabel = type === "Chapter" || globalScale > 2;
      if (showLabel) {
        const raw   = node.label || node.id;
        const label = raw.length > 28 ? raw.substring(0, 27) + "…" : raw;
        const fontSize = Math.max(
          type === "Chapter" ? 13 / globalScale : 10 / globalScale,
          4
        );
        ctx.font         = `${type === "Chapter" ? "bold " : ""}${fontSize}px Inter,sans-serif`;
        ctx.textAlign    = "center";
        ctx.textBaseline = "top";
        // White halo behind text for readability on any background
        ctx.shadowColor  = "rgba(255,255,255,0.95)";
        ctx.shadowBlur   = 4;
        ctx.fillStyle    = "#28251d";
        ctx.fillText(label, node.x, node.y + radius + 2);
        ctx.shadowBlur   = 0;
      }
    },
    [selectedNodeId, highlightNodes]
  );

  // ── Sidebar / external focus function ─────────────────────────────────────
  useEffect(() => {
    focusNodeRef.current = (id: string) => {
      const tryFocus = (attempts = 0) => {
        const pos = posMap.current[id];
        if (pos && fgRef.current) {
          // Centre on node then zoom in after pan animation completes
          fgRef.current.centerAt(pos.x, pos.y, 800);
          setTimeout(() => fgRef.current?.zoom(5, 600), 850);
        } else if (attempts < 15) {
          // Positions not ready yet (simulation still running) — retry
          setTimeout(() => tryFocus(attempts + 1), 200);
        } else {
          // Fallback: just fit everything in view
          fgRef.current?.zoomToFit(500, 80);
        }
      };
      tryFocus();
    };
  }, [focusNodeRef]);

  // ── Node click: select + zoom in ──────────────────────────────────────────
  const handleNodeClick = useCallback(
    (node: any) => {
      onSelectNode(node as GraphNode);
      if (fgRef.current && node.x != null) {
        fgRef.current.centerAt(node.x, node.y, 600);
        setTimeout(() => fgRef.current?.zoom(5, 500), 650);
      }
    },
    [onSelectNode]
  );

  // Convert to react-force-graph format
  const fgData = {
    nodes: graphData.nodes.map((n) => ({ ...n })),
    links: graphData.edges.map((e) => ({
      source: e.source,
      target: e.target,
      rel:    e.rel,
    })),
  };

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height: "100%", overflow: "hidden" }}
    >
      <ForceGraph2D
        ref={fgRef as any}
        graphData={fgData}
        width={dimensions.w}
        height={dimensions.h}
        nodeId="id"
        linkSource="source"
        linkTarget="target"
        // ── Simulation stability ──────────────────────────────────────
        cooldownTicks={300}
        cooldownTime={5000}
        d3AlphaDecay={0.025}
        d3VelocityDecay={0.45}
        onEngineStop={handleEngineStop}
        // ── Painting ─────────────────────────────────────────────────
        nodeCanvasObject={paintNode}
        nodeCanvasObjectMode={() => "replace"}
        linkLabel="rel"
        linkColor={() => "rgba(180,175,165,0.4)"}
        linkWidth={1}
        linkDirectionalArrowLength={4}
        linkDirectionalArrowRelPos={1}
        // ── Interaction ───────────────────────────────────────────────
        onNodeClick={handleNodeClick}
        onNodeDragEnd={handleNodeDragEnd}
        enableNodeDrag={true}
        enableZoomInteraction={true}
        enablePanInteraction={true}
        minZoom={0.2}
        maxZoom={10}
        backgroundColor="#f7f6f2"
      />
    </div>
  );
};

export default GraphView;