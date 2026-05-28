/**
 * client.ts – API helper functions for the GraphRAG backend.
 * All endpoints are relative to /api (proxied to http://localhost:8000).
 */

import axios from "axios";

const BASE = "";  // Uses the CRA proxy setting in package.json

export interface GraphNode {
  id: string;
  label: string;
  type: "Chapter" | "Section" | "Concept" | "Method" | "Metric" | "Chunk" | string;
  chapter_index?: number;
  section_index?: number;
  text_snippet?: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  rel: string;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface NodeDetail {
  node: GraphNode;
  neighbors: Array<{
    node_id: string;
    label: string;
    type: string;
    rel: string;
    direction: "in" | "out";
  }>;
  chunks: Array<{
    chunk_id: string;
    text: string;
    page: number;
    section: string;
  }>;
  explanation: string;
}

export interface ChunkUsed {
  chunk_id: string;
  text: string;
  page: number;
  chapter: string;
  section: string;
  graph_nodes: string[];
}

export interface QueryResponse {
  answer: string;
  nodes_used: string[];
  chunks_used: ChunkUsed[];
  graph_subgraph?: GraphData;
}

export interface PathResponse {
  path: string[];
  nodes: GraphNode[];
  edges: GraphEdge[];
  explanation: string;
}

export interface ChapterNode {
  id: string;
  label: string;
  type: string;
  chapter_index: number;
  sections: Array<{ id: string; label: string; type: string }>;
}

// ── API calls ────────────────────────────────────────────────────────────────

export const fetchGraph = (): Promise<GraphData> =>
  axios.get(`${BASE}/graph`).then((r) => r.data);

export const fetchChapters = (): Promise<ChapterNode[]> =>
  axios.get(`${BASE}/chapters`).then((r) => r.data);

export const fetchNode = (nodeId: string): Promise<NodeDetail> =>
  axios.get(`${BASE}/node/${encodeURIComponent(nodeId)}`).then((r) => r.data);

export const fetchPath = (from: string, to: string): Promise<PathResponse> =>
  axios
    .get(`${BASE}/path`, { params: { from_node: from, to_node: to } })
    .then((r) => r.data);

export const postQuery = (
  question: string,
  topKChunks = 10
): Promise<QueryResponse> =>
  axios
    .post(`${BASE}/query`, { question, top_k_chunks: topKChunks })
    .then((r) => r.data);