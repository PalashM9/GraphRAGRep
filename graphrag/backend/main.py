"""
main.py – FastAPI application with all GraphRAG endpoints.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Place your thesis PDF at: graphrag/backend/data/thesis.pdf
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graph_store import (
    get_node,
    get_neighbors,
    find_path,
    graph_to_json,
    k_hop_subgraph,
    top_k_nodes_by_label,
)
from ingest import APP_STATE, ingest
from llm import generate_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "thesis.pdf"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run ingestion on startup."""
    logger.info("Starting ingestion pipeline...")
    logger.info(f"BASE_DIR: {BASE_DIR}")
    logger.info(f"PDF_PATH: {PDF_PATH}")
    logger.info(f"PDF exists: {PDF_PATH.exists()}")
    logger.info(f"PDF is file: {PDF_PATH.is_file()}")
    logger.info(f"Data dir exists: {PDF_PATH.parent.exists()}")

    try:
        ingest(PDF_PATH, EMBEDDING_MODEL)
        logger.info("Ingestion completed successfully.")
        logger.info(f"Chapters loaded: {len(APP_STATE.chapter_tree) if APP_STATE.chapter_tree else 0}")
        logger.info(f"Graph nodes: {APP_STATE.graph.number_of_nodes() if APP_STATE.graph else 0}")
        logger.info(f"Graph edges: {APP_STATE.graph.number_of_edges() if APP_STATE.graph else 0}")
    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")

    yield
    logger.info("Shutting down.")


app = FastAPI(title="Thesis GraphRAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    top_k_chunks: int = 10
    hop_k: int = 2


class QueryResponse(BaseModel):
    answer: str
    nodes_used: list[str]
    chunks_used: list[dict]
    graph_subgraph: Optional[dict] = None


@app.get("/")
def root() -> dict:
    return {
        "message": "Thesis GraphRAG API is running",
        "pdf_path": str(PDF_PATH),
        "pdf_exists": PDF_PATH.exists(),
        "chapter_count": len(APP_STATE.chapter_tree) if APP_STATE.chapter_tree else 0,
    }


@app.get("/debug-pdf")
def debug_pdf() -> dict:
    return {
        "base_dir": str(BASE_DIR),
        "cwd": str(Path.cwd()),
        "pdf_path": str(PDF_PATH),
        "exists": PDF_PATH.exists(),
        "is_file": PDF_PATH.is_file(),
        "parent_exists": PDF_PATH.parent.exists(),
        "parent_contents": sorted([p.name for p in PDF_PATH.parent.iterdir()]) if PDF_PATH.parent.exists() else [],
    }


@app.get("/debug-state")
def debug_state() -> dict:
    return {
        "chapter_count": len(APP_STATE.chapter_tree) if APP_STATE.chapter_tree else 0,
        "chapters_preview": APP_STATE.chapter_tree[:3] if APP_STATE.chapter_tree else [],
        "graph_nodes": APP_STATE.graph.number_of_nodes() if APP_STATE.graph else 0,
        "graph_edges": APP_STATE.graph.number_of_edges() if APP_STATE.graph else 0,
        "has_vector_store": APP_STATE.vector_store is not None,
        "embedding_model_loaded": APP_STATE.embedding_model is not None,
    }


@app.get("/graph")
def get_graph() -> dict:
    """Return the full knowledge graph (nodes + edges)."""
    return graph_to_json(APP_STATE.graph)


@app.get("/chapters")
def get_chapters() -> list[dict]:
    """Return the chapter/section tree."""
    return APP_STATE.chapter_tree


@app.get("/node/{node_id}")
def get_node_detail(node_id: str) -> dict:
    """
    Return detailed information + LLM-generated explanation for a node.
    Implements GraphRAG: fetches relevant chunks and explains via LLM.
    """
    g = APP_STATE.graph
    node = get_node(g, node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found.")

    neighbors = get_neighbors(g, node_id)
    neighbor_labels = [n["label"] for n in neighbors[:10]]

    relevant_chunks = APP_STATE.vector_store.chunks_for_nodes([node_id], limit=5)
    chunk_texts = [c.text for c in relevant_chunks]

    prompt = _build_node_explanation_prompt(node, neighbor_labels, chunk_texts)
    explanation = generate_answer(prompt)

    return {
        "node": node,
        "neighbors": neighbors,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "text": c.text[:400],
                "page": c.page,
                "section": c.section,
            }
            for c in relevant_chunks
        ],
        "explanation": explanation,
    }


@app.get("/path")
def get_path(from_node: str, to_node: str) -> dict:
    """
    Find the shortest path between two nodes and explain it via LLM.
    Query params: from_node, to_node
    """
    g = APP_STATE.graph
    for nid in [from_node, to_node]:
        if not g.has_node(nid):
            raise HTTPException(status_code=404, detail=f"Node '{nid}' not found.")

    path_ids = find_path(g, from_node, to_node)
    if not path_ids:
        raise HTTPException(
            status_code=404,
            detail=f"No path found between '{from_node}' and '{to_node}'.",
        )

    path_nodes = []
    path_snippets = []
    for nid in path_ids:
        node_data = get_node(g, nid)
        if node_data:
            path_nodes.append(node_data)
            path_snippets.append(
                f"[{node_data.get('type','?')}] {node_data.get('label','?')}: "
                f"{node_data.get('text_snippet', '')[:150]}"
            )

    path_edges = []
    for i in range(len(path_ids) - 1):
        if g.has_edge(path_ids[i], path_ids[i + 1]):
            rel = g.edges[path_ids[i], path_ids[i + 1]].get("rel", "RELATED")
        else:
            rel = "RELATED"
        path_edges.append({"source": path_ids[i], "target": path_ids[i + 1], "rel": rel})

    prompt = _build_path_explanation_prompt(path_nodes, path_snippets)
    explanation = generate_answer(prompt)

    return {
        "path": path_ids,
        "nodes": path_nodes,
        "edges": path_edges,
        "explanation": explanation,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """
    GraphRAG QA endpoint.
    Step 1: Map question to graph nodes (keyword + embedding).
    Step 2: k-hop subgraph expansion.
    Step 3: Retrieve relevant text chunks guided by graph.
    Step 4: LLM answer generation.
    """
    g = APP_STATE.graph
    vs = APP_STATE.vector_store
    model = APP_STATE.embedding_model

    seed_ids = top_k_nodes_by_label(g, req.question, k=6)

    if model is not None:
        q_emb = list(model.embed([req.question]))[0].astype("float32")
        emb_results = vs.query(q_emb, k=req.top_k_chunks)
        for cr in emb_results[:4]:
            for nid in cr.graph_nodes:
                if nid not in seed_ids:
                    seed_ids.append(nid)

    subgraph_nodes, subgraph_edges = k_hop_subgraph(g, seed_ids, k=req.hop_k)
    subgraph_node_ids = [n["id"] for n in subgraph_nodes]

    candidate_chunks = vs.chunks_for_nodes(subgraph_node_ids, limit=req.top_k_chunks * 2)

    if model is not None and candidate_chunks:
        q_emb = list(model.embed([req.question]))[0].astype("float32")
        emb_top = vs.query(q_emb, k=req.top_k_chunks)
        emb_ids = {cr.chunk_id for cr in emb_top}
        merged: list = []
        for c in candidate_chunks:
            if c.chunk_id in emb_ids:
                merged.insert(0, c)
            else:
                merged.append(c)
        candidate_chunks = merged[: req.top_k_chunks]
    else:
        candidate_chunks = candidate_chunks[: req.top_k_chunks]

    prompt = _build_query_prompt(req.question, candidate_chunks, subgraph_nodes)
    answer = generate_answer(prompt)

    chunks_used = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text[:300],
            "page": c.page,
            "chapter": c.chapter,
            "section": c.section,
            "graph_nodes": c.graph_nodes,
        }
        for c in candidate_chunks
    ]

    return QueryResponse(
        answer=answer,
        nodes_used=subgraph_node_ids[:20],
        chunks_used=chunks_used,
        graph_subgraph={"nodes": subgraph_nodes[:30], "edges": subgraph_edges[:60]},
    )


def _build_query_prompt(question: str, chunks: list, relevant_nodes: list[dict]) -> str:
    node_summary = "\n".join(
        f"- [{n.get('type','?')}] {n.get('label','?')}"
        for n in relevant_nodes[:15]
        if n.get("type") != "Chunk"
    )
    chunk_context = "\n\n".join(
        f"[Source: {c.chapter or c.section}, Page {c.page}]\n{c.text}"
        for c in chunks
    )
    return f"""You are an expert research assistant analysing a Master's thesis on
ASR (Automatic Speech Recognition), TTS, and LLM pipelines.

QUESTION: {question}

RELEVANT KNOWLEDGE GRAPH NODES:
{node_summary}

RELEVANT THESIS EXCERPTS:
{chunk_context}

Based on the above excerpts and knowledge graph context, provide a clear, structured answer.
For each key claim, reference which chapter/section it comes from.
If the answer cannot be determined from the provided context, say so honestly."""


def _build_node_explanation_prompt(
    node: dict, neighbor_labels: list[str], chunk_texts: list[str]
) -> str:
    neighbors_str = ", ".join(neighbor_labels) if neighbor_labels else "none"
    chunks_str = "\n\n".join(chunk_texts[:3]) if chunk_texts else "No excerpts available."
    return f"""You are analysing a node in the knowledge graph of a Master's thesis on ASR/TTS/LLM.

NODE: "{node.get('label', '?')}" (type: {node.get('type', '?')})
CONNECTED TO: {neighbors_str}
THESIS SNIPPET: {node.get('text_snippet', '')}

RELEVANT EXCERPTS FROM THE THESIS:
{chunks_str}

Write 2–4 sentences explaining what this node represents in the context of the thesis,
what role it plays, and how it connects to other concepts."""


def _build_path_explanation_prompt(path_nodes: list[dict], snippets: list[str]) -> str:
    start = path_nodes[0].get("label", "?") if path_nodes else "?"
    end = path_nodes[-1].get("label", "?") if path_nodes else "?"
    path_str = " → ".join(n.get("label", "?") for n in path_nodes)
    snippets_str = "\n".join(snippets)
    return f"""You are analysing the conceptual path in a Master's thesis on ASR/TTS/LLM pipelines.

PATH: {path_str}

CONTEXT FOR EACH NODE:
{snippets_str}

Explain in 3–5 sentences how the thesis moves from "{start}" to "{end}" along this path.
Describe the logical or argumentative connection between each step."""