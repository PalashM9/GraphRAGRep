"""
vector_store.py – FAISS-backed (or in-memory fallback) vector store.
Swap for Chroma/Weaviate/Pinecone by keeping the same interface.
"""
from __future__ import annotations
import logging
import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ingest import TextChunk

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    chunk_id: str
    text: str
    page: int
    chapter: str
    section: str
    section_index: str
    graph_nodes: list[str]
    score: float = 0.0


class VectorStore:
    """FAISS index with chunk metadata. Falls back to cosine numpy if FAISS unavailable."""

    def __init__(self):
        self._chunks: list["TextChunk"] = []
        self._embeddings: np.ndarray | None = None
        self._faiss_index = None
        self._use_faiss = False

    def build(self, chunks: list["TextChunk"]) -> None:
        self._chunks = chunks
        embeddings = [c.embedding for c in chunks if c.embedding is not None]
        if not embeddings:
            logger.warning("No embeddings found; vector search disabled.")
            return

        self._embeddings = np.vstack(embeddings).astype("float32")
        dim = self._embeddings.shape[1]

        try:
            import faiss
            index = faiss.IndexFlatIP(dim)  # inner product (cosine after normalisation)
            # L2 normalise for cosine similarity
            norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            normed = self._embeddings / norms
            index.add(normed)
            self._faiss_index = index
            self._use_faiss = True
            logger.info(f"FAISS index built: {len(chunks)} vectors, dim={dim}")
        except ImportError:
            logger.info("FAISS not installed; using numpy cosine fallback.")
            self._use_faiss = False

    def query(self, query_embedding: np.ndarray, k: int = 10) -> list[ChunkResult]:
        if self._embeddings is None or len(self._chunks) == 0:
            return []
        k = min(k, len(self._chunks))
        query_embedding = query_embedding.astype("float32")

        if self._use_faiss:
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
            scores, indices = self._faiss_index.search(query_embedding.reshape(1, -1), k)
            scores = scores[0].tolist()
            indices = indices[0].tolist()
        else:
            # Numpy cosine
            norms = np.linalg.norm(self._embeddings, axis=1)
            norms = np.where(norms == 0, 1, norms)
            q_norm = np.linalg.norm(query_embedding)
            if q_norm > 0:
                query_embedding = query_embedding / q_norm
            sims = (self._embeddings / norms[:, None]) @ query_embedding
            top_k = np.argsort(-sims)[:k]
            indices = top_k.tolist()
            scores = sims[top_k].tolist()

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self._chunks):
                continue
            c = self._chunks[idx]
            results.append(ChunkResult(
                chunk_id=c.chunk_id,
                text=c.text,
                page=c.page,
                chapter=c.chapter,
                section=c.section,
                section_index=c.section_index,
                graph_nodes=c.graph_nodes,
                score=float(score),
            ))
        return results

    def chunks_for_nodes(self, node_ids: list[str], limit: int = 15) -> list[ChunkResult]:
        """Retrieve chunks that are linked to any of the given node IDs."""
        results = []
        seen = set()
        node_set = set(node_ids)
        for c in self._chunks:
            if c.chunk_id in seen:
                continue
            node_overlap = set(c.graph_nodes) & node_set
            if node_overlap:
                results.append(ChunkResult(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    page=c.page,
                    chapter=c.chapter,
                    section=c.section,
                    section_index=c.section_index,
                    graph_nodes=c.graph_nodes,
                    score=float(len(node_overlap)),
                ))
                seen.add(c.chunk_id)
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]
