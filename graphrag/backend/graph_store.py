"""
graph_store.py – NetworkX graph operations for the GraphRAG system.
Modular: swap for Neo4j/ArangoDB by keeping the same function signatures.
"""
from __future__ import annotations
import re
import networkx as nx


def get_node(g: nx.DiGraph, node_id: str) -> dict | None:
    if not g.has_node(node_id):
        return None
    data = dict(g.nodes[node_id])
    data["id"] = node_id
    return data


def get_neighbors(g: nx.DiGraph, node_id: str) -> list[dict]:
    results = []
    # Outgoing edges
    for _, tgt, data in g.out_edges(node_id, data=True):
        nd = g.nodes.get(tgt, {})
        results.append({
            "node_id": tgt,
            "label": nd.get("label", tgt),
            "type": nd.get("type", "?"),
            "rel": data.get("rel", "RELATED"),
            "direction": "out",
        })
    # Incoming edges
    for src, _, data in g.in_edges(node_id, data=True):
        nd = g.nodes.get(src, {})
        results.append({
            "node_id": src,
            "label": nd.get("label", src),
            "type": nd.get("type", "?"),
            "rel": data.get("rel", "RELATED"),
            "direction": "in",
        })
    return results


def find_path(g: nx.DiGraph, from_id: str, to_id: str) -> list[str]:
    try:
        return nx.shortest_path(g, from_id, to_id)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        # Try undirected fallback
        try:
            return nx.shortest_path(g.to_undirected(), from_id, to_id)
        except Exception:
            return []


def k_hop_subgraph(
    g: nx.DiGraph, seed_ids: list[str], k: int = 2
) -> tuple[list[dict], list[dict]]:
    """Return (nodes, edges) for the k-hop neighbourhood of seed_ids."""
    visited: set[str] = set()
    frontier = [nid for nid in seed_ids if g.has_node(nid)]
    for _ in range(k):
        next_frontier = []
        for nid in frontier:
            if nid in visited:
                continue
            visited.add(nid)
            for _, tgt in g.out_edges(nid):
                if tgt not in visited:
                    next_frontier.append(tgt)
            for src, _ in g.in_edges(nid):
                if src not in visited:
                    next_frontier.append(src)
        frontier = next_frontier

    all_nodes_in_subgraph = visited | set(frontier)
    nodes = []
    for nid in all_nodes_in_subgraph:
        if g.has_node(nid):
            d = dict(g.nodes[nid])
            d["id"] = nid
            nodes.append(d)

    edges = []
    for src, tgt, data in g.edges(data=True):
        if src in all_nodes_in_subgraph and tgt in all_nodes_in_subgraph:
            edges.append({"source": src, "target": tgt, "rel": data.get("rel", "RELATED")})

    return nodes, edges


def top_k_nodes_by_label(g: nx.DiGraph, query: str, k: int = 8) -> list[str]:
    """
    Find top-k non-Chunk nodes whose label/snippet overlaps with the query words.
    Uses simple token overlap score.
    """
    query_tokens = set(re.findall(r"\w+", query.lower()))
    query_tokens -= {"the", "a", "an", "of", "in", "and", "is", "to", "for", "on", "how"}
    scores: list[tuple[float, str]] = []
    for nid, data in g.nodes(data=True):
        if data.get("type") == "Chunk":
            continue
        label = (data.get("label", "") + " " + data.get("text_snippet", "")).lower()
        label_tokens = set(re.findall(r"\w+", label))
        overlap = len(query_tokens & label_tokens)
        if overlap > 0:
            scores.append((overlap, nid))
    scores.sort(reverse=True)
    return [nid for _, nid in scores[:k]]


def graph_to_json(g: nx.DiGraph) -> dict:
    nodes = []
    for nid, data in g.nodes(data=True):
        d = dict(data)
        d["id"] = nid
        nodes.append(d)
    edges = [
        {"source": s, "target": t, "rel": d.get("rel", "RELATED")}
        for s, t, d in g.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges}
