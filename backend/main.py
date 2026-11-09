"""
AtmoGraph FastAPI Backend (Week 1 - Graph Foundations & Supply Chain API).

Serves the global multi-hop supply chain graph stored in Neo4j:
  Supplier (Tier 2) -> Supplier (Tier 1) -> Manufacturer -> Port -> Distributor

Automatically uses live Neo4j when running, and gracefully falls back to the
deterministic supply-chain generator (`generate_graph.py`) if Neo4j Desktop is stopped.
"""
from __future__ import annotations

from collections import deque
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from backend.db import db
from generate_graph import generate as generate_mock_graph


def _stage_and_subtype(label: str, props: dict[str, Any]) -> tuple[int, str]:
    """Map node label + tier into an ordered supply-chain stage (0..4)."""
    if label == "Supplier":
        tier = int(props.get("tier", 1))
        if tier == 2:
            return 0, "Tier-2 Supplier"
        return 1, "Tier-1 Supplier"
    if label == "Manufacturer":
        return 2, "Manufacturer"
    if label == "Port":
        return 3, "Port"
    if label == "Distributor":
        return 4, "Distributor"
    return 2, label


def _serialize_node(label: str, props: dict[str, Any]) -> dict[str, Any]:
    stage, sub_type = _stage_and_subtype(label, props)
    return {
        "id": props["id"],
        "name": props.get("name", props["id"]),
        "label": label,
        "sub_type": sub_type,
        "stage": stage,
        "tier": props.get("tier"),
        "country": props.get("country"),
        "region": props.get("region"),
        "industry": props.get("industry"),
        "risk": float(props.get("risk", 0.0)),
    }


def _fallback_dataset() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Build serialized nodes and links from generate_graph.generate() when Neo4j is offline."""
    raw_nodes, raw_edges = generate_mock_graph()
    nodes_by_id: dict[str, dict[str, Any]] = {}
    for label, items in raw_nodes.items():
        for props in items:
            nodes_by_id[props["id"]] = _serialize_node(label, props)

    links: list[dict[str, Any]] = []
    for rel, _sl, _dl, src_id, dst_id, props in raw_edges:
        days = props.get("lead_days", props.get("transit_days", 0))
        links.append(
            {
                "id": f"{src_id}->{dst_id}",
                "source": src_id,
                "target": dst_id,
                "type": rel,
                "weight": float(props.get("weight", 0.5)),
                "days": int(days),
                "lead_days": props.get("lead_days"),
                "transit_days": props.get("transit_days"),
            }
        )
    return nodes_by_id, links


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        yield
    finally:
        db.close()


app = FastAPI(
    title="AtmoGraph Supply Chain API",
    description="Global Supply Chain Ripple Effect Predictor — Neo4j Graph API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    """Verify API and Neo4j database connectivity."""
    try:
        rows = db.query("RETURN 1 AS ok")
        connected = bool(rows and rows[0]["ok"] == 1)
        return {
            "status": "ok",
            "neo4j_connected": connected,
            "data_source": "neo4j" if connected else "fallback",
        }
    except (ServiceUnavailable, Neo4jError, OSError) as exc:
        return {
            "status": "ok",
            "neo4j_connected": False,
            "data_source": "fallback",
            "message": str(exc),
        }


@app.get("/api/stats")
def get_stats() -> dict[str, Any]:
    """Return summary KPIs across the global supply chain network."""
    try:
        node_rows = db.query(
            """
            MATCH (n)
            RETURN labels(n)[0] AS label,
                   n.tier AS tier,
                   count(*) AS cnt,
                   sum(CASE WHEN coalesce(n.risk, 0.0) >= 0.3 THEN 1 ELSE 0 END) AS at_risk
            """
        )
        edge_rows = db.query(
            "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS cnt"
        )
        meta_rows = db.query(
            """
            MATCH (n)
            RETURN collect(DISTINCT n.country) AS countries,
                   collect(DISTINCT n.industry) AS industries,
                   collect(DISTINCT n.region) AS regions
            """
        )

        counts = {
            "tier2_suppliers": 0,
            "tier1_suppliers": 0,
            "manufacturers": 0,
            "ports": 0,
            "distributors": 0,
            "total_nodes": 0,
            "at_risk_nodes": 0,
        }
        for r in node_rows:
            cnt = int(r["cnt"])
            counts["total_nodes"] += cnt
            counts["at_risk_nodes"] += int(r["at_risk"])
            lbl = r["label"]
            if lbl == "Supplier" and r["tier"] == 2:
                counts["tier2_suppliers"] += cnt
            elif lbl == "Supplier":
                counts["tier1_suppliers"] += cnt
            elif lbl == "Manufacturer":
                counts["manufacturers"] += cnt
            elif lbl == "Port":
                counts["ports"] += cnt
            elif lbl == "Distributor":
                counts["distributors"] += cnt

        relationships = {r["rel_type"]: int(r["cnt"]) for r in edge_rows}
        counts["total_edges"] = sum(relationships.values())
        meta = meta_rows[0] if meta_rows else {"countries": [], "industries": [], "regions": []}
        return {
            **counts,
            "relationships": relationships,
            "countries": sorted(c for c in meta["countries"] if c),
            "industries": sorted(i for i in meta["industries"] if i),
            "regions": sorted(reg for reg in meta["regions"] if reg),
            "data_source": "neo4j",
        }
    except (ServiceUnavailable, Neo4jError, OSError):
        nodes_by_id, links = _fallback_dataset()
        nodes = list(nodes_by_id.values())
        rels: dict[str, int] = {}
        for link in links:
            rels[link["type"]] = rels.get(link["type"], 0) + 1
        return {
            "tier2_suppliers": sum(1 for n in nodes if n["stage"] == 0),
            "tier1_suppliers": sum(1 for n in nodes if n["stage"] == 1),
            "manufacturers": sum(1 for n in nodes if n["stage"] == 2),
            "ports": sum(1 for n in nodes if n["stage"] == 3),
            "distributors": sum(1 for n in nodes if n["stage"] == 4),
            "total_nodes": len(nodes),
            "at_risk_nodes": sum(1 for n in nodes if n["risk"] >= 0.3),
            "total_edges": len(links),
            "relationships": rels,
            "countries": sorted({n["country"] for n in nodes if n["country"]}),
            "industries": sorted({n["industry"] for n in nodes if n["industry"]}),
            "regions": sorted({n["region"] for n in nodes if n["region"]}),
            "data_source": "fallback",
        }


@app.get("/api/graph")
def get_graph(
    country: str | None = None,
    industry: str | None = None,
) -> dict[str, Any]:
    """Return full or filtered supply chain nodes and directed edges for D3 visualization."""
    try:
        node_records = db.query(
            """
            MATCH (n)
            WHERE ($country IS NULL OR n.country = $country)
              AND ($industry IS NULL OR n.industry = $industry)
            RETURN labels(n)[0] AS label, properties(n) AS props
            ORDER BY props.id
            """,
            {"country": country, "industry": industry},
        )
        nodes = [_serialize_node(r["label"], r["props"]) for r in node_records]
        node_ids = {n["id"] for n in nodes}

        edge_records = db.query(
            """
            MATCH (a)-[r]->(b)
            RETURN a.id AS source,
                   b.id AS target,
                   type(r) AS rel_type,
                   properties(r) AS props
            """
        )
        links = []
        for r in edge_records:
            if r["source"] in node_ids and r["target"] in node_ids:
                props = r["props"]
                days = props.get("lead_days", props.get("transit_days", 0))
                links.append(
                    {
                        "id": f"{r['source']}->{r['target']}",
                        "source": r["source"],
                        "target": r["target"],
                        "type": r["rel_type"],
                        "weight": float(props.get("weight", 0.5)),
                        "days": int(days),
                        "lead_days": props.get("lead_days"),
                        "transit_days": props.get("transit_days"),
                    }
                )
        return {"nodes": nodes, "links": links, "data_source": "neo4j"}
    except (ServiceUnavailable, Neo4jError, OSError):
        nodes_by_id, all_links = _fallback_dataset()
        nodes = [
            n
            for n in nodes_by_id.values()
            if (country is None or n["country"] == country)
            and (industry is None or n["industry"] == industry)
        ]
        keep = {n["id"] for n in nodes}
        links = [l for l in all_links if l["source"] in keep and l["target"] in keep]
        return {"nodes": nodes, "links": links, "data_source": "fallback"}


@app.get("/api/nodes/{node_id}")
def get_node_details(node_id: str) -> dict[str, Any]:
    """Return detailed metadata, direct neighbors, and multi-hop downstream ripple impact."""
    try:
        rows = db.query(
            "MATCH (n {id: $node_id}) RETURN labels(n)[0] AS label, properties(n) AS props",
            {"node_id": node_id},
        )
        if not rows:
            raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

        node = _serialize_node(rows[0]["label"], rows[0]["props"])

        upstream_rows = db.query(
            """
            MATCH (u)-[r]->(n {id: $node_id})
            RETURN labels(u)[0] AS label, properties(u) AS props, type(r) AS rel_type, properties(r) AS rprops
            ORDER BY u.id
            """,
            {"node_id": node_id},
        )
        downstream_rows = db.query(
            """
            MATCH (n {id: $node_id})-[r]->(d)
            RETURN labels(d)[0] AS label, properties(d) AS props, type(r) AS rel_type, properties(r) AS rprops
            ORDER BY d.id
            """,
            {"node_id": node_id},
        )
        ripple_rows = db.query(
            """
            MATCH path = (n {id: $node_id})-[*1..4]->(d)
            WITH d,
                 labels(d)[0] AS label,
                 min(length(path)) AS hops,
                 min(reduce(total = 0, rel IN relationships(path) |
                     total + coalesce(rel.lead_days, rel.transit_days, 0))) AS min_days
            RETURN label, properties(d) AS props, hops, min_days
            ORDER BY hops ASC, min_days ASC, props.id ASC
            """,
            {"node_id": node_id},
        )

        upstream = [
            {
                **_serialize_node(r["label"], r["props"]),
                "rel_type": r["rel_type"],
                "days": int(r["rprops"].get("lead_days", r["rprops"].get("transit_days", 0))),
                "weight": float(r["rprops"].get("weight", 0.5)),
            }
            for r in upstream_rows
        ]
        downstream = [
            {
                **_serialize_node(r["label"], r["props"]),
                "rel_type": r["rel_type"],
                "days": int(r["rprops"].get("lead_days", r["rprops"].get("transit_days", 0))),
                "weight": float(r["rprops"].get("weight", 0.5)),
            }
            for r in downstream_rows
        ]
        ripple_impact = [
            {
                **_serialize_node(r["label"], r["props"]),
                "hops": int(r["hops"]),
                "cumulative_days": int(r["min_days"]),
            }
            for r in ripple_rows
        ]
        return {
            "node": node,
            "upstream": upstream,
            "downstream": downstream,
            "ripple_impact": ripple_impact,
            "data_source": "neo4j",
        }
    except (ServiceUnavailable, Neo4jError, OSError):
        nodes_by_id, links = _fallback_dataset()
        if node_id not in nodes_by_id:
            raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
        node = nodes_by_id[node_id]
        upstream = [
            {
                **nodes_by_id[l["source"]],
                "rel_type": l["type"],
                "days": l["days"],
                "weight": l["weight"],
            }
            for l in links
            if l["target"] == node_id
        ]
        downstream = [
            {
                **nodes_by_id[l["target"]],
                "rel_type": l["type"],
                "days": l["days"],
                "weight": l["weight"],
            }
            for l in links
            if l["source"] == node_id
        ]

        adj: dict[str, list[dict[str, Any]]] = {}
        for l in links:
            adj.setdefault(l["source"], []).append(l)

        best_hops: dict[str, int] = {}
        best_days: dict[str, int] = {}
        q: deque[tuple[str, int, int]] = deque([(node_id, 0, 0)])
        while q:
            cur, hops, cdays = q.popleft()
            if hops >= 4:
                continue
            for edge in adj.get(cur, []):
                nxt = edge["target"]
                nhops = hops + 1
                ndays = cdays + edge["days"]
                if nxt not in best_hops or nhops < best_hops[nxt] or (
                    nhops == best_hops[nxt] and ndays < best_days[nxt]
                ):
                    best_hops[nxt] = nhops
                    best_days[nxt] = ndays
                    q.append((nxt, nhops, ndays))

        ripple_impact = sorted(
            [
                {
                    **nodes_by_id[nid],
                    "hops": best_hops[nid],
                    "cumulative_days": best_days[nid],
                }
                for nid in best_hops
            ],
            key=lambda x: (x["hops"], x["cumulative_days"], x["id"]),
        )
        return {
            "node": node,
            "upstream": upstream,
            "downstream": downstream,
            "ripple_impact": ripple_impact,
            "data_source": "fallback",
        }
