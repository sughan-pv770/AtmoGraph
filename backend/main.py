"""
AtmoGraph FastAPI Backend (Weeks 1-4 Full Delivery):
- Week 1: Neo4j Graph Queries, KPIs, Multi-hop Path Traversal
- Week 2: NLP News Ingestion, Entity Extraction, Node Risk State Updates
- Week 3: Graph Neural Network (GNN) Ripple Prediction & Node Regression
- Week 4: Real-time WebSocket Streaming, Simulation Engine, Timeline Projection (30/60/90 days)
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.db import db
from backend.nlp import analyze
from generate_graph import generate as generate_mock_graph
from ml.graph import SupplyGraph, from_generator, from_records
from ml.predict import predict_ripple

ROOT_DIR = Path(__file__).resolve().parent.parent

# In-memory graph cache for fast GNN inference and WebSocket simulation
_ACTIVE_GRAPH: SupplyGraph | None = None


def get_current_supply_graph() -> SupplyGraph:
    global _ACTIVE_GRAPH
    if _ACTIVE_GRAPH is not None:
        return _ACTIVE_GRAPH

    # Try loading from live Neo4j first
    try:
        node_records = db.query("MATCH (n) RETURN labels(n)[0] AS label, properties(n) AS props")
        edge_records = db.query("MATCH (a)-[r]->(b) RETURN a.id AS source, b.id AS target, properties(r) AS props")
        if node_records and edge_records:
            _ACTIVE_GRAPH = from_records(node_records, edge_records)
            return _ACTIVE_GRAPH
    except Exception:
        pass

    # Fallback to deterministic generator
    raw_nodes, raw_edges = generate_mock_graph()
    _ACTIVE_GRAPH = from_generator(raw_nodes, raw_edges)
    return _ACTIVE_GRAPH


def reset_graph_cache() -> None:
    global _ACTIVE_GRAPH
    _ACTIVE_GRAPH = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_current_supply_graph()
    try:
        yield
    finally:
        db.close()


app = FastAPI(
    title="AtmoGraph Global Supply Chain API",
    description="Multi-hop Graph Neural Network & NLP Ripple Effect Predictor",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Schemas ---

class NewsAnalysisRequest(BaseModel):
    text: str
    apply_to_graph: bool = False


class DisruptionScenarioRequest(BaseModel):
    seeds: dict[str, float]  # {node_id: severity 0.0 - 1.0}


# --- REST Endpoints ---

@app.get("/api/health")
def health_check() -> dict[str, Any]:
    try:
        rows = db.query("RETURN 1 AS ok")
        connected = bool(rows and rows[0]["ok"] == 1)
        return {
            "status": "ok",
            "neo4j_connected": connected,
            "data_source": "neo4j" if connected else "in_memory_graph",
        }
    except Exception as exc:
        return {
            "status": "ok",
            "neo4j_connected": False,
            "data_source": "in_memory_graph",
            "message": str(exc),
        }


@app.get("/api/stats")
def get_stats() -> dict[str, Any]:
    g = get_current_supply_graph()
    counts = {
        "tier2_suppliers": sum(1 for n in g.nodes if n["stage"] == 0),
        "tier1_suppliers": sum(1 for n in g.nodes if n["stage"] == 1),
        "manufacturers": sum(1 for n in g.nodes if n["stage"] == 2),
        "ports": sum(1 for n in g.nodes if n["stage"] == 3),
        "distributors": sum(1 for n in g.nodes if n["stage"] == 4),
        "total_nodes": g.num_nodes,
        "total_edges": g.num_edges,
        "at_risk_nodes": sum(1 for n in g.nodes if n["risk"] >= 0.3),
        "countries": sorted({n["country"] for n in g.nodes if n.get("country")}),
        "industries": sorted({n["industry"] for n in g.nodes if n.get("industry")}),
        "regions": sorted({n["region"] for n in g.nodes if n.get("region")}),
        "data_source": "neo4j" if db._driver is not None else "in_memory_graph",
    }
    return counts


@app.get("/api/graph")
def get_graph(country: str | None = None, industry: str | None = None) -> dict[str, Any]:
    g = get_current_supply_graph()
    filtered_nodes = [
        n for n in g.nodes
        if (country is None or n.get("country") == country)
        and (industry is None or n.get("industry") == industry)
    ]
    node_set = {n["id"] for n in filtered_nodes}

    links = []
    for s_idx, d_idx, w, d in zip(g.src, g.dst, g.weight, g.days):
        src_id, dst_id = g.nodes[s_idx]["id"], g.nodes[d_idx]["id"]
        if src_id in node_set and dst_id in node_set:
            links.append({
                "id": f"{src_id}->{dst_id}",
                "source": src_id,
                "target": dst_id,
                "weight": w,
                "days": d,
            })

    return {
        "nodes": filtered_nodes,
        "links": links,
        "data_source": "neo4j" if db._driver is not None else "in_memory_graph",
    }


@app.get("/api/nodes/{node_id}")
def get_node_details(node_id: str) -> dict[str, Any]:
    g = get_current_supply_graph()
    if node_id not in g.index:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

    v = g.index[node_id]
    node = g.nodes[v]

    upstream = [
        {
            **g.nodes[g.src[e]],
            "days": g.days[e],
            "weight": g.weight[e],
            "share": round(g.share(e), 2),
        }
        for e in g.in_edges[v]
    ]

    downstream = [
        {
            **g.nodes[g.dst[e]],
            "days": g.days[e],
            "weight": g.weight[e],
            "share": round(g.share(e), 2),
        }
        for e in g.out_edges[v]
    ]

    # Calculate multi-hop downstream reach
    ripple = []
    visited = {v}
    queue = [(v, 0, 0)]
    while queue:
        curr, hops, acc_days = queue.pop(0)
        if hops >= 4:
            continue
        for e in g.out_edges[curr]:
            nxt = g.dst[e]
            if nxt not in visited:
                visited.add(nxt)
                nxt_days = acc_days + g.days[e]
                ripple.append({
                    **g.nodes[nxt],
                    "hops": hops + 1,
                    "cumulative_days": nxt_days,
                })
                queue.append((nxt, hops + 1, nxt_days))

    return {
        "node": node,
        "upstream": upstream,
        "downstream": downstream,
        "ripple_impact": sorted(ripple, key=lambda x: (x["hops"], x["cumulative_days"])),
    }


# --- Week 2: NLP Endpoints ---

@app.get("/api/news/feed")
def get_sample_news() -> list[dict[str, Any]]:
    path = ROOT_DIR / "data" / "news_headlines.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/nlp/analyze")
def analyze_news_text(req: NewsAnalysisRequest) -> dict[str, Any]:
    g = get_current_supply_graph()
    result = analyze(req.text, g)

    if req.apply_to_graph and result["seeds"]:
        # Update node risks in memory
        for nid, sev in result["seeds"].items():
            if nid in g.index:
                g.nodes[g.index[nid]]["risk"] = sev
        
        # Try updating Neo4j directly if available
        try:
            db.query(
                """
                UNWIND $seeds AS s
                MATCH (n {id: s.id})
                SET n.risk = s.risk
                """,
                {"seeds": [{"id": k, "risk": v} for k, v in result["seeds"].items()]}
            )
        except Exception:
            pass

    return result


# --- Week 3: GNN Ripple Prediction ---

@app.post("/api/predict/ripple")
def predict_ripple_effect(req: DisruptionScenarioRequest) -> dict[str, Any]:
    g = get_current_supply_graph()
    return predict_ripple(g, req.seeds)


# --- Week 4: Real-time WebSocket Streaming ---

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Live streaming simulation: pushes state updates and ripple telemetry
    to connected React clients in real-time.
    """
    await websocket.accept()
    g = get_current_supply_graph()
    try:
        while True:
            # Listen for client triggers (or ping)
            data_text = await asyncio.wait_for(websocket.receive_text(), timeout=2.0)
            data = json.loads(data_text)
            action = data.get("action")

            if action == "disrupt":
                seeds = data.get("seeds", {})
                pred = predict_ripple(g, seeds)
                await websocket.send_json({
                    "type": "prediction_update",
                    "data": pred,
                })
            elif action == "timeline_project":
                days_forward = data.get("days", 30)
                # Filter nodes affected on or before days_forward
                seeds = data.get("seeds", {})
                pred = predict_ripple(g, seeds)
                active_nodes = [
                    n for n in pred["nodes"]
                    if n["affected"] and (n["arrival_day"] is None or n["arrival_day"] <= days_forward)
                ]
                await websocket.send_json({
                    "type": "timeline_update",
                    "days": days_forward,
                    "affected_count": len(active_nodes),
                    "nodes": active_nodes,
                })
    except asyncio.TimeoutError:
        # Keepalive heartbeat
        try:
            await websocket.send_json({"type": "heartbeat", "time": asyncio.get_event_loop().time()})
        except Exception:
            pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket Error]: {e}")
