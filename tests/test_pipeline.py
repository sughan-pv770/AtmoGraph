"""
Unit and integration tests for AtmoGraph:
- Graph generator integrity
- NLP entity and event extraction
- PyTorch Geometric GNN inference
- FastAPI REST endpoints
"""
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from generate_graph import generate
from ml.graph import from_generator
from ml.predict import predict_ripple
from backend.nlp import analyze

client = TestClient(app)


def test_graph_generation():
    nodes, edges = generate(n_t2=40, n_t1=30, n_mfr=15, seed=42)
    assert len(nodes["Supplier"]) == 70
    assert len(nodes["Manufacturer"]) == 15
    assert len(nodes["Port"]) == 12
    assert len(nodes["Distributor"]) == 6
    assert len(edges) > 100


def test_nlp_extraction():
    g = from_generator(*generate(n_t2=20, n_t1=15, n_mfr=5, seed=42))
    headline = "Dockworkers launch strike at Port of Rotterdam halting logistics"
    res = analyze(headline, g)
    assert "Rotterdam" in res["matched"]["ports"]
    assert res["event_type"] == "labor_strike"
    assert len(res["seeds"]) >= 1


def test_gnn_prediction():
    g = from_generator(*generate(n_t2=20, n_t1=15, n_mfr=5, seed=42))
    ports = [n["id"] for n in g.nodes if n["label"] == "Port"]
    assert len(ports) > 0
    pred = predict_ripple(g, {ports[0]: 0.9})
    assert "summary" in pred
    assert pred["summary"]["total_affected"] >= 1
    assert pred["summary"]["max_delay_days"] > 0


def test_api_health_and_stats():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    res_stats = client.get("/api/stats")
    assert res_stats.status_code == 200
    data = res_stats.json()
    assert data["total_nodes"] > 0
    assert data["total_edges"] > 0

