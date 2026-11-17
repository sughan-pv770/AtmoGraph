"""
Predictor service for inference: loads the trained RippleGNN (or falls back
to the deterministic ripple simulator if torch or weights are not yet present).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ml.features import ARRIVAL_SCALE, NODE_FEATURES, node_features, static_tensors
from ml.graph import SupplyGraph
from ml.model import RippleGNN
from ml.simulate import AFFECTED_DELAY, NOT_REACHED, RippleResult, simulate

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = ROOT / "ml" / "artifacts" / "ripple_gnn.pt"

_LOADED_MODEL: RippleGNN | None = None
_CACHED_GRAPH_ID: int | None = None
_CACHED_STATIC: dict[str, Any] | None = None


def get_model() -> RippleGNN | None:
    global _LOADED_MODEL
    if _LOADED_MODEL is not None:
        return _LOADED_MODEL
    if not WEIGHTS_PATH.exists():
        return None
    try:
        import torch
        ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {"in_dim": len(NODE_FEATURES), "edge_dim": 3, "hidden": 64, "layers": 5})
        model = RippleGNN(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _LOADED_MODEL = model
        return _LOADED_MODEL
    except Exception as e:
        print(f"[Predictor] Warning: could not load GNN checkpoint: {e}")
        return None


def predict_ripple(g: SupplyGraph, seed_dict: dict[str, float]) -> dict[str, Any]:
    """
    Given graph `g` and seed nodes {node_id: severity}, return predictions for each node:
      - delay: expected delay days
      - risk: risk score [0..1]
      - arrival_day: arrival day of disruption
      - affected: boolean flag
    """
    global _CACHED_GRAPH_ID, _CACHED_STATIC
    model = get_model()
    
    # Map node string IDs to integer indices
    idx_seeds = {g.index[nid]: float(sev) for nid, sev in seed_dict.items() if nid in g.index}

    if model is None:
        # Fallback to ground truth analytical ripple simulation
        res: RippleResult = simulate(g, idx_seeds)
        records = []
        for i, node in enumerate(g.nodes):
            d = round(float(res.delay[i]), 1)
            r = round(float(res.risk[i]), 2)
            arr = round(float(res.arrival[i]), 1) if res.arrival[i] < NOT_REACHED else None
            records.append({
                "id": node["id"],
                "name": node["name"],
                "stage": node["stage"],
                "label": node["label"],
                "predicted_delay": d,
                "risk": r,
                "arrival_day": arr,
                "is_seed": i in idx_seeds,
                "affected": d >= AFFECTED_DELAY
            })
        return {
            "engine": "analytical_simulator",
            "nodes": records,
            "summary": {
                "total_affected": sum(1 for r in records if r["affected"]),
                "max_delay_days": max((r["predicted_delay"] for r in records), default=0.0),
                "seed_count": len(idx_seeds)
            }
        }

    import torch
    gid = id(g)
    if _CACHED_GRAPH_ID != gid or _CACHED_STATIC is None:
        _CACHED_STATIC = static_tensors(g)
        _CACHED_GRAPH_ID = gid

    x = node_features(_CACHED_STATIC, idx_seeds)
    with torch.no_grad():
        out = model(x, _CACHED_STATIC["edge_index"], _CACHED_STATIC["edge_attr"])

    pred_delay = torch.expm1(out["delay"]).numpy()
    pred_risk = out["risk"].numpy()
    pred_arr = out["arrival"].numpy() * ARRIVAL_SCALE

    records = []
    for i, node in enumerate(g.nodes):
        raw_d = float(pred_delay[i])
        d = round(raw_d, 1) if raw_d >= AFFECTED_DELAY else 0.0
        r = round(float(pred_risk[i]), 2)
        arr = round(float(pred_arr[i]), 1) if d > 0 else None
        records.append({
            "id": node["id"],
            "name": node["name"],
            "stage": node["stage"],
            "label": node["label"],
            "predicted_delay": d,
            "risk": r,
            "arrival_day": arr,
            "is_seed": i in idx_seeds,
            "affected": d >= AFFECTED_DELAY
        })

    return {
        "engine": "ripple_gnn",
        "nodes": records,
        "summary": {
            "total_affected": sum(1 for r in records if r["affected"]),
            "max_delay_days": max((r["predicted_delay"] for r in records), default=0.0),
            "seed_count": len(idx_seeds)
        }
    }

