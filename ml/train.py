"""
Train and evaluate the RippleGNN.

  python -m ml.train                 # full run (~ several minutes on CPU)
  python -m ml.train --quick         # tiny smoke run

Data:  disruption scenarios sampled on many *different* synthetic networks
       (random sizes / seeds), labelled by ml.simulate.simulate().
Splits: train / val networks, a held-out set of unseen test networks, and the
       actual deployed AtmoGraph network (seed 42, 1,088 nodes).
Outputs:
  ml/artifacts/ripple_gnn.pt     model weights + config
  ml/artifacts/metrics.json      evaluation vs. baselines
  docs/figures/*.png             training curve, predicted-vs-true, per-stage error
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from generate_graph import DEFAULT_MFR, DEFAULT_T1, DEFAULT_T2, generate
from ml.features import ARRIVAL_SCALE, EDGE_FEATURES, NODE_FEATURES, node_features, static_tensors, targets
from ml.graph import SupplyGraph, from_generator
from ml.model import GraphMLP, RippleGNN
from ml.simulate import AFFECTED_DELAY, NOT_REACHED, OUTAGE_DAYS, sample_scenario, simulate

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "ml" / "artifacts"
FIGURES = ROOT / "docs" / "figures"
LABEL_NOISE = 0.05


# ----------------------------------------------------------------------------- data

def random_graph(seed: int, rng: random.Random) -> SupplyGraph:
    f = rng.uniform(0.05, 1.0)
    return from_generator(*generate(int(DEFAULT_T2 * f) + 10, int(DEFAULT_T1 * f) + 8,
                                    int(DEFAULT_MFR * f) + 4, seed=seed))


def build_samples(graphs: list[SupplyGraph], per_graph: int, seed: int) -> list[Data]:
    rng = random.Random(seed)
    out = []
    for g in graphs:
        st = static_tensors(g)
        for _ in range(per_graph):
            desc, seeds = sample_scenario(g, rng)
            res = simulate(g, seeds, noise=LABEL_NOISE, rng=rng)
            t = targets(res)
            out.append(Data(x=node_features(st, seeds), edge_index=st["edge_index"],
                            edge_attr=st["edge_attr"], stage=torch.tensor([n["stage"] for n in g.nodes]),
                            seed_sev=torch.tensor([seeds.get(i, 0.0) for i in range(g.num_nodes)]),
                            num_nodes=g.num_nodes, **t))
    return out


# ----------------------------------------------------------------------------- loss

def loss_fn(pred: dict[str, torch.Tensor], b: Data) -> torch.Tensor:
    affected = (b.y_delay > math.log1p(AFFECTED_DELAY)).float()
    w = 1.0 + 4.0 * affected
    l_delay = (w * (pred["delay"] - b.y_delay) ** 2).sum() / w.sum()
    l_risk = ((pred["risk"] - b.y_risk) ** 2).mean()
    m = b.arrival_mask
    l_arr = ((pred["arrival"][m] - b.y_arrival[m]) ** 2).mean() if m.any() else torch.tensor(0.0)
    return l_delay + 0.5 * l_risk + 0.5 * l_arr


def run_epoch(model, loader, opt=None) -> float:
    model.train(opt is not None)
    total, count = 0.0, 0
    for b in loader:
        with torch.set_grad_enabled(opt is not None):
            loss = loss_fn(model(b.x, b.edge_index, b.edge_attr), b)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        total += float(loss.detach()) * b.num_graphs
        count += b.num_graphs
    return total / max(count, 1)


# ----------------------------------------------------------------------------- baselines

def baseline_seeds_only(d: Data) -> dict[str, np.ndarray]:
    stage = d.stage.numpy()
    sev = d.seed_sev.numpy()
    delay = np.array([sev[i] * OUTAGE_DAYS[int(stage[i])] for i in range(len(sev))])
    return {"delay": delay, "risk": sev.copy(), "arrival": np.where(sev > 0, 0.0, NOT_REACHED)}


def baseline_hop_decay(d: Data, decay: float = 0.55) -> dict[str, np.ndarray]:
    """Classic heuristic: shock decays geometrically per hop, arrives via the fastest path."""
    n = d.num_nodes
    src, dst = d.edge_index.numpy()
    days = d.edge_attr[:, 2].numpy() * 45.0
    stage = d.stage.numpy()
    sev = d.seed_sev.numpy()
    out = [[] for _ in range(n)]
    for e in range(len(src)):
        out[src[e]].append(e)
    shock = np.zeros(n)
    arrival = np.full(n, NOT_REACHED)
    heap = []
    for i in np.nonzero(sev)[0]:
        shock[i] = sev[i] * OUTAGE_DAYS[int(stage[i])]
        arrival[i] = 0.0
        heapq.heappush(heap, (0.0, int(i)))
    while heap:  # Dijkstra on transit days for arrival
        t, u = heapq.heappop(heap)
        if t > arrival[u]:
            continue
        for e in out[u]:
            v = dst[e]
            if t + days[e] < arrival[v]:
                arrival[v] = t + days[e]
                heapq.heappush(heap, (arrival[v], int(v)))
    for u in sorted(range(n), key=lambda k: stage[k]):  # topological propagation of the shock
        for e in out[u]:
            v = dst[e]
            shock[v] = max(shock[v], shock[u] * decay)
    delay = np.where(shock >= AFFECTED_DELAY, shock, 0.0)
    arrival = np.where(delay > 0, arrival, NOT_REACHED)
    return {"delay": delay, "risk": np.clip(delay / 40.0, 0, 1), "arrival": arrival}


def model_predict(model, d: Data) -> dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        p = model(d.x, d.edge_index, d.edge_attr)
    delay = torch.expm1(p["delay"]).numpy()
    delay = np.where(delay >= AFFECTED_DELAY, delay, 0.0)
    arrival = np.where(delay > 0, p["arrival"].numpy() * ARRIVAL_SCALE, NOT_REACHED)
    return {"delay": delay, "risk": p["risk"].numpy(), "arrival": arrival}


# ----------------------------------------------------------------------------- metrics

def evaluate(samples: list[Data], predictor) -> dict[str, float]:
    t_delay, p_delay, t_risk, p_risk, t_arr, p_arr, stages = [], [], [], [], [], [], []
    for d in samples:
        p = predictor(d)
        td = np.expm1(d.y_delay.numpy())
        td = np.where(td >= AFFECTED_DELAY, td, 0.0)
        t_delay.append(td)
        p_delay.append(p["delay"])
        t_risk.append(d.y_risk.numpy())
        p_risk.append(p["risk"])
        both = (td > 0) & (p["delay"] > 0) & d.arrival_mask.numpy()
        t_arr.append(d.y_arrival.numpy()[both] * ARRIVAL_SCALE)
        p_arr.append(p["arrival"][both])
        stages.append(d.stage.numpy())
    td, pd = np.concatenate(t_delay), np.concatenate(p_delay)
    tr, pr = np.concatenate(t_risk), np.concatenate(p_risk)
    ta, pa = np.concatenate(t_arr), np.concatenate(p_arr)
    st = np.concatenate(stages)
    ta_mask, pa_mask = td > 0, pd > 0
    tp = float((ta_mask & pa_mask).sum())
    precision = tp / max(pa_mask.sum(), 1)
    recall = tp / max(ta_mask.sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    scope = ta_mask | pa_mask   # nodes where either truth or prediction says "affected"
    ss_res = float(((td[scope] - pd[scope]) ** 2).sum())
    ss_tot = float(((td[scope] - td[scope].mean()) ** 2).sum())
    per_stage = {}
    for s, name in enumerate(["tier2", "tier1", "manufacturer", "port", "distributor"]):
        m = (st == s) & ta_mask
        per_stage[name] = round(float(np.abs(td[m] - pd[m]).mean()), 3) if m.any() else None
    return {
        "delay_mae_all_nodes": round(float(np.abs(td - pd).mean()), 4),
        "delay_mae_affected": round(float(np.abs(td[ta_mask] - pd[ta_mask]).mean()), 3),
        "delay_rmse_affected": round(float(np.sqrt(((td[ta_mask] - pd[ta_mask]) ** 2).mean())), 3),
        "delay_r2_in_scope": round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else None,
        "affected_precision": round(precision, 4),
        "affected_recall": round(recall, 4),
        "affected_f1": round(f1, 4),
        "arrival_mae_days": round(float(np.abs(ta - pa).mean()), 3) if len(ta) else None,
        "risk_mae": round(float(np.abs(tr - pr).mean()), 4),
        "mae_affected_by_stage": per_stage,
        "nodes_evaluated": int(len(td)),
        "true_affected_nodes": int(ta_mask.sum()),
    }


# ----------------------------------------------------------------------------- plots

def save_plots(history: dict, gnn, test: list[Data]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train"], label="train")
    ax.plot(history["val"], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("RippleGNN training curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "training_curve.png", dpi=130)
    plt.close(fig)

    tds, pds = [], []
    for d in test[:120]:
        p = model_predict(gnn, d)
        td = np.expm1(d.y_delay.numpy())
        m = (td >= AFFECTED_DELAY) | (p["delay"] > 0)
        tds.append(td[m])
        pds.append(p["delay"][m])
    td, pd = np.concatenate(tds), np.concatenate(pds)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(td, pd, s=5, alpha=0.35)
    lim = max(td.max(), pd.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlabel("simulated delay (days)")
    ax.set_ylabel("GNN predicted delay (days)")
    ax.set_title("Predicted vs. true delay (unseen networks)")
    fig.tight_layout()
    fig.savefig(FIGURES / "pred_vs_true.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    torch.manual_seed(0)
    rng = random.Random(0)
    n_train, n_val, n_test, per = (6, 2, 2, 6) if args.quick else (40, 8, 10, 25)
    epochs = 2 if args.quick else args.epochs

    t0 = time.time()
    train_g = [random_graph(1000 + i, rng) for i in range(n_train)]
    val_g = [random_graph(2000 + i, rng) for i in range(n_val)]
    test_g = [random_graph(3000 + i, rng) for i in range(n_test)]
    deploy_g = from_generator(*generate())  # the network actually served by the API

    train = build_samples(train_g, per, seed=1)
    val = build_samples(val_g, max(per - 5, 4), seed=2)
    test = build_samples(test_g, per + 5, seed=3)
    deploy = build_samples([deploy_g], 30 if args.quick else 150, seed=4)
    print(f"samples: train={len(train)} val={len(val)} test={len(test)} deploy={len(deploy)} "
          f"({time.time() - t0:.1f}s)")

    in_dim, edge_dim = len(NODE_FEATURES), len(EDGE_FEATURES)
    results = {}
    history_gnn = None
    trained = {}
    for name, model in [("gnn", RippleGNN(in_dim, edge_dim)), ("mlp", GraphMLP(in_dim))]:
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        tl = DataLoader(train, batch_size=8, shuffle=True)
        vl = DataLoader(val, batch_size=16)
        best, best_state, hist = float("inf"), None, {"train": [], "val": []}
        for ep in range(epochs):
            tr = run_epoch(model, tl, opt)
            va = run_epoch(model, vl)
            sched.step()
            hist["train"].append(tr)
            hist["val"].append(va)
            if va < best:
                best, best_state = va, {k: v.clone() for k, v in model.state_dict().items()}
            print(f"[{name}] epoch {ep + 1:02d}/{epochs} train={tr:.4f} val={va:.4f} ({time.time() - t0:.0f}s)")
        model.load_state_dict(best_state)
        trained[name] = model
        if name == "gnn":
            history_gnn = hist

    predictors = {
        "seeds_only_baseline": baseline_seeds_only,
        "hop_decay_heuristic": baseline_hop_decay,
        "mlp_no_graph": lambda d: model_predict(trained["mlp"], d),
        "ripple_gnn": lambda d: model_predict(trained["gnn"], d),
    }
    for split_name, split in [("test_unseen_networks", test), ("deployed_network", deploy)]:
        results[split_name] = {k: evaluate(split, f) for k, f in predictors.items()}

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": trained["gnn"].state_dict(), "config": trained["gnn"].config,
                "node_features": NODE_FEATURES, "edge_features": EDGE_FEATURES}, ARTIFACTS / "ripple_gnn.pt")
    meta = {
        "label_noise": LABEL_NOISE,
        "epochs": epochs,
        "train_samples": len(train), "val_samples": len(val),
        "test_samples": len(test), "deploy_samples": len(deploy),
        "train_networks": n_train, "test_networks": n_test,
        "training_seconds": round(time.time() - t0, 1),
        "parameters": sum(p.numel() for p in trained["gnn"].parameters()),
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    save_plots(history_gnn, trained["gnn"], test)

    print(json.dumps(meta, indent=2))
    for split_name, res in results.items():
        print(f"\n== {split_name} ==")
        print(f"{'model':22s} {'MAE(aff)':>9s} {'R2':>7s} {'F1':>6s} {'ArrMAE':>7s}")
        for k, m in res.items():
            print(f"{k:22s} {m['delay_mae_affected']:9.2f} {m['delay_r2_in_scope'] or 0:7.3f} "
                  f"{m['affected_f1']:6.3f} {m['arrival_mae_days'] or 0:7.2f}")


if __name__ == "__main__":
    main()

