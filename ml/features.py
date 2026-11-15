"""Turn a SupplyGraph + disruption seeds into PyTorch Geometric tensors."""
from __future__ import annotations

import math

import torch

from ml.graph import SupplyGraph
from ml.simulate import NOT_REACHED, RippleResult

NODE_FEATURES = [
    "seed_severity", "is_seed",
    "stage_t2", "stage_t1", "stage_mfr", "stage_port", "stage_dist",
    "buffer_days", "log_in_degree", "log_out_degree", "in_weight_sum", "mean_in_days",
]
EDGE_FEATURES = ["weight", "share", "days"]
ARRIVAL_SCALE = 100.0


def static_tensors(g: SupplyGraph) -> dict[str, torch.Tensor]:
    """Seed-independent tensors (cache per graph)."""
    n = g.num_nodes
    x = torch.zeros(n, len(NODE_FEATURES) - 2)
    for v, node in enumerate(g.nodes):
        x[v, node["stage"]] = 1.0
        x[v, 5] = node["buffer_days"] / 20.0
        x[v, 6] = math.log1p(len(g.in_edges[v]))
        x[v, 7] = math.log1p(len(g.out_edges[v]))
        x[v, 8] = g.in_weight_sum[v] / 5.0
        ins = g.in_edges[v]
        x[v, 9] = (sum(g.days[e] for e in ins) / len(ins) / 45.0) if ins else 0.0
    edge_index = torch.tensor([g.src, g.dst], dtype=torch.long)
    edge_attr = torch.tensor(
        [[g.weight[e], g.share(e), g.days[e] / 45.0] for e in range(g.num_edges)],
        dtype=torch.float,
    ).reshape(-1, len(EDGE_FEATURES))
    return {"x_static": x, "edge_index": edge_index, "edge_attr": edge_attr}


def node_features(static: dict[str, torch.Tensor], seeds: dict[int, float]) -> torch.Tensor:
    n = static["x_static"].shape[0]
    seed_cols = torch.zeros(n, 2)
    for i, s in seeds.items():
        seed_cols[i, 0] = float(s)
        seed_cols[i, 1] = 1.0
    return torch.cat([seed_cols, static["x_static"]], dim=1)


def targets(res: RippleResult) -> dict[str, torch.Tensor]:
    delay = torch.tensor(res.delay, dtype=torch.float)
    arrival = torch.tensor(res.arrival, dtype=torch.float)
    reached = arrival < NOT_REACHED
    return {
        "y_delay": torch.log1p(delay),                       # regress log(1 + delay days)
        "y_risk": torch.tensor(res.risk, dtype=torch.float),
        "y_arrival": torch.where(reached, arrival / ARRIVAL_SCALE, torch.zeros_like(arrival)),
        "arrival_mask": reached,
    }

