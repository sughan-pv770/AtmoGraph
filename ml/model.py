"""
RippleGNN: a directed message-passing network for node-level disruption regression.

Messages flow only downstream (supplier -> ... -> distributor), mirroring how a
disruption physically propagates. Each layer = one supply-chain hop; five layers
cover the longest path (T2 -> T1 -> Manufacturer -> Port -> Distributor) plus one
extra layer of refinement.

Per node the model outputs three heads:
  * delay   : log(1 + expected delay days)
  * risk    : disruption risk score in [0, 1]
  * arrival : day the impact reaches the node (scaled by 1/100)
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import MessagePassing


class RippleConv(MessagePassing):
    def __init__(self, hidden: int, edge_dim: int) -> None:
        super().__init__(aggr=["sum", "max"], flow="source_to_target")
        self.msg = nn.Sequential(nn.Linear(hidden + edge_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden))
        self.update_mlp = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.SiLU(),
                                        nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        agg = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return self.norm(h + self.update_mlp(torch.cat([h, agg], dim=-1)))

    def message(self, h_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.msg(torch.cat([h_j, edge_attr], dim=-1))


class RippleGNN(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden: int = 64, layers: int = 5) -> None:
        super().__init__()
        self.config = {"in_dim": in_dim, "edge_dim": edge_dim, "hidden": hidden, "layers": layers}
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.convs = nn.ModuleList(RippleConv(hidden, edge_dim) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(hidden + in_dim, hidden), nn.SiLU(), nn.Linear(hidden, 3))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
        out = self.head(torch.cat([h, x], dim=-1))
        return {
            "delay": torch.relu(out[:, 0]),
            "risk": torch.sigmoid(out[:, 1]),
            "arrival": torch.relu(out[:, 2]),
        }


class GraphMLP(nn.Module):
    """Baseline: same node features, no message passing (cannot see the network)."""

    def __init__(self, in_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Linear(hidden, 3))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.net(x)
        return {"delay": torch.relu(out[:, 0]), "risk": torch.sigmoid(out[:, 1]),
                "arrival": torch.relu(out[:, 2])}

