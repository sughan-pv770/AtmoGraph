"""
Index-based supply chain graph used by the simulator, feature builder and GNN.

Both Neo4j query results and the offline generator are converted into this one
structure, so training and serving see exactly the same representation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

STAGE_OF_LABEL = {"Manufacturer": 2, "Port": 3, "Distributor": 4}
DEFAULT_BUFFER = {0: 4, 1: 4, 2: 6, 3: 2, 4: 6}


def stage_of(label: str, tier: Any = None) -> int:
    if label == "Supplier":
        return 0 if int(tier or 1) == 2 else 1
    return STAGE_OF_LABEL.get(label, 2)


@dataclass
class SupplyGraph:
    nodes: list[dict[str, Any]]                       # serialized node dicts (id, name, stage, ...)
    src: list[int]                                    # edge source indices
    dst: list[int]                                    # edge target indices
    weight: list[float]                               # edge dependency weight (0..1)
    days: list[int]                                   # lead / transit days on the edge
    index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.index = {n["id"]: i for i, n in enumerate(self.nodes)}
        n = len(self.nodes)
        self.in_edges: list[list[int]] = [[] for _ in range(n)]
        self.out_edges: list[list[int]] = [[] for _ in range(n)]
        for e, (s, d) in enumerate(zip(self.src, self.dst)):
            self.out_edges[s].append(e)
            self.in_edges[d].append(e)
        self.in_weight_sum = [sum(self.weight[e] for e in self.in_edges[v]) for v in range(n)]
        # Edges always go to a higher stage (T2 -> T1 -> MFR -> Port -> Dist), so
        # sorting by stage is a valid topological order.
        self.topo_order = sorted(range(n), key=lambda v: self.nodes[v]["stage"])

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.src)

    def share(self, e: int) -> float:
        """Fraction of the target's inbound supply that comes through edge e."""
        total = self.in_weight_sum[self.dst[e]]
        return self.weight[e] / total if total > 0 else 0.0

    def ids(self, idxs: Iterable[int]) -> list[str]:
        return [self.nodes[i]["id"] for i in idxs]


def _node_dict(label: str, props: dict[str, Any]) -> dict[str, Any]:
    stage = stage_of(label, props.get("tier"))
    return {
        "id": props["id"],
        "name": props.get("name", props["id"]),
        "label": label,
        "stage": stage,
        "tier": props.get("tier"),
        "country": props.get("country"),
        "region": props.get("region"),
        "industry": props.get("industry"),
        "risk": float(props.get("risk", 0.0) or 0.0),
        "buffer_days": int(props.get("buffer_days", DEFAULT_BUFFER[stage])),
    }


def from_generator(raw_nodes: dict[str, list[dict]], raw_edges: list[tuple]) -> SupplyGraph:
    """Build from generate_graph.generate() output."""
    nodes = [_node_dict(label, p) for label, items in raw_nodes.items() for p in items]
    idx = {n["id"]: i for i, n in enumerate(nodes)}
    src, dst, w, d = [], [], [], []
    for _rel, _sl, _dl, s_id, d_id, props in raw_edges:
        src.append(idx[s_id])
        dst.append(idx[d_id])
        w.append(float(props.get("weight", 0.5)))
        d.append(int(props.get("lead_days", props.get("transit_days", 0))))
    return SupplyGraph(nodes, src, dst, w, d)


def from_records(node_rows: list[dict], edge_rows: list[dict]) -> SupplyGraph:
    """Build from Neo4j rows: node_rows {label, props}; edge_rows {source, target, props}."""
    nodes = [_node_dict(r["label"], r["props"]) for r in node_rows]
    idx = {n["id"]: i for i, n in enumerate(nodes)}
    src, dst, w, d = [], [], [], []
    for r in edge_rows:
        if r["source"] in idx and r["target"] in idx:
            p = r["props"]
            src.append(idx[r["source"]])
            dst.append(idx[r["target"]])
            w.append(float(p.get("weight", 0.5)))
            d.append(int(p.get("lead_days", p.get("transit_days", 0)) or 0))
    return SupplyGraph(nodes, src, dst, w, d)
