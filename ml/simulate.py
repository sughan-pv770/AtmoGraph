"""
Ground-truth ripple simulator and disruption scenario sampler.

Real disruption-propagation labels for a synthetic network do not exist, so we
define a transparent, domain-motivated process and use it to label training
scenarios for the GNN. The GNN then learns to approximate this process from
graph structure alone (and generalises to unseen networks of different sizes).

Process (nodes visited in topological order, upstream -> downstream):
  * A directly disrupted node (severity s) suffers an own outage of
    s * OUTAGE_DAYS[stage] days.
  * Each disrupted upstream partner u passes risk to v in proportion to how much
    of v's supply goes through that edge (its "share"), attenuated per hop
    (see PASS_THROUGH). Risk from multiple partners combines as 1 - prod(1 - r).
  * The inherited delay is a share-weighted pass-through of the partner's delay,
    and the node's inventory buffer absorbs part of it.
  * The impact reaches v after the edge's lead/transit time (arrival day).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from ml.graph import SupplyGraph

OUTAGE_DAYS = {0: 28.0, 1: 32.0, 2: 38.0, 3: 30.0, 4: 16.0}
HOP_ATTENUATION = 0.92
# (base, slope) of how much of an upstream partner's risk/delay passes into a node of
# a given stage: base + slope * share. Suppliers/manufacturers can re-source, so the
# share matters a lot; ports and distribution hubs are conduits for the goods
# themselves, so most of the delay flows straight through them.
PASS_THROUGH = {0: (0.35, 0.65), 1: (0.35, 0.65), 2: (0.35, 0.65), 3: (0.70, 0.30), 4: (0.70, 0.30)}
AFFECTED_DELAY = 1.0          # a node counts as "affected" at >= 1 day predicted delay
NOT_REACHED = 999.0           # arrival day for unaffected nodes


@dataclass
class RippleResult:
    risk: list[float]
    delay: list[float]
    arrival: list[float]

    def affected(self) -> list[int]:
        return [i for i, d in enumerate(self.delay) if d >= AFFECTED_DELAY]


def simulate(g: SupplyGraph, seeds: dict[int, float], noise: float = 0.0,
             rng: random.Random | None = None) -> RippleResult:
    n = g.num_nodes
    risk = [0.0] * n
    delay = [0.0] * n
    arrival = [NOT_REACHED] * n
    rng = rng or random.Random(0)

    for v in g.topo_order:
        stage = g.nodes[v]["stage"]
        s = float(seeds.get(v, 0.0))
        own_delay = s * OUTAGE_DAYS[stage]

        keep_prob = 1.0
        inherited = 0.0
        arr = NOT_REACHED
        for e in g.in_edges[v]:
            u = g.src[e]
            if risk[u] < 0.02 or delay[u] < AFFECTED_DELAY:
                continue
            share = g.share(e)
            base, slope = PASS_THROUGH[stage]
            r = min(1.0, risk[u] * HOP_ATTENUATION * (base + slope * share))
            keep_prob *= (1.0 - r)
            inherited = max(inherited, delay[u] * (base + slope * share))
            arr = min(arr, arrival[u] + g.days[e])
        inbound_risk = 1.0 - keep_prob

        absorbed = max(0.0, inherited - g.nodes[v]["buffer_days"]) * (0.45 + 0.55 * inbound_risk)
        d = own_delay + absorbed
        if noise > 0 and d > 0:
            d *= math.exp(rng.gauss(0.0, noise))

        risk[v] = max(s, inbound_risk if absorbed > 0 else 0.0)
        delay[v] = d
        if s > 0:
            arrival[v] = 0.0
        elif d >= AFFECTED_DELAY:
            arrival[v] = arr
        if d < AFFECTED_DELAY and s == 0:
            risk[v], delay[v] = 0.0, 0.0
    return RippleResult(risk, delay, arrival)


def hops_from_seeds(g: SupplyGraph, seed_idxs: list[int]) -> list[int]:
    """Downstream hop distance from the nearest seed (-1 = unreachable)."""
    hops = [-1] * g.num_nodes
    frontier = list(seed_idxs)
    for s in seed_idxs:
        hops[s] = 0
    while frontier:
        nxt = []
        for u in frontier:
            for e in g.out_edges[u]:
                v = g.dst[e]
                if hops[v] == -1:
                    hops[v] = hops[u] + 1
                    nxt.append(v)
        frontier = nxt
    return hops


# --------------------------------------------------------------------------- scenarios

def sample_scenario(g: SupplyGraph, rng: random.Random) -> tuple[str, dict[int, float]]:
    """Sample a realistic disruption event; returns (description, seeds)."""
    nodes = g.nodes
    kind = rng.choices(
        ["port", "country", "industry", "facility", "compound"],
        weights=[0.25, 0.25, 0.2, 0.2, 0.1],
    )[0]

    def port_event() -> tuple[str, dict[int, float]]:
        ports = [i for i, n in enumerate(nodes) if n["label"] == "Port"]
        p = rng.choice(ports)
        return f"port strike at {nodes[p]['name']}", {p: rng.uniform(0.5, 1.0)}

    def country_event() -> tuple[str, dict[int, float]]:
        countries = sorted({n["country"] for n in nodes if n["country"] and n["stage"] <= 2})
        c = rng.choice(countries)
        frac = rng.uniform(0.3, 1.0)
        sev = rng.uniform(0.3, 0.9)
        members = [i for i, n in enumerate(nodes) if n["country"] == c and n["stage"] <= 2]
        chosen = [i for i in members if rng.random() < frac] or members[:1]
        return f"regional crisis in {c}", {i: sev * rng.uniform(0.8, 1.0) for i in chosen}

    def industry_event() -> tuple[str, dict[int, float]]:
        inds = sorted({n["industry"] for n in nodes if n["industry"] and n["stage"] <= 1})
        ind = rng.choice(inds)
        frac = rng.uniform(0.2, 0.6)
        sev = rng.uniform(0.2, 0.7)
        members = [i for i, n in enumerate(nodes) if n["industry"] == ind and n["stage"] <= 1]
        chosen = [i for i in members if rng.random() < frac] or members[:1]
        return f"{ind} shortage", {i: sev * rng.uniform(0.8, 1.0) for i in chosen}

    def facility_event() -> tuple[str, dict[int, float]]:
        cands = [i for i, n in enumerate(nodes) if n["stage"] <= 2]
        f = rng.choice(cands)
        return f"fire at {nodes[f]['name']}", {f: rng.uniform(0.6, 1.0)}

    makers = {"port": port_event, "country": country_event, "industry": industry_event,
              "facility": facility_event}
    if kind == "compound":
        a_desc, a = makers[rng.choice(list(makers))]()
        b_desc, b = makers[rng.choice(list(makers))]()
        merged: dict[int, float] = dict(a)
        for k, v in b.items():
            merged[k] = max(merged.get(k, 0.0), v)
        return f"{a_desc} + {b_desc}", merged
    return makers[kind]()


def summarize(g: SupplyGraph, res: RippleResult) -> dict[str, Any]:
    aff = res.affected()
    return {
        "affected_nodes": len(aff),
        "affected_distributors": sum(1 for i in aff if g.nodes[i]["stage"] == 4),
        "max_delay_days": round(max(res.delay) if res.delay else 0.0, 1),
    }
