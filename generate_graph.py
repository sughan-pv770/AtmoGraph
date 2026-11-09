"""
AtmoGraph - Week 1: generate a mock global supply chain and load it into Neo4j.

Graph shape (multi-hop on purpose, so the GNN has something to learn later):

  Supplier(tier 2) -[:SUPPLIES]-> Supplier(tier 1) -[:SUPPLIES]-> Manufacturer
  Manufacturer -[:SHIPS_VIA]-> Port -[:SHIPS_TO]-> Distributor

Usage:
  pip install neo4j
  set NEO4J_PASSWORD=your_password        (Windows CMD)
  export NEO4J_PASSWORD=your_password     (macOS / Linux)

  python generate_graph.py --dry-run      # only generate and print counts
  python generate_graph.py --wipe         # delete old data, then load
  python generate_graph.py --wipe --t2 400 --t1 300 --mfr 100   # bigger graph
"""
import argparse
import os
import random

COUNTRIES = ["China", "Taiwan", "South Korea", "Japan", "Vietnam", "India",
             "Germany", "Netherlands", "Mexico", "USA", "Malaysia", "Thailand"]
INDUSTRIES = ["Semiconductors", "Electronics", "Batteries", "Plastics",
              "Metals", "Textiles", "Automotive Parts"]
PORTS = [("Shanghai", "China"), ("Shenzhen", "China"), ("Singapore", "Singapore"),
         ("Busan", "South Korea"), ("Rotterdam", "Netherlands"),
         ("Hamburg", "Germany"), ("Antwerp", "Belgium"), ("Los Angeles", "USA"),
         ("Long Beach", "USA"), ("Chennai", "India"), ("Nhava Sheva", "India"),
         ("Port Klang", "Malaysia")]
REGIONS = ["North America", "Europe", "South Asia", "East Asia",
           "Latin America", "Middle East"]


def generate(n_t2=40, n_t1=30, n_mfr=15, seed=42):
    """Return (nodes, edges). nodes: {label: [props]}, edges: [(rel, src_label, dst_label, src_id, dst_id, props)]"""
    rng = random.Random(seed)

    def supplier(i, tier):
        return {"id": f"SUP{tier}-{i:04d}", "name": f"Supplier T{tier}-{i:04d}",
                "country": rng.choice(COUNTRIES), "industry": rng.choice(INDUSTRIES),
                "tier": tier, "risk": 0.0}

    t2 = [supplier(i, 2) for i in range(1, n_t2 + 1)]
    t1 = [supplier(i, 1) for i in range(1, n_t1 + 1)]
    mfrs = [{"id": f"MFR-{i:04d}", "name": f"Manufacturer {i:04d}",
             "country": rng.choice(COUNTRIES),
             "industry": rng.choice(["Electronics", "Automotive Parts", "Textiles"]),
             "risk": 0.0} for i in range(1, n_mfr + 1)]
    ports = [{"id": f"PORT-{i:03d}", "name": name, "country": country, "risk": 0.0}
             for i, (name, country) in enumerate(PORTS, 1)]
    dists = [{"id": f"DIST-{i:03d}", "name": f"{region} Distribution Hub",
              "region": region, "risk": 0.0} for i, region in enumerate(REGIONS, 1)]

    nodes = {"Supplier": t2 + t1, "Manufacturer": mfrs, "Port": ports, "Distributor": dists}
    edges = []

    def connect(rel, src_label, dst_label, srcs, dsts, kmin, kmax, make_props):
        seen = set()

        def add(s, d):
            if (s["id"], d["id"]) not in seen:
                seen.add((s["id"], d["id"]))
                edges.append((rel, src_label, dst_label, s["id"], d["id"], make_props()))

        for s in srcs:
            k = rng.randint(kmin, min(kmax, len(dsts)))
            for d in rng.sample(dsts, k):
                add(s, d)
        # make sure no destination node is left without an incoming edge
        covered = {d_id for _, d_id in seen}
        for d in dsts:
            if d["id"] not in covered:
                add(rng.choice(srcs), d)

    connect("SUPPLIES", "Supplier", "Supplier", t2, t1, 1, 3,
            lambda: {"weight": round(rng.uniform(0.3, 1.0), 2), "lead_days": rng.randint(5, 45)})
    connect("SUPPLIES", "Supplier", "Manufacturer", t1, mfrs, 1, 3,
            lambda: {"weight": round(rng.uniform(0.3, 1.0), 2), "lead_days": rng.randint(5, 30)})
    connect("SHIPS_VIA", "Manufacturer", "Port", mfrs, ports, 1, 2,
            lambda: {"weight": round(rng.uniform(0.3, 1.0), 2), "transit_days": rng.randint(2, 15)})
    connect("SHIPS_TO", "Port", "Distributor", ports, dists, 1, 3,
            lambda: {"weight": round(rng.uniform(0.3, 1.0), 2), "transit_days": rng.randint(5, 35)})
    return nodes, edges


def load(nodes, edges, uri, user, password, wipe, database="neo4j"):
    from neo4j import GraphDatabase  # imported here so --dry-run works without it

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver:
        if database.isidentifier():
            try:
                with driver.session(database="system") as sys_session:
                    sys_session.run(f"CREATE DATABASE {database} IF NOT EXISTS WAIT")
            except Exception:
                pass
        with driver.session(database=database) as session:
            if wipe:
                session.run("MATCH (n) DETACH DELETE n")

            for label in nodes:  # labels are a fixed list above, so f-strings are safe here
                session.run(f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS "
                            f"FOR (n:{label}) REQUIRE n.id IS UNIQUE")
                session.run(f"UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) SET n += row",
                            rows=nodes[label])

            grouped = {}
            for rel, sl, dl, s_id, d_id, props in edges:
                grouped.setdefault((rel, sl, dl), []).append({"src": s_id, "dst": d_id, "props": props})
            for (rel, sl, dl), rows in grouped.items():
                session.run(
                    f"UNWIND $rows AS r "
                    f"MATCH (a:{sl} {{id: r.src}}) MATCH (b:{dl} {{id: r.dst}}) "
                    f"MERGE (a)-[e:{rel}]->(b) SET e += r.props", rows=rows)

            print("Loaded into Neo4j:")
            for rec in session.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c ORDER BY label"):
                print(f"  {rec['label']}: {rec['c']} nodes")
            for rec in session.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY t"):
                print(f"  {rec['t']}: {rec['c']} relationships")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t2", type=int, default=40, help="number of tier-2 suppliers")
    ap.add_argument("--t1", type=int, default=30, help="number of tier-1 suppliers")
    ap.add_argument("--mfr", type=int, default=15, help="number of manufacturers")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wipe", action="store_true", help="delete all existing data first")
    ap.add_argument("--dry-run", action="store_true", help="generate only, do not connect")
    ap.add_argument("--password", type=str, default=None, help="Neo4j password (or set NEO4J_PASSWORD in .env)")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)
    except ImportError:
        pass

    nodes, edges = generate(args.t2, args.t1, args.mfr, args.seed)
    print({k: len(v) for k, v in nodes.items()}, f"edges={len(edges)}")
    if args.dry_run:
        return

    password = args.password or os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise SystemExit("Set NEO4J_PASSWORD in f:\\atomgraph\\.env (or pass --password).")
    load(nodes, edges,
         os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687"),
         os.environ.get("NEO4J_USER", "neo4j"),
         password, args.wipe,
         os.environ.get("NEO4J_DATABASE", "neo4j"))


if __name__ == "__main__":
    main()
