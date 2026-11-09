import { useCallback, useEffect, useState } from "react";
import { fetchGraph, fetchNodeDetails, fetchStats } from "./api";
import SupplyChainGraph from "./components/SupplyChainGraph";
import NodeInspector from "./components/NodeInspector";
import type {
  GraphLink,
  GraphNode,
  LayoutMode,
  NodeDetailsResponse,
  StatsResponse,
} from "./types";

export default function App() {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [links, setLinks] = useState<GraphLink[]>([]);
  const [dataSource, setDataSource] = useState<"neo4j" | "fallback">("neo4j");

  const [countryFilter, setCountryFilter] = useState<string>("");
  const [industryFilter, setIndustryFilter] = useState<string>("");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [layoutMode, setLayoutMode] = useState<LayoutMode>("tiered");

  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [nodeDetails, setNodeDetails] = useState<NodeDetailsResponse | null>(null);
  const [inspectorLoading, setInspectorLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    try {
      setError(null);
      const [statsRes, graphRes] = await Promise.all([
        fetchStats(),
        fetchGraph({
          country: countryFilter || undefined,
          industry: industryFilter || undefined,
        }),
      ]);
      setStats(statsRes);
      setNodes(graphRes.nodes);
      setLinks(graphRes.links);
      setDataSource(graphRes.data_source);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to connect to backend API");
    }
  }, [countryFilter, industryFilter]);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  useEffect(() => {
    if (!selectedNodeId) {
      setNodeDetails(null);
      return;
    }
    let cancelled = false;
    setInspectorLoading(true);
    fetchNodeDetails(selectedNodeId)
      .then((res) => {
        if (!cancelled) setNodeDetails(res);
      })
      .catch(() => {
        if (!cancelled) setNodeDetails(null);
      })
      .finally(() => {
        if (!cancelled) setInspectorLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedNodeId]);

  const handleResetFilters = () => {
    setCountryFilter("");
    setIndustryFilter("");
    setSearchQuery("");
    setSelectedNodeId(null);
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-logo">AG</div>
          <div>
            <h1>AtmoGraph</h1>
            <p className="subtitle">Global Supply Chain Ripple Effect Predictor</p>
          </div>
        </div>

        <div className="kpi-strip">
          <div className="kpi-pill">
            <span className="kpi-num">{stats?.total_nodes ?? nodes.length}</span>
            <span className="kpi-lbl">Nodes</span>
          </div>
          <div className="kpi-pill">
            <span className="kpi-num">{stats?.total_edges ?? links.length}</span>
            <span className="kpi-lbl">Routes</span>
          </div>
          <div className="kpi-pill">
            <span className="kpi-num">
              {(stats?.tier2_suppliers ?? 0) + (stats?.tier1_suppliers ?? 0)}
            </span>
            <span className="kpi-lbl">Suppliers</span>
          </div>
          <div className="kpi-pill">
            <span className="kpi-num">{stats?.ports ?? 0}</span>
            <span className="kpi-lbl">Ports</span>
          </div>
          <div className="kpi-pill">
            <span className="kpi-num">{stats?.countries.length ?? 0}</span>
            <span className="kpi-lbl">Countries</span>
          </div>
        </div>

        <div className="status-block">
          <span
            className={`db-badge ${
              dataSource === "neo4j" ? "db-live" : "db-fallback"
            }`}
            title={
              dataSource === "neo4j"
                ? "Connected to Neo4j Desktop (neo4j://127.0.0.1:7687)"
                : "Neo4j Desktop is stopped — showing built-in graph snapshot. Start atomgraph in Neo4j Desktop and click Refresh."
            }
          >
            <span className="status-dot" />
            {dataSource === "neo4j" ? "Neo4j Live" : "Snapshot Mode"}
          </span>
          <button type="button" className="refresh-btn" onClick={() => void loadData()}>
            Refresh
          </button>
        </div>
      </header>

      <div className="toolbar">
        <input
          type="search"
          className="search-input"
          placeholder="Highlight node, port, country, or sector (e.g. Rotterdam, Taiwan, Semiconductors)..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />

        <select
          className="filter-select"
          value={countryFilter}
          onChange={(e) => {
            setSelectedNodeId(null);
            setCountryFilter(e.target.value);
          }}
        >
          <option value="">All Countries</option>
          {stats?.countries.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>

        <select
          className="filter-select"
          value={industryFilter}
          onChange={(e) => {
            setSelectedNodeId(null);
            setIndustryFilter(e.target.value);
          }}
        >
          <option value="">All Industries</option>
          {stats?.industries.map((ind) => (
            <option key={ind} value={ind}>
              {ind}
            </option>
          ))}
        </select>

        <div className="layout-toggle">
          <button
            type="button"
            className={layoutMode === "tiered" ? "active" : ""}
            onClick={() => setLayoutMode("tiered")}
          >
            Tiered Flow
          </button>
          <button
            type="button"
            className={layoutMode === "force" ? "active" : ""}
            onClick={() => setLayoutMode("force")}
          >
            Force Graph
          </button>
        </div>

        {(countryFilter || industryFilter || searchQuery || selectedNodeId) && (
          <button type="button" className="clear-btn" onClick={handleResetFilters}>
            Reset
          </button>
        )}
      </div>

      {error && <div className="error-banner">{error}</div>}

      <main className="workspace-grid">
        <SupplyChainGraph
          nodes={nodes}
          links={links}
          layoutMode={layoutMode}
          searchQuery={searchQuery}
          selectedNodeId={selectedNodeId}
          nodeDetails={nodeDetails}
          onSelectNode={setSelectedNodeId}
        />
        <NodeInspector
          details={nodeDetails}
          loading={inspectorLoading}
          onSelectNode={setSelectedNodeId}
        />
      </main>
    </div>
  );
}

