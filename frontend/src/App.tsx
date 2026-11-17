import { useCallback, useEffect, useState } from "react";
import {
  analyzeNews,
  fetchGraph,
  fetchNewsFeed,
  fetchNodeDetails,
  fetchStats,
  predictRipple,
} from "./api";
import DisruptionFeed from "./components/DisruptionFeed";
import NodeInspector from "./components/NodeInspector";
import SupplyChainGraph from "./components/SupplyChainGraph";
import type {
  GraphLink,
  GraphNode,
  NewsItem,
  NLPAnalysisResult,
  NodeDetailsResponse,
  RipplePredictionResponse,
  StatsResponse,
} from "./types";

export default function App() {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [links, setLinks] = useState<GraphLink[]>([]);
  const [dataSource, setDataSource] = useState<string>("in_memory_graph");

  const [countryFilter, setCountryFilter] = useState<string>("");
  const [industryFilter, setIndustryFilter] = useState<string>("");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [layoutMode, setLayoutMode] = useState<"tiered" | "force">("tiered");

  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [nodeDetails, setNodeDetails] = useState<NodeDetailsResponse | null>(null);
  const [inspectorLoading, setInspectorLoading] = useState<boolean>(false);

  // NLP & GNN State
  const [newsFeed, setNewsFeed] = useState<NewsItem[]>([]);
  const [nlpAnalyzing, setNlpAnalyzing] = useState<boolean>(false);
  const [activeNlpResult, setActiveNlpResult] = useState<NLPAnalysisResult | null>(null);
  const [predictionData, setPredictionData] = useState<RipplePredictionResponse | null>(null);

  // Timeline Slider (30 / 60 / 90 days out)
  const [timelineDays, setTimelineDays] = useState<number>(90);
  const [activeTab, setActiveTab] = useState<"inspector" | "feed">("feed");

  // Load Initial Graph & Stats
  const loadBaseData = useCallback(async () => {
    try {
      const [s, g, feed] = await Promise.all([
        fetchStats(),
        fetchGraph(countryFilter || undefined, industryFilter || undefined),
        fetchNewsFeed(),
      ]);
      setStats(s);
      setNodes(g.nodes);
      setLinks(g.links);
      setDataSource(g.data_source);
      setNewsFeed(feed);
    } catch (e) {
      console.error("Failed to load initial data", e);
    }
  }, [countryFilter, industryFilter]);

  useEffect(() => {
    loadBaseData();
  }, [loadBaseData]);

  // Load Node Details on Selection
  useEffect(() => {
    if (!selectedNodeId) {
      setNodeDetails(null);
      return;
    }
    setInspectorLoading(true);
    fetchNodeDetails(selectedNodeId)
      .then((d) => setNodeDetails(d))
      .catch(() => setNodeDetails(null))
      .finally(() => setInspectorLoading(false));
  }, [selectedNodeId]);

  // Trigger Full Disruption Simulation Pipeline
  const runSimulation = async (text: string) => {
    setNlpAnalyzing(true);
    try {
      const nlp = await analyzeNews(text, true);
      setActiveNlpResult(nlp);
      if (Object.keys(nlp.seeds).length > 0) {
        const pred = await predictRipple(nlp.seeds);
        setPredictionData(pred);

        // Merge GNN predictions into nodes for visual highlighting
        const predMap = new Map(pred.nodes.map((p) => [p.id, p]));
        setNodes((prev) =>
          prev.map((n) => {
            const p = predMap.get(n.id);
            if (p) {
              return {
                ...n,
                predicted_delay: p.predicted_delay,
                risk: p.risk,
                arrival_day: p.arrival_day,
                affected: p.affected,
                is_seed: p.is_seed,
              };
            }
            return n;
          })
        );
      }
    } catch (e) {
      console.error("Simulation failed", e);
    } finally {
      setNlpAnalyzing(false);
    }
  };

  // Reset Simulation
  const handleResetSimulation = () => {
    setActiveNlpResult(null);
    setPredictionData(null);
    setNodes((prev) =>
      prev.map((n) => ({
        ...n,
        predicted_delay: 0,
        risk: 0,
        arrival_day: null,
        affected: false,
        is_seed: false,
      }))
    );
  };

  // Filter affected nodes based on Timeline Slider
  const visibleNodes = nodes.map((n) => {
    if (predictionData && n.affected) {
      const arrives = n.arrival_day ?? 0;
      const isVisibleInTimeline = arrives <= timelineDays;
      return {
        ...n,
        affected: isVisibleInTimeline,
      };
    }
    return n;
  });

  return (
    <div className="app-shell">
      {/* Top Navbar */}
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-logo">AG</div>
          <div>
            <h1>AtmoGraph</h1>
            <p className="subtitle">Supply Chain Ripple Effect Predictor (GNN + NLP)</p>
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
          >
            <span className="status-dot" />
            {dataSource === "neo4j" ? "Neo4j Live" : "Snapshot Mode"}
          </span>
          {predictionData && (
            <span className="gnn-badge">
              ⚡ GNN Active ({predictionData.summary.total_affected} impacted)
            </span>
          )}
        </div>
      </header>

      {/* Control & Timeline Toolbar */}
      <div className="toolbar">
        <input
          type="search"
          className="search-input"
          placeholder="Filter node by name, city, port, sector..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />

        <select
          className="filter-select"
          value={countryFilter}
          onChange={(e) => setCountryFilter(e.target.value)}
        >
          <option value="">All Countries</option>
          {stats?.countries.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>

        <select
          className="filter-select"
          value={industryFilter}
          onChange={(e) => setIndustryFilter(e.target.value)}
        >
          <option value="">All Sectors</option>
          {stats?.industries.map((i) => (
            <option key={i} value={i}>{i}</option>
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

        {/* Timeline Slider (Week 4 Milestone: 30 / 60 / 90 days out) */}
        <div className="timeline-slider-box">
          <span className="slider-label">Projection Horizon: <strong>{timelineDays} Days</strong></span>
          <input
            type="range"
            min={15}
            max={120}
            step={15}
            value={timelineDays}
            onChange={(e) => setTimelineDays(Number(e.target.value))}
            className="slider-range"
          />
        </div>

        {predictionData && (
          <button type="button" className="clear-btn" onClick={handleResetSimulation}>
            Clear Simulation
          </button>
        )}
      </div>

      {/* Main Grid View */}
      <main className="workspace-grid">
        <div className="center-stage">
          <SupplyChainGraph
            nodes={visibleNodes}
            links={links}
            layoutMode={layoutMode}
            searchQuery={searchQuery}
            selectedNodeId={selectedNodeId}
            nodeDetails={nodeDetails}
            onSelectNode={(id) => {
              setSelectedNodeId(id);
              if (id) setActiveTab("inspector");
            }}
          />
        </div>

        {/* Right Sidebar Tabbed Interface */}
        <div className="side-drawer">
          <div className="drawer-tabs">
            <button
              className={`tab-btn ${activeTab === "feed" ? "active" : ""}`}
              onClick={() => setActiveTab("feed")}
            >
              NLP Crisis Radar
            </button>
            <button
              className={`tab-btn ${activeTab === "inspector" ? "active" : ""}`}
              onClick={() => setActiveTab("inspector")}
            >
              Entity Inspector {selectedNodeId ? `(${selectedNodeId})` : ""}
            </button>
          </div>

          <div className="drawer-content">
            {activeTab === "feed" ? (
              <DisruptionFeed
                newsList={newsFeed}
                analyzing={nlpAnalyzing}
                nlpResult={activeNlpResult}
                onSelectHeadline={(h) => runSimulation(h)}
                onSimulateText={(t) => runSimulation(t)}
              />
            ) : (
              <NodeInspector
                details={nodeDetails}
                loading={inspectorLoading}
                onSelectNode={setSelectedNodeId}
              />
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
