export interface GraphNode {
  id: string;
  name: string;
  label: "Supplier" | "Manufacturer" | "Port" | "Distributor" | string;
  sub_type?: string;
  stage: number;
  tier?: number | null;
  country?: string | null;
  region?: string | null;
  industry?: string | null;
  risk: number;
  predicted_delay?: number;
  arrival_day?: number | null;
  affected?: boolean;
  is_seed?: boolean;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
}

export interface GraphLink {
  id: string;
  source: string | GraphNode;
  target: string | GraphNode;
  weight: number;
  days: number;
  type?: string;
}

export type LayoutMode = "tiered" | "force";

export interface StatsResponse {
  tier2_suppliers: number;
  tier1_suppliers: number;
  manufacturers: number;
  ports: number;
  distributors: number;
  total_nodes: number;
  total_edges: number;
  at_risk_nodes: number;
  countries: string[];
  industries: string[];
  regions: string[];
  data_source: string;
}

export interface NodeDetailsResponse {
  node: GraphNode;
  upstream: Array<GraphNode & { days: number; weight: number; share: number }>;
  downstream: Array<GraphNode & { days: number; weight: number; share: number }>;
  ripple_impact: Array<GraphNode & { hops: number; cumulative_days: number }>;
}

export interface NewsItem {
  id: number;
  source: string;
  text: string;
  labels: {
    ports?: string[];
    countries?: string[];
    industries?: string[];
    regions?: string[];
    event: string;
    recovery?: boolean;
  };
}

export interface NLPAnalysisResult {
  text: string;
  entities: Array<{ text: string; label: string; source: string }>;
  matched: {
    ports: string[];
    countries: string[];
    industries: string[];
    regions: string[];
    nodes: string[];
  };
  event_type: string;
  severity: number;
  is_recovery: boolean;
  seeds: Record<string, number>;
  explanation: string;
  ner_engine: string;
}

export interface RipplePredictionResponse {
  engine: string;
  nodes: Array<{
    id: string;
    name: string;
    stage: number;
    label: string;
    predicted_delay: number;
    risk: number;
    arrival_day: number | null;
    is_seed: boolean;
    affected: boolean;
  }>;
  summary: {
    total_affected: number;
    max_delay_days: number;
    seed_count: number;
  };
}
