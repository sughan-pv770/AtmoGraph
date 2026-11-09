export interface GraphNode {
  id: string;
  name: string;
  label: "Supplier" | "Manufacturer" | "Port" | "Distributor" | string;
  sub_type: string;
  stage: number;
  tier?: number | null;
  country?: string | null;
  region?: string | null;
  industry?: string | null;
  risk: number;
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
  type: "SUPPLIES" | "SHIPS_VIA" | "SHIPS_TO" | string;
  weight: number;
  days: number;
  lead_days?: number | null;
  transit_days?: number | null;
}

export interface GraphResponse {
  nodes: GraphNode[];
  links: GraphLink[];
  data_source: "neo4j" | "fallback";
}

export interface StatsResponse {
  tier2_suppliers: number;
  tier1_suppliers: number;
  manufacturers: number;
  ports: number;
  distributors: number;
  total_nodes: number;
  at_risk_nodes: number;
  total_edges: number;
  relationships: Record<string, number>;
  countries: string[];
  industries: string[];
  regions: string[];
  data_source: "neo4j" | "fallback";
}

export interface NeighborNode extends GraphNode {
  rel_type: string;
  days: number;
  weight: number;
}

export interface RippleNode extends GraphNode {
  hops: number;
  cumulative_days: number;
}

export interface NodeDetailsResponse {
  node: GraphNode;
  upstream: NeighborNode[];
  downstream: NeighborNode[];
  ripple_impact: RippleNode[];
  data_source: "neo4j" | "fallback";
}

export type LayoutMode = "tiered" | "force";

