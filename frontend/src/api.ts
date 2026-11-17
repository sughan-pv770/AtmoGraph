import type {
  GraphLink,
  GraphNode,
  NewsItem,
  NLPAnalysisResult,
  NodeDetailsResponse,
  RipplePredictionResponse,
  StatsResponse,
} from "./types";

export async function fetchStats(): Promise<StatsResponse> {
  const res = await fetch("/api/stats");
  if (!res.ok) throw new Error("Failed to load statistics");
  return res.json();
}

export async function fetchGraph(country?: string, industry?: string): Promise<{
  nodes: GraphNode[];
  links: GraphLink[];
  data_source: string;
}> {
  const params = new URLSearchParams();
  if (country) params.set("country", country);
  if (industry) params.set("industry", industry);
  const res = await fetch(`/api/graph?${params.toString()}`);
  if (!res.ok) throw new Error("Failed to load graph data");
  return res.json();
}

export async function fetchNodeDetails(id: string): Promise<NodeDetailsResponse> {
  const res = await fetch(`/api/nodes/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error(`Failed to load details for ${id}`);
  return res.json();
}

export async function fetchNewsFeed(): Promise<NewsItem[]> {
  const res = await fetch("/api/news/feed");
  if (!res.ok) return [];
  return res.json();
}

export async function analyzeNews(text: string, applyToGraph = false): Promise<NLPAnalysisResult> {
  const res = await fetch("/api/nlp/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, apply_to_graph: applyToGraph }),
  });
  if (!res.ok) throw new Error("NLP analysis request failed");
  return res.json();
}

export async function predictRipple(seeds: Record<string, number>): Promise<RipplePredictionResponse> {
  const res = await fetch("/api/predict/ripple", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ seeds }),
  });
  if (!res.ok) throw new Error("GNN prediction request failed");
  return res.json();
}
