import type { GraphResponse, NodeDetailsResponse, StatsResponse } from "./types";

export async function fetchStats(): Promise<StatsResponse> {
  const res = await fetch("/api/stats");
  if (!res.ok) throw new Error(`Failed to fetch stats (${res.status})`);
  return res.json();
}

export async function fetchGraph(params?: {
  country?: string;
  industry?: string;
}): Promise<GraphResponse> {
  const search = new URLSearchParams();
  if (params?.country) search.set("country", params.country);
  if (params?.industry) search.set("industry", params.industry);
  const qs = search.toString();
  const res = await fetch(`/api/graph${qs ? `?${qs}` : ""}`);
  if (!res.ok) throw new Error(`Failed to fetch graph (${res.status})`);
  return res.json();
}

export async function fetchNodeDetails(nodeId: string): Promise<NodeDetailsResponse> {
  const res = await fetch(`/api/nodes/${encodeURIComponent(nodeId)}`);
  if (!res.ok) throw new Error(`Failed to fetch node ${nodeId} (${res.status})`);
  return res.json();
}

