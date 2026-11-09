import { useEffect, useMemo, useRef } from "react";
import * as d3 from "d3";
import type { GraphLink, GraphNode, LayoutMode, NodeDetailsResponse } from "../types";

interface Props {
  nodes: GraphNode[];
  links: GraphLink[];
  layoutMode: LayoutMode;
  searchQuery: string;
  selectedNodeId: string | null;
  nodeDetails: NodeDetailsResponse | null;
  onSelectNode: (nodeId: string | null) => void;
}

export const STAGE_COLORS: Record<number, string> = {
  0: "#a78bfa", // Tier-2 Supplier (Violet)
  1: "#38bdf8", // Tier-1 Supplier (Sky)
  2: "#34d399", // Manufacturer (Emerald)
  3: "#fbbf24", // Port (Amber)
  4: "#fb7185", // Distributor (Rose)
};

export const STAGE_LABELS: Record<number, string> = {
  0: "Tier-2 Suppliers",
  1: "Tier-1 Suppliers",
  2: "Manufacturers",
  3: "Global Ports",
  4: "Regional Hubs",
};

function nodeRadius(stage: number): number {
  switch (stage) {
    case 0:
      return 6.5;
    case 1:
      return 7.5;
    case 2:
      return 9.5;
    case 3:
      return 11;
    case 4:
      return 12.5;
    default:
      return 8;
  }
}

export default function SupplyChainGraph({
  nodes,
  links,
  layoutMode,
  searchQuery,
  selectedNodeId,
  nodeDetails,
  onSelectNode,
}: Props) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const zoomBehaviorRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);

  const highlightSets = useMemo(() => {
    const upstreamIds = new Set<string>();
    const rippleHops = new Map<string, number>();
    if (selectedNodeId && nodeDetails && nodeDetails.node.id === selectedNodeId) {
      for (const u of nodeDetails.upstream) upstreamIds.add(u.id);
      for (const r of nodeDetails.ripple_impact) rippleHops.set(r.id, r.hops);
    }
    return { upstreamIds, rippleHops };
  }, [selectedNodeId, nodeDetails]);

  const searchMatches = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    if (!q) return null;
    const matched = new Set<string>();
    for (const n of nodes) {
      const hay = `${n.id} ${n.name} ${n.country ?? ""} ${n.region ?? ""} ${n.industry ?? ""}`.toLowerCase();
      if (hay.includes(q)) matched.add(n.id);
    }
    return matched;
  }, [nodes, searchQuery]);

  useEffect(() => {
    const svgEl = svgRef.current;
    if (!svgEl) return;

    const width = svgEl.clientWidth || 1100;
    const height = svgEl.clientHeight || 680;

    const simNodes: GraphNode[] = nodes.map((n) => ({ ...n }));
    const simLinks: GraphLink[] = links.map((l) => ({
      ...l,
      source: typeof l.source === "string" ? l.source : l.source.id,
      target: typeof l.target === "string" ? l.target : l.target.id,
    }));

    const svg = d3.select(svgEl);
    svg.selectAll("*").remove();

    const defs = svg.append("defs");
    const markers = [
      { id: "arrow-default", color: "#475569" },
      { id: "arrow-upstream", color: "#38bdf8" },
      { id: "arrow-ripple", color: "#f97316" },
    ];
    for (const m of markers) {
      defs
        .append("marker")
        .attr("id", m.id)
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 18)
        .attr("refY", 0)
        .attr("markerWidth", 5.5)
        .attr("markerHeight", 5.5)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-5L10,0L0,5")
        .attr("fill", m.color);
    }

    const gRoot = svg.append("g").attr("class", "graph-root");

    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.25, 4])
      .on("zoom", (event) => {
        gRoot.attr("transform", event.transform);
      });

    svg.call(zoom);
    svg.on("click", (event) => {
      if (event.target === svgEl) {
        onSelectNode(null);
      }
    });
    zoomBehaviorRef.current = zoom;

    const padX = 115;
    const stageX = (stage: number) =>
      padX + (stage / 4) * Math.max(400, width - padX * 2);

    if (layoutMode === "tiered") {
      const lanes = gRoot.append("g").attr("class", "stage-lanes");
      for (let s = 0; s <= 4; s++) {
        const x = stageX(s);
        lanes
          .append("line")
          .attr("x1", x)
          .attr("y1", 44)
          .attr("x2", x)
          .attr("y2", height - 24)
          .attr("stroke", "#1e293b")
          .attr("stroke-dasharray", "4 6")
          .attr("stroke-width", 1);

        lanes
          .append("text")
          .attr("x", x)
          .attr("y", 26)
          .attr("text-anchor", "middle")
          .attr("fill", STAGE_COLORS[s])
          .attr("font-size", "11.5px")
          .attr("font-weight", "600")
          .attr("letter-spacing", "0.04em")
          .text(STAGE_LABELS[s].toUpperCase());
      }
    }

    const linkGroup = gRoot.append("g").attr("class", "links");
    const linkSel = linkGroup
      .selectAll<SVGLineElement, GraphLink>("line")
      .data(simLinks, (d) => d.id)
      .join("line")
      .attr("stroke", "#334155")
      .attr("stroke-opacity", 0.55)
      .attr("stroke-width", (d) => Math.max(1, d.weight * 2.1))
      .attr("marker-end", "url(#arrow-default)");

    linkSel.append("title").text((d) => `${d.type} • ${d.days} days (weight ${d.weight})`);

    const nodeGroup = gRoot.append("g").attr("class", "nodes");
    const nodeSel = nodeGroup
      .selectAll<SVGGElement, GraphNode>("g")
      .data(simNodes, (d) => d.id)
      .join("g")
      .attr("class", "node-item")
      .style("cursor", "pointer")
      .on("click", (event, d) => {
        event.stopPropagation();
        onSelectNode(d.id);
      });

    nodeSel
      .append("circle")
      .attr("class", "node-halo")
      .attr("r", (d) => nodeRadius(d.stage) + 5)
      .attr("fill", "none")
      .attr("stroke-width", 2)
      .attr("stroke", "transparent");

    nodeSel
      .append("circle")
      .attr("class", "node-core")
      .attr("r", (d) => nodeRadius(d.stage))
      .attr("fill", (d) => STAGE_COLORS[d.stage] ?? "#38bdf8")
      .attr("stroke", "#0f172a")
      .attr("stroke-width", 1.8);

    nodeSel
      .append("text")
      .attr("class", "node-label")
      .attr("dy", (d) => nodeRadius(d.stage) + 12)
      .attr("text-anchor", "middle")
      .attr("fill", "#cbd5e1")
      .attr("font-size", (d) => (d.stage >= 3 ? "10.5px" : "9px"))
      .attr("font-weight", (d) => (d.stage >= 3 ? "600" : "400"))
      .attr("pointer-events", "none")
      .text((d) => (d.stage >= 2 ? d.name.replace(" Distribution Hub", "") : d.id));

    nodeSel
      .append("title")
      .text(
        (d) =>
          `${d.name} (${d.id})\nType: ${d.sub_type}\nLocation: ${
            d.country ?? d.region ?? "Global"
          }${d.industry ? `\nIndustry: ${d.industry}` : ""}`
      );

    const simulation = d3
      .forceSimulation<GraphNode>(simNodes)
      .force(
        "link",
        d3
          .forceLink<GraphNode, GraphLink>(simLinks)
          .id((d) => d.id)
          .distance(layoutMode === "tiered" ? 95 : 75)
          .strength(layoutMode === "tiered" ? 0.14 : 0.45)
      )
      .force(
        "collide",
        d3.forceCollide<GraphNode>().radius((d) => nodeRadius(d.stage) + (layoutMode === "tiered" ? 11 : 15))
      );

    if (layoutMode === "tiered") {
      simulation
        .force("x", d3.forceX<GraphNode>((d) => stageX(d.stage)).strength(0.92))
        .force("y", d3.forceY<GraphNode>(height / 2 + 12).strength(0.09))
        .force("charge", d3.forceManyBody().strength(-42));
    } else {
      simulation
        .force("charge", d3.forceManyBody().strength(-190))
        .force("center", d3.forceCenter(width / 2, height / 2));
    }

    const drag = d3
      .drag<SVGGElement, GraphNode>()
      .on("start", (event, d) => {
        if (!event.active) simulation.alphaTarget(0.25).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on("drag", (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on("end", (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });

    nodeSel.call(drag);

    simulation.on("tick", () => {
      linkSel
        .attr("x1", (d) => (d.source as GraphNode).x ?? 0)
        .attr("y1", (d) => (d.source as GraphNode).y ?? 0)
        .attr("x2", (d) => (d.target as GraphNode).x ?? 0)
        .attr("y2", (d) => (d.target as GraphNode).y ?? 0);

      nodeSel.attr("transform", (d) => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });

    return () => {
      simulation.stop();
    };
  }, [nodes, links, layoutMode, onSelectNode]);

  // Apply selection, ripple, and search styling reactively without restarting simulation
  useEffect(() => {
    const svgEl = svgRef.current;
    if (!svgEl) return;
    const svg = d3.select(svgEl);
    const { upstreamIds, rippleHops } = highlightSets;
    const hasSelection = Boolean(selectedNodeId);

    svg.selectAll<SVGGElement, GraphNode>("g.node-item").each(function (d) {
      const g = d3.select(this);
      const isSelected = d.id === selectedNodeId;
      const isUpstream = upstreamIds.has(d.id);
      const hop = rippleHops.get(d.id);
      const isRipple = hop !== undefined;
      const matchesSearch = searchMatches ? searchMatches.has(d.id) : true;

      let opacity = 1;
      if (hasSelection && !isSelected && !isUpstream && !isRipple) {
        opacity = 0.16;
      } else if (searchMatches && !matchesSearch) {
        opacity = 0.14;
      }
      g.attr("opacity", opacity);

      const halo = g.select("circle.node-halo");
      if (isSelected) {
        halo.attr("stroke", "#ffffff").attr("stroke-width", 2.6);
      } else if (isRipple) {
        halo
          .attr("stroke", hop === 1 ? "#ef4444" : hop === 2 ? "#f97316" : "#eab308")
          .attr("stroke-width", 2.3);
      } else if (isUpstream) {
        halo.attr("stroke", "#38bdf8").attr("stroke-width", 2);
      } else if (searchMatches && matchesSearch) {
        halo.attr("stroke", "#22d3ee").attr("stroke-width", 2);
      } else if (d.risk >= 0.3) {
        halo.attr("stroke", "#ef4444").attr("stroke-width", 1.8);
      } else {
        halo.attr("stroke", "transparent");
      }
    });

    svg.selectAll<SVGLineElement, GraphLink>("g.links line").each(function (d) {
      const line = d3.select(this);
      const src = typeof d.source === "string" ? d.source : d.source.id;
      const dst = typeof d.target === "string" ? d.target : d.target.id;

      if (!hasSelection) {
        line
          .attr("stroke", "#334155")
          .attr("stroke-opacity", 0.55)
          .attr("stroke-width", Math.max(1, d.weight * 2.1))
          .attr("marker-end", "url(#arrow-default)");
        return;
      }

      const isUpstreamEdge = dst === selectedNodeId && upstreamIds.has(src);
      const isRippleEdge =
        (src === selectedNodeId && rippleHops.has(dst)) ||
        (rippleHops.has(src) &&
          rippleHops.has(dst) &&
          (rippleHops.get(dst) ?? 99) > (rippleHops.get(src) ?? 0));

      if (isRippleEdge) {
        line
          .attr("stroke", "#f97316")
          .attr("stroke-opacity", 0.92)
          .attr("stroke-width", 2.4)
          .attr("marker-end", "url(#arrow-ripple)");
      } else if (isUpstreamEdge) {
        line
          .attr("stroke", "#38bdf8")
          .attr("stroke-opacity", 0.9)
          .attr("stroke-width", 2.2)
          .attr("marker-end", "url(#arrow-upstream)");
      } else {
        line
          .attr("stroke", "#334155")
          .attr("stroke-opacity", 0.08)
          .attr("stroke-width", 1)
          .attr("marker-end", "url(#arrow-default)");
      }
    });
  }, [selectedNodeId, highlightSets, searchMatches, nodes, links]);

  const handleZoom = (factor: number) => {
    if (!svgRef.current || !zoomBehaviorRef.current) return;
    d3.select(svgRef.current)
      .transition()
      .duration(220)
      .call(zoomBehaviorRef.current.scaleBy, factor);
  };

  const handleResetZoom = () => {
    if (!svgRef.current || !zoomBehaviorRef.current) return;
    d3.select(svgRef.current)
      .transition()
      .duration(280)
      .call(zoomBehaviorRef.current.transform, d3.zoomIdentity);
  };

  return (
    <div className="graph-canvas-wrapper">
      <svg ref={svgRef} className="graph-svg" />

      <div className="canvas-controls">
        <button type="button" onClick={() => handleZoom(1.28)} title="Zoom In">
          +
        </button>
        <button type="button" onClick={() => handleZoom(0.78)} title="Zoom Out">
          −
        </button>
        <button type="button" onClick={handleResetZoom} title="Reset View">
          Reset
        </button>
      </div>

      <div className="canvas-legend">
        {Object.entries(STAGE_LABELS).map(([stageStr, label]) => {
          const st = Number(stageStr);
          return (
            <span key={st} className="legend-item">
              <span
                className="legend-dot"
                style={{ backgroundColor: STAGE_COLORS[st] }}
              />
              {label}
            </span>
          );
        })}
      </div>
    </div>
  );
}

