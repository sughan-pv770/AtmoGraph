import { STAGE_COLORS } from "./SupplyChainGraph";
import type { NodeDetailsResponse } from "../types";

interface Props {
  details: NodeDetailsResponse | null;
  loading: boolean;
  onSelectNode: (nodeId: string | null) => void;
}

export default function NodeInspector({ details, loading, onSelectNode }: Props) {
  if (loading) {
    return (
      <aside className="inspector-panel">
        <div className="inspector-empty">Loading node telemetry...</div>
      </aside>
    );
  }

  if (!details) {
    return (
      <aside className="inspector-panel">
        <div className="inspector-placeholder">
          <h3>Supply Chain Inspector</h3>
          <p>
            Click any <strong>Supplier</strong>, <strong>Manufacturer</strong>,{" "}
            <strong>Port</strong>, or <strong>Regional Hub</strong> on the graph to
            trace its upstream inputs and multi-hop downstream ripple impact.
          </p>
          <div className="inspector-tips">
            <div className="tip-row">
              <span className="tip-swatch" style={{ background: "#38bdf8" }} />
              <span>Blue arrows = Direct upstream suppliers</span>
            </div>
            <div className="tip-row">
              <span className="tip-swatch" style={{ background: "#f97316" }} />
              <span>Orange/Red glow = Downstream multi-hop ripple path</span>
            </div>
          </div>
        </div>
      </aside>
    );
  }

  const { node, upstream, downstream, ripple_impact } = details;
  const riskPct = Math.round(node.risk * 100);
  const badgeColor = STAGE_COLORS[node.stage] ?? "#38bdf8";

  return (
    <aside className="inspector-panel">
      <div className="inspector-header">
        <div>
          <span
            className="node-type-pill"
            style={{
              backgroundColor: `${badgeColor}22`,
              color: badgeColor,
              borderColor: `${badgeColor}55`,
            }}
          >
            {node.sub_type}
          </span>
          <h2>{node.name}</h2>
          <span className="node-id-mono">{node.id}</span>
        </div>
        <button
          type="button"
          className="close-btn"
          onClick={() => onSelectNode(null)}
          title="Clear Selection"
        >
          ✕
        </button>
      </div>

      <div className="meta-grid">
        <div className="meta-card">
          <span className="meta-label">Location</span>
          <span className="meta-val">{node.country ?? node.region ?? "Global"}</span>
        </div>
        <div className="meta-card">
          <span className="meta-label">Sector</span>
          <span className="meta-val">{node.industry ?? "Logistics"}</span>
        </div>
        <div className="meta-card full-span">
          <div className="risk-row">
            <span className="meta-label">Disruption Risk Score</span>
            <span className="meta-val">{riskPct}%</span>
          </div>
          <div className="risk-bar-bg">
            <div
              className="risk-bar-fill"
              style={{
                width: `${Math.max(4, riskPct)}%`,
                backgroundColor:
                  node.risk >= 0.6
                    ? "#ef4444"
                    : node.risk >= 0.3
                    ? "#f59e0b"
                    : "#10b981",
              }}
            />
          </div>
        </div>
      </div>

      <div className="ripple-summary-banner">
        <div>
          <span className="ripple-count">{ripple_impact.length}</span>
          <span className="ripple-caption">Downstream Entities Reachable</span>
        </div>
        <div>
          <span className="ripple-count">{upstream.length}</span>
          <span className="ripple-caption">Direct Upstream Feeders</span>
        </div>
      </div>

      <div className="inspector-sections">
        <section>
          <h4>Downstream Ripple Cascade ({ripple_impact.length})</h4>
          {ripple_impact.length === 0 ? (
            <p className="muted-note">Terminal distribution hub (no further hops).</p>
          ) : (
            <ul className="entity-list">
              {ripple_impact.slice(0, 25).map((item) => (
                <li key={item.id}>
                  <button
                    type="button"
                    className="entity-row-btn"
                    onClick={() => onSelectNode(item.id)}
                  >
                    <div>
                      <span className="entity-name">{item.name}</span>
                      <span className="entity-sub">
                        {item.sub_type} • {item.country ?? item.region}
                      </span>
                    </div>
                    <div className="hop-badges">
                      <span className={`hop-pill hop-${item.hops}`}>
                        Hop {item.hops}
                      </span>
                      <span className="days-tag">+{item.cumulative_days}d</span>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section>
          <h4>Direct Outbound Routes ({downstream.length})</h4>
          {downstream.length === 0 ? (
            <p className="muted-note">None</p>
          ) : (
            <ul className="entity-list">
              {downstream.map((d) => (
                <li key={d.id}>
                  <button
                    type="button"
                    className="entity-row-btn"
                    onClick={() => onSelectNode(d.id)}
                  >
                    <div>
                      <span className="entity-name">{d.name}</span>
                      <span className="entity-sub">{d.rel_type}</span>
                    </div>
                    <span className="days-tag">{d.days}d</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section>
          <h4>Direct Upstream Suppliers ({upstream.length})</h4>
          {upstream.length === 0 ? (
            <p className="muted-note">Origin tier (raw materials / components).</p>
          ) : (
            <ul className="entity-list">
              {upstream.map((u) => (
                <li key={u.id}>
                  <button
                    type="button"
                    className="entity-row-btn"
                    onClick={() => onSelectNode(u.id)}
                  >
                    <div>
                      <span className="entity-name">{u.name}</span>
                      <span className="entity-sub">
                        {u.country ?? u.region} • {u.industry ?? u.rel_type}
                      </span>
                    </div>
                    <span className="days-tag">{u.days}d lead</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </aside>
  );
}

