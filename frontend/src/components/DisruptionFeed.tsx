import React from "react";
import type { NewsItem, NLPAnalysisResult } from "../types";

interface Props {
  newsList: NewsItem[];
  analyzing: boolean;
  nlpResult: NLPAnalysisResult | null;
  onSelectHeadline: (headline: string) => void;
  onSimulateText: (text: string) => void;
}

export default function DisruptionFeed({
  newsList,
  analyzing,
  nlpResult,
  onSelectHeadline,
  onSimulateText,
}: Props) {
  const [customText, setCustomText] = React.useState("");

  const handleCustomSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!customText.trim()) return;
    onSimulateText(customText.trim());
  };

  return (
    <div className="news-feed-panel">
      <div className="panel-title-bar">
        <h3>Live Disruption Radar (NLP Engine)</h3>
        <span className="badge-pill">{newsList.length} Feed Items</span>
      </div>

      <form onSubmit={handleCustomSubmit} className="custom-input-box">
        <textarea
          rows={2}
          placeholder="Paste or type any custom crisis event (e.g., 'Dockworkers strike at Rotterdam port halts logistics')..."
          value={customText}
          onChange={(e) => setCustomText(e.target.value)}
        />
        <button type="submit" disabled={analyzing || !customText.trim()} className="btn-primary">
          {analyzing ? "Extracting Entities..." : "Simulate Live NLP Crisis"}
        </button>
      </form>

      {nlpResult && (
        <div className="nlp-result-card">
          <div className="nlp-header">
            <strong>NLP Extraction Results ({nlpResult.ner_engine})</strong>
            <span className={`tag-pill event-${nlpResult.event_type}`}>
              {nlpResult.event_type}
            </span>
          </div>
          <p className="nlp-explanation">{nlpResult.explanation}</p>
          <div className="entity-chips">
            {nlpResult.matched.ports.map((p) => (
              <span key={p} className="chip chip-port">⚓ Port: {p}</span>
            ))}
            {nlpResult.matched.countries.map((c) => (
              <span key={c} className="chip chip-country">🌐 Country: {c}</span>
            ))}
            {nlpResult.matched.industries.map((ind) => (
              <span key={ind} className="chip chip-industry">⚙ {ind}</span>
            ))}
          </div>
          <div className="severity-row">
            <span>Disruption Severity:</span>
            <strong>{Math.round(nlpResult.severity * 100)}%</strong>
            <span style={{ marginLeft: "auto", fontSize: "12px", color: "#64748b" }}>
              {Object.keys(nlpResult.seeds).length} graph nodes seeded
            </span>
          </div>
        </div>
      )}

      <div className="feed-items-scroll">
        <span className="section-label">SELECT RECENT GLOBAL WIRES</span>
        {newsList.map((item) => (
          <div
            key={item.id}
            className="headline-card"
            onClick={() => {
              setCustomText(item.text);
              onSelectHeadline(item.text);
            }}
          >
            <div className="headline-meta">
              <span className="source-name">{item.source}</span>
              <span className={`badge-event badge-${item.labels.event}`}>
                {item.labels.event}
              </span>
            </div>
            <p className="headline-text">"{item.text}"</p>
          </div>
        ))}
      </div>
    </div>
  );
}

