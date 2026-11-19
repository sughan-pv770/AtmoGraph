# AtmoGraph: Supply Chain Ripple Effect Predictor
## Final Project Technical Documentation & Report
**Organization:** Infotact Solutions  
**Role:** Data Associate L1. Intern (Data Science & Machine Learning Domain)  
**Assigned Project:** Project 2 — "AtmoGraph"  

---

### Executive Summary
Traditional supply chain predictive models rely on linear, isolated time-series data. They fail to capture complex global dependency graphs and cannot predict how a localized shock (such as a port strike in Rotterdam or a semiconductor shortage in Taiwan) ripples across unrelated manufacturing sectors and distribution hubs worldwide.

**AtmoGraph** is an enterprise-grade Graph Neural Network and NLP-powered intelligence platform that models global supply chains as a directed graph, ingests real-time news to detect disruption events, and predicts downstream delay cascades across multi-hop node paths.

---

### System Architecture & Key Modules

1. **Graph Database & Knowledge Graph (Neo4j / Network Engine)**
   - Schema: `Supplier (Tier 2) -> Supplier (Tier 1) -> Manufacturer -> Port -> Regional Distributor`
   - Scale: 1,088 nodes and 2,157 directed relationships across 12 countries, 7 manufacturing industries, 12 international container ports, and 6 continental hubs.
   - Nodes contain features such as safety buffer days, country, sector, and risk ratings; edges contain lead days, transit times, and shipment dependency weights.

2. **NLP Ingestion Engine (spaCy + Domain Gazetteer)**
   - Extracts GPE, LOC, and FAC entities from unstructured news feeds.
   - Disruption event classifier and severity estimator calculates impact severity and maps free-text headlines directly to graph seed nodes.
   - Evaluated on benchmark headlines: 99.2% entity F1-score and 100% seed mapping coverage.

3. **Graph Neural Network (PyTorch Geometric — RippleGNN)**
   - Custom 5-layer directed Message Passing Neural Network (`RippleConv`) with Layer Normalization and dual sum/max aggregation.
   - Accurately models the physical direction of supply propagation.
   - Multi-task head outputs:
     * Log delay regression: $\log(1 + \text{delay days})$
     * Node disruption risk score $[0, 1]$
     * Wave arrival day estimation

4. **Real-Time Visualization & Predictive UI (React + D3.js + FastAPI)**
   - Interactive SVG canvas with Tiered Flow and Force-directed layouts handling 1,000+ nodes seamlessly.
   - 30 / 60 / 90-day timeline slider projecting wave propagation horizons dynamically.
   - Live Disruption Radar allowing interactive simulation of any news wire or custom crisis input.
   - Deep multi-hop inspector revealing full upstream supplier roots and downstream ripple cascades.

---

### Benchmark & Model Evaluation Results

Evaluated across 40 training networks, 10 held-out unseen test networks, and the deployed 1,088-node global graph:

| Model Architecture | Delay MAE (Days) | $R^2$ Score | Disruption F1 | Arrival MAE (Days) |
| :--- | :---: | :---: | :---: | :---: |
| **Seeds-Only Baseline** | 2.66 d | 0.669 | 0.635 | — |
| **Hop-Decay Heuristic** | 3.09 d | 0.571 | 0.722 | 0.05 d |
| **No-Graph MLP** | 2.76 d | 0.756 | 0.617 | 0.94 d |
| **RippleGNN (Ours)** | **0.49 d** | **0.990** | **0.985** | **0.54 d** |

The Graph Neural Network demonstrates significant predictive superiority over non-graph ML and heuristic baselines, achieving near-perfect ranking and delay regression across multi-hop paths.

---

### Deliverables Checklist (Infotact Requirements)
- [x] **Source Code**: Full production codebase with FastAPI backend, PyG ML pipeline, and React + D3.js frontend.
- [x] **Project Documentation**: Comprehensive system architecture and implementation details.
- [x] **Model Evaluation Results**: Quantitative metrics, error analysis, and benchmark comparisons.
- [x] **Presentation Deck**: 16:9 executive PowerPoint slide deck (`docs/AtmoGraph_Presentation.pptx`).
- [x] **Test Suite**: Automated Pytest verification for generator, NLP, GNN inference, and REST APIs.
- [x] **Version Control**: Modular, descriptive git commits pushed to the repository.

