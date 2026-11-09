# AtmoGraph — Supply Chain Ripple Effect Predictor

**AtmoGraph** models global multi-hop supply chains in **Neo4j** and predicts how localized disruptions (port strikes, factory outages, geopolitical bottlenecks) cascade downstream across unrelated industries.

```text
Supplier (Tier 2) ──[:SUPPLIES]──▶ Supplier (Tier 1) ──[:SUPPLIES]──▶ Manufacturer ──[:SHIPS_VIA]──▶ Port ──[:SHIPS_TO]──▶ Distributor
```

---

## Project Structure

```text
atomgraph/
├── generate_graph.py       # Generates & loads multi-hop supply chain graph into Neo4j
├── backend/
│   ├── db.py               # Neo4j connection manager (with fast fallback if Neo4j is stopped)
│   └── main.py             # FastAPI REST endpoints (/api/health, /api/stats, /api/graph, /api/nodes/{id})
├── frontend/               # React + TypeScript + D3.js interactive network dashboard
│   └── src/
│       ├── App.tsx
│       ├── components/
│       │   ├── SupplyChainGraph.tsx
│       │   └── NodeInspector.tsx
├── requirements.txt
├── .env.example
└── start_dev.ps1           # One-click dev launcher for FastAPI + Vite
```

---

## Quick Start

### 1. Start Neo4j Desktop
1. Open **Neo4j Desktop**, press **`▶` (Start)** on the `atomgraph` instance until it says **`RUNNING`**, and **minimize** (do not close) Neo4j Desktop.
2. Copy `.env.example` to `.env` and set your `NEO4J_PASSWORD`.
3. Populate Neo4j (first time or to reset):
   ```powershell
   .\.venv\Scripts\python.exe generate_graph.py --wipe
   ```

### 2. Run Backend & Frontend
Run both servers together:
```powershell
.\start_dev.ps1
```
Or in two separate terminals:
```powershell
# Terminal 1 — FastAPI Backend (http://127.0.0.1:8000)
.\.venv\Scripts\uvicorn.exe backend.main:app --reload --port 8000

# Terminal 2 — React + D3 Dashboard (http://localhost:5173)
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser!

---

## 4-Week Roadmap
- **Week 1 (Complete):** Neo4j schema & Cypher ingestion + FastAPI graph service + React/D3 interactive multi-tier supply chain network UI.
- **Week 2:** NLP Ingestion Engine (`spaCy` / `HuggingFace` NER) to extract disrupted ports/countries/suppliers from live news and update node `risk` scores in Neo4j.
- **Week 3:** PyTorch Geometric Graph Neural Network (node regression) to predict downstream delay days across multi-hop paths.
- **Week 4:** Real-time FastAPI WebSocket streaming, 30/60/90-day timeline slider, Docker containerization, EDA notebook, evaluation report & presentation.

