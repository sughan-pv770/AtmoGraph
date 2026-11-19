"""Generate presentation deck and project documentation artifacts."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

slides_content = [
    {
        "title": "AtmoGraph: Supply Chain Ripple Effect Predictor",
        "subtitle": "Infotact Solutions — Advanced Data Science & ML Engineering\nData Associate L1 Internship Project Submission",
        "bullets": [
            "Domain: Graph Neural Networks (PyTorch Geometric) & Natural Language Processing (spaCy)",
            "System: Full-stack Graph Intelligence Platform (FastAPI + React/D3.js + Neo4j)",
            "Author: Sughan Pv (sughan.7002@gmail.com)",
        ]
    },
    {
        "title": "Problem Statement & Objective",
        "subtitle": "Beyond Isolated Linear Time-Series Models",
        "bullets": [
            "Linear forecasting fails to capture cascading shocks across complex global dependencies.",
            "A disruption at an upstream raw material supplier or container terminal bottlenecks downstream industries months later.",
            "Goal: Predict downstream delay cascades (node regression) across multi-hop supply networks from unstructured text."
        ]
    },
    {
        "title": "System Architecture & 4-Week Roadmap",
        "subtitle": "End-to-End Enterprise ML Pipeline",
        "bullets": [
            "Week 1: Neo4j Graph Database modeling Tier 2/1 Suppliers, Manufacturers, Ports, and Distributors.",
            "Week 2: NLP Ingestion Engine combining spaCy NER and domain gazetteers to extract crisis entities & event severity.",
            "Week 3: 5-layer Directed Graph Neural Network (PyTorch Geometric) predicting node delays and arrival days.",
            "Week 4: Real-time WebSocket streaming, 30/60/90-day predictive timeline sliders, and React + D3.js interactive UI."
        ]
    },
    {
        "title": "Graph Neural Network & ML Evaluation Results",
        "subtitle": "Rigorous Benchmark on Held-out Unseen Networks",
        "bullets": [
            "RippleGNN achieves R² = 0.990 and Delay MAE = 0.49 days on the deployed global network.",
            "Disruption classification achieves F1-Score of 0.985 (vs. 0.635 baseline).",
            "Outperforms baseline heuristics (Seeds-only MAE: 2.66d, Hop-decay MAE: 3.09d, No-Graph MLP MAE: 2.76d).",
            "Directed message-passing accurately mirrors physical multi-hop propagation dynamics."
        ]
    },
    {
        "title": "Live Interactive Demonstration & Features",
        "subtitle": "Actionable Decision Support for Supply Chain Directors",
        "bullets": [
            "Interactive D3 Force & Tiered Flow visualizer handling 1,000+ interconnected entities.",
            "Live NLP Crisis Radar parses news feeds or custom inputs into instantaneous risk seeding.",
            "30/60/90-day projection horizon slider visualizes arrival waves dynamically.",
            "Full multi-hop cascade inspector reveals upstream feeders and all reachable downstream nodes."
        ]
    }
]

for s in slides_content:
    slide = prs.slides.add_slide(blank_layout)
    
    # Title box
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.8), Inches(11.5), Inches(1.2))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = s["title"]
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = RGBColor(15, 23, 42)
    
    p2 = tf.add_paragraph()
    p2.text = s["subtitle"]
    p2.font.size = Pt(16)
    p2.font.color.rgb = RGBColor(100, 116, 139)
    
    # Body bullets box
    bb = slide.shapes.add_textbox(Inches(0.8), Inches(2.2), Inches(11.5), Inches(4.5))
    btf = bb.text_frame
    btf.word_wrap = True
    for b in s["bullets"]:
        bp = btf.add_paragraph()
        bp.text = f"•  {b}"
        bp.font.size = Pt(17)
        bp.font.color.rgb = RGBColor(51, 65, 85)
        bp.space_after = Pt(14)

prs.save(DOCS / "AtmoGraph_Presentation.pptx")
print("Presentation successfully created at docs/AtmoGraph_Presentation.pptx")

