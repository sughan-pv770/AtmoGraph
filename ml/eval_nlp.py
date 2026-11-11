"""
Evaluate the NLP engine against the labelled headline set (data/news_headlines.json).

  python -m ml.eval_nlp

Reports micro precision / recall / F1 for port, country and industry extraction,
event-type accuracy, recovery-detection accuracy, and the share of headlines that
produce at least one graph seed. Writes ml/artifacts/nlp_metrics.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.nlp import analyze, spacy_available
from generate_graph import generate
from ml.graph import from_generator

ROOT = Path(__file__).resolve().parent.parent


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    g = from_generator(*generate())
    items = json.loads((ROOT / "data" / "news_headlines.json").read_text(encoding="utf-8"))
    counts = {k: [0, 0, 0] for k in ("ports", "countries", "industries")}
    event_ok = recovery_ok = seeded = 0
    errors = []
    for it in items:
        res = analyze(it["text"], g)
        lab = it["labels"]
        for k in counts:
            pred, gold = set(res["matched"][k]), set(lab.get(k, []))
            counts[k][0] += len(pred & gold)
            counts[k][1] += len(pred - gold)
            counts[k][2] += len(gold - pred)
            if pred != gold:
                errors.append({"id": it["id"], "field": k, "pred": sorted(pred), "gold": sorted(gold)})
        event_ok += res["event_type"] == lab["event"]
        if res["event_type"] != lab["event"]:
            errors.append({"id": it["id"], "field": "event", "pred": res["event_type"], "gold": lab["event"]})
        recovery_ok += res["is_recovery"] == bool(lab.get("recovery", False))
        seeded += bool(res["seeds"])

    n = len(items)
    tp = sum(c[0] for c in counts.values())
    fp = sum(c[1] for c in counts.values())
    fn = sum(c[2] for c in counts.values())
    metrics = {
        "headlines": n,
        "ner_engine": "spacy+gazetteer" if spacy_available() else "gazetteer",
        "entity_micro": prf(tp, fp, fn),
        **{f"entity_{k}": prf(*v) for k, v in counts.items()},
        "event_accuracy": round(event_ok / n, 4),
        "recovery_accuracy": round(recovery_ok / n, 4),
        "headlines_with_seeds": round(seeded / n, 4),
        "errors": errors,
    }
    out = ROOT / "ml" / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    (out / "nlp_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "errors"}, indent=2))
    for e in errors:
        print("  mismatch:", e)


if __name__ == "__main__":
    main()

