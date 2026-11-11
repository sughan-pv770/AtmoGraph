"""
AtmoGraph NLP engine: turn a news headline into disruption seeds on the graph.

Pipeline
  1. Named-entity recognition with spaCy (GPE / LOC / FAC / ORG) when the
     `en_core_web_sm` model is installed.
  2. A gazetteer built from the live graph (port names, countries, regions,
     industries) plus aliases and demonyms ("Taiwanese" -> Taiwan, "chipmaker"
     -> Semiconductors, "Port of LA" -> Los Angeles). This resolves spaCy's free
     text entities to concrete graph nodes and also catches what spaCy misses.
  3. Event classification + severity scoring from a keyword lexicon with
     intensifiers ("massive", "indefinite"), mitigators ("partial", "brief")
     and recovery cues ("strike ends", "port reopens").
  4. Mapping to seeds {node_id: severity}: ports, suppliers/manufacturers in the
     affected country and/or industry, explicitly named facilities, or the
     regional distribution hub.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from ml.graph import SupplyGraph

COUNTRY_ALIASES = {
    "china": "China", "chinese": "China", "prc": "China",
    "taiwan": "Taiwan", "taiwanese": "Taiwan",
    "south korea": "South Korea", "korea": "South Korea", "korean": "South Korea", "south korean": "South Korea",
    "japan": "Japan", "japanese": "Japan",
    "vietnam": "Vietnam", "vietnamese": "Vietnam", "viet nam": "Vietnam",
    "india": "India", "indian": "India",
    "germany": "Germany", "german": "Germany",
    "netherlands": "Netherlands", "dutch": "Netherlands", "holland": "Netherlands",
    "mexico": "Mexico", "mexican": "Mexico",
    "usa": "USA", "united states": "USA", "america": "USA", "american": "USA", "u.s.": "USA",
    "malaysia": "Malaysia", "malaysian": "Malaysia",
    "thailand": "Thailand", "thai": "Thailand",
    "belgium": "Belgium", "belgian": "Belgium",
    "singapore": "Singapore", "singaporean": "Singapore",
}
CASE_SENSITIVE_COUNTRY = {"US": "USA", "U.S.": "USA"}

PORT_ALIASES = {
    "port of la": "Los Angeles", "la port": "Los Angeles", "san pedro bay": "Los Angeles",
    "jnpt": "Nhava Sheva", "jawaharlal nehru port": "Nhava Sheva", "mumbai port": "Nhava Sheva",
    "madras port": "Chennai", "klang": "Port Klang", "yantian": "Shenzhen",
}

INDUSTRY_ALIASES = {
    "semiconductor": "Semiconductors", "semiconductors": "Semiconductors", "chip": "Semiconductors",
    "chips": "Semiconductors", "chipmaker": "Semiconductors", "chipmakers": "Semiconductors",
    "foundry": "Semiconductors", "foundries": "Semiconductors", "wafer": "Semiconductors",
    "battery": "Batteries", "batteries": "Batteries", "lithium": "Batteries", "cathode": "Batteries",
    "electronics": "Electronics", "electronic": "Electronics", "smartphone": "Electronics",
    "smartphones": "Electronics", "display panel": "Electronics", "display panels": "Electronics",
    "plastic": "Plastics", "plastics": "Plastics", "resin": "Plastics", "polymer": "Plastics",
    "petrochemical": "Plastics", "petrochemicals": "Plastics",
    "steel": "Metals", "aluminium": "Metals", "aluminum": "Metals", "copper": "Metals",
    "metal": "Metals", "metals": "Metals", "nickel": "Metals",
    "textile": "Textiles", "textiles": "Textiles", "garment": "Textiles", "garments": "Textiles",
    "apparel": "Textiles", "cotton": "Textiles",
    "auto parts": "Automotive Parts", "automotive": "Automotive Parts", "car parts": "Automotive Parts",
    "automaker": "Automotive Parts", "automakers": "Automotive Parts", "carmaker": "Automotive Parts",
}

REGION_ALIASES = {
    "north america": "North America", "europe": "Europe", "european": "Europe", "eu": "Europe",
    "south asia": "South Asia", "east asia": "East Asia", "latin america": "Latin America",
    "middle east": "Middle East", "gulf": "Middle East",
}

EVENT_LEXICON: dict[str, tuple[tuple[str, ...], float]] = {
    "natural_disaster": (("earthquake", "typhoon", "cyclone", "hurricane", "flood", "flooding",
                          "tsunami", "landslide", "storm", "wildfire"), 0.85),
    "fire_explosion": (("fire", "blaze", "explosion", "blast"), 0.80),
    "labor_strike": (("strike", "strikes", "walkout", "industrial action", "labor dispute",
                      "labour dispute", "union protest", "work stoppage"), 0.75),
    "closure": (("shut down", "shutdown", "closed", "closure", "halts", "halted", "suspends",
                 "suspended", "lockdown"), 0.70),
    "geopolitical": (("sanction", "sanctions", "export ban", "embargo", "blockade", "tariff",
                      "tariffs", "conflict", "war", "export controls"), 0.65),
    "cyberattack": (("cyberattack", "cyber attack", "ransomware", "hack", "hacked", "it outage"), 0.60),
    "power_outage": (("power outage", "blackout", "power cuts", "grid failure"), 0.55),
    "shortage": (("shortage", "shortages", "scarcity", "supply crunch", "rationing"), 0.50),
    "congestion": (("congestion", "backlog", "bottleneck", "delays", "delayed", "queue"), 0.45),
}
INTENSIFIERS = ("massive", "major", "severe", "indefinite", "nationwide", "worst", "crippling",
                "record", "devastating", "total", "week-long", "prolonged")
MITIGATORS = ("minor", "partial", "brief", "short", "small", "limited", "temporary", "localized")
RECOVERY = ("ends", "ended", "resumes", "resumed", "reopens", "reopened", "lifted", "called off",
            "restored", "recovers", "back to normal", "eases")
PORT_CONTEXT = ("port", "ports", "dock", "docks", "terminal", "terminals", "harbor", "harbour",
                "dockworkers", "longshore", "shipping", "container")


@lru_cache(maxsize=1)
def _spacy_model():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:  # model or library not installed -> gazetteer only
        return None


def spacy_available() -> bool:
    return _spacy_model() is not None


def _find(phrases: dict[str, str], text_lower: str) -> dict[str, str]:
    """Whole-word phrase matches -> {canonical: matched_text}. Longest phrases first."""
    found: dict[str, str] = {}
    taken: list[tuple[int, int]] = []
    for phrase in sorted(phrases, key=len, reverse=True):
        for m in re.finditer(r"(?<![\w.])" + re.escape(phrase) + r"(?![\w])", text_lower):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            found.setdefault(phrases[phrase], m.group(0))
    return found


def _has_any(words: tuple[str, ...], text_lower: str) -> list[str]:
    return [w for w in words if re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", text_lower)]


def analyze(text: str, g: SupplyGraph) -> dict[str, Any]:
    text = (text or "").strip()
    low = text.lower()

    # --- 1. spaCy NER --------------------------------------------------------
    entities: list[dict[str, str]] = []
    nlp = _spacy_model()
    if nlp is not None and text:
        for ent in nlp(text).ents:
            if ent.label_ in {"GPE", "LOC", "FAC", "ORG", "NORP"}:
                entities.append({"text": ent.text, "label": ent.label_, "source": "spacy"})

    # --- 2. gazetteer resolution ----------------------------------------------
    port_names = {n["name"].lower(): n["name"] for n in g.nodes if n["label"] == "Port"}
    port_phrases = {**port_names, **{k: v for k, v in PORT_ALIASES.items() if v.lower() in port_names}}
    # Generic names like "Singapore" are both a port and a country; treat as port only with port context.
    ports = _find(port_phrases, low)
    countries = _find(COUNTRY_ALIASES, low)
    for token, canon in CASE_SENSITIVE_COUNTRY.items():
        if re.search(r"(?<![\w.])" + re.escape(token) + r"(?![\w])", text):
            countries.setdefault(canon, token)
    industries = _find(INDUSTRY_ALIASES, low)
    regions = _find(REGION_ALIASES, low)
    port_context = bool(_has_any(PORT_CONTEXT, low))
    for name in list(ports):
        if name in countries and not port_context:
            ports.pop(name)  # "Singapore economy" != "Port of Singapore"
    for name in list(countries):
        if name in ports and port_context and not _supplier_country(g, name):
            countries.pop(name)

    named_nodes = []
    for m in re.finditer(r"\b(SUP[12]-\d{4}|MFR-\d{4}|PORT-\d{3}|DIST-\d{3})\b", text, re.I):
        if m.group(1).upper() in g.index:
            named_nodes.append(m.group(1).upper())
    for m in re.finditer(r"\b(supplier t[12]-\d{4}|manufacturer \d{4})\b", low):
        for n in g.nodes:
            if n["name"].lower() == m.group(1):
                named_nodes.append(n["id"])

    for canon, surface in {**ports, **countries, **industries, **regions}.items():
        if not any(e["text"].lower() == surface.lower() for e in entities):
            entities.append({"text": surface, "label": "GAZETTEER", "source": "gazetteer"})

    # --- 3. event + severity --------------------------------------------------
    event_type, severity, matched_kw = "general_disruption", 0.40, []
    for etype, (words, base) in EVENT_LEXICON.items():
        hits = _has_any(words, low)
        if hits:
            event_type, severity, matched_kw = etype, base, hits
            break
    intens = _has_any(INTENSIFIERS, low)
    mitig = _has_any(MITIGATORS, low)
    recov = _has_any(RECOVERY, low)
    severity = severity + 0.12 * min(len(intens), 2) - 0.18 * min(len(mitig), 2)
    is_recovery = bool(recov)
    if is_recovery:
        severity *= 0.15
    severity = round(max(0.05, min(1.0, severity)), 3)

    # --- 4. map to graph seeds ------------------------------------------------
    seeds: dict[str, float] = {}

    def add(node_id: str, sev: float) -> None:
        seeds[node_id] = round(max(seeds.get(node_id, 0.0), min(1.0, sev)), 3)

    scope = []
    for name in ports:
        for n in g.nodes:
            if n["label"] == "Port" and n["name"] == name:
                add(n["id"], severity)
        scope.append(f"port of {name}")
    for nid in named_nodes:
        add(nid, severity)
        scope.append(nid)
    ind_set = set(industries)
    for c in countries:
        country_nodes = [n for n in g.nodes if n.get("country") == c and n["stage"] <= 2]
        if ind_set:
            country_nodes = [n for n in country_nodes if n.get("industry") in ind_set]
        factor = 1.0 if event_type in {"natural_disaster", "geopolitical", "power_outage"} else 0.8
        for n in country_nodes:
            add(n["id"], severity * factor)
        if port_context and not ports:
            for n in g.nodes:
                if n["label"] == "Port" and n.get("country") == c:
                    add(n["id"], severity)
                    scope.append(f"ports in {c}")
        scope.append(f"{', '.join(sorted(ind_set)) + ' ' if ind_set else ''}suppliers in {c}")
    if ind_set and not countries:
        for n in g.nodes:
            if n.get("industry") in ind_set and n["stage"] <= 1:
                add(n["id"], severity * 0.6)
        scope.append(f"global {', '.join(sorted(ind_set))} suppliers")
    if regions and not seeds:
        for n in g.nodes:
            if n["label"] == "Distributor" and n.get("region") in regions:
                add(n["id"], severity)
        scope.append(f"{', '.join(regions)} distribution")

    if is_recovery:
        explanation = "Recovery news detected — residual risk only."
    elif seeds:
        explanation = (f"{event_type.replace('_', ' ').title()} (severity {severity:.2f}) "
                       f"affecting {'; '.join(dict.fromkeys(scope))}: {len(seeds)} node(s) seeded.")
    else:
        explanation = "No supply chain entity from the network was recognised in this text."

    return {
        "text": text,
        "entities": entities,
        "matched": {"ports": sorted(ports), "countries": sorted(countries),
                    "industries": sorted(industries), "regions": sorted(regions),
                    "nodes": sorted(set(named_nodes))},
        "event_type": event_type,
        "event_keywords": matched_kw,
        "severity": severity,
        "is_recovery": is_recovery,
        "seeds": seeds,
        "explanation": explanation,
        "ner_engine": "spacy+gazetteer" if nlp is not None else "gazetteer",
    }


def _supplier_country(g: SupplyGraph, country: str) -> bool:
    return any(n.get("country") == country and n["stage"] <= 2 for n in g.nodes)

