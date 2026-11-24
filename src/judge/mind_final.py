"""Generate scoped conclusions and bind each decision to its registered numeric evidence."""
import argparse
import csv
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.present.mind_final import CONFIG, CONCLUSIONS, STAGING, plan, load_sources, render_tex
from tools.common import head, payload_sha256, snapshot, status_paths
from tools.rv import write_result


def conclusions(indicators):
    values = {entry["id"]: entry["value"] for entry in indicators}

    def supported(prefix, controls):
        bounds = {}
        for obj, left, right in controls:
            low = f"I-{prefix}_{left}-low"
            high = f"I-{prefix}_{left}-high"
            value = f"I-{prefix}_{left}-estimate"
            if values[low] > 0:
                result = "higher than"
            elif values[high] < 0:
                result = "lower than"
            else:
                result = "inconclusive against"
            bounds[obj] = (result, right, [value, low, high])
        return bounds

    def statements(bounds):
        return ", ".join(f"{result} {label}" for result, label, _ in bounds.values())

    def evidence(*bounds):
        return [entry for compared in bounds for _, _, ids in compared.values() for entry in ids]

    base = [("pop", "popularity", "24-hour popularity"),
            ("recent", "recent_click", "recent-click TF-IDF")]
    tower_official = supported("official_tower_minus", base)
    tower_recall = supported("recall_tower_minus", base)
    din_official = supported("official_din_minus", base + [("tower", "tower_s0", "the encoder tower")])
    condition = [("tower", "din_minus_tower_s0", "the fixed tower order")]
    pool = supported("pool", condition)
    end = supported("end", condition)

    return [
        {"id": CONCLUSIONS[0], "scope": "small-dev official nDCG@10 and temporal Recall@100",
         "statement_en": f"Tower official nDCG is {statements(tower_official)}. "
                         f"Tower temporal recall is {statements(tower_recall)}.",
         "evidence": evidence(tower_official, tower_recall) + ["I-denom-official", "I-retrieval-tower-clicks-mean"]},
        {"id": CONCLUSIONS[1], "scope": "small-dev original mixed-label exposure lists",
         "statement_en": f"DIN official nDCG is {statements(din_official)}.",
         "evidence": evidence(din_official) + ["I-denom-official"]},
        {"id": CONCLUSIONS[2], "scope": "small-dev fixed candidates; conditional and end-to-end scores",
         "statement_en": f"DIN conditional nDCG is {statements(pool)}. DIN end-to-end nDCG is {statements(end)}.",
         "evidence": evidence(pool, end) + ["I-denom-pool", "I-denom-end", "I-destination-outside",
                                           "I-destination-missed", "I-destination-recalled"]},
        {"id": CONCLUSIONS[3], "scope": "CARD-008 and CARD-010 internal validation only",
         "statement_en": "Index numbers use the internal static pool. Audit numbers quantify sensitivity to the "
                         "recorded evaluation interventions on internal validation.",
         "evidence": ["I-internal-Flat-recall-mean", "I-internal-HNSWFlat-overlap-mean",
                      "I-internal-IVFPQ-overlap-mean", "I-internal-audit-split-mean",
                      "I-internal-audit-history-mean", "I-internal-audit-random-mean",
                      "I-internal-audit-uniform-mean", "I-internal-audit-weighted-mean"]}]


def verify(indicators, claims):
    entries = {entry["id"]: entry for entry in indicators}
    if len(entries) != len(indicators) or {item["id"] for item in claims} != set(CONCLUSIONS):
        raise ValueError("invalid indicator/conclusion IDs")
    for item in claims:
        if not item["scope"] or not item["evidence"] or any(
            not entries[ref]["sources"] for ref in item["evidence"]
        ):
            raise ValueError(f"{item['id']}: missing evidence")



def verify_registered():
    if status_paths():
        raise ValueError("report audit requires a clean result commit")
    from tools.report import fmt, tex_escape
    snap = snapshot()
    metrics, sections = plan()
    sources = load_sources({field[0] for metric in metrics for field in metric.fields})
    expected = [metric.calculate(sources) for metric in metrics]
    packs = {}
    for kind, filename in (("IV", "indicators.json"), ("CV", "conclusions.json")):
        doc = json.loads((ROOT / "results" / filename).read_bytes())
        row = snap[kind]
        meta = doc["_meta"]
        if (meta["version"] != row["version"] or meta["kind"] != kind or meta["dirty"] or
                payload_sha256(doc["payload"]) != meta["payload_sha256"] or
                meta["payload_sha256"] != row["identity"]):
            raise ValueError(f"{kind} metadata and registry differ")
        packs[kind] = doc
    iv, cv = packs["IV"], packs["CV"]
    if cv["_meta"]["indicator_payload_sha256"] != iv["_meta"]["payload_sha256"] or iv["payload"] != expected:
        raise ValueError("IV/CV sources do not reproduce the report")
    recomputed = conclusions(expected)
    if cv["payload"] != recomputed:
        raise ValueError("conclusions do not reproduce from IV")
    verify(expected, recomputed)
    expected_tex = render_tex(sections)
    if (ROOT / "report/g3-results.tex").read_text(encoding="utf8") != expected_tex:
        raise ValueError("report body does not match the frozen plan")
    used = re.findall(r"\\(iv|cv)\{([^}]*)\}", expected_tex)
    by_id = {item["id"]: fmt(item) for item in expected}
    with (ROOT / "results/number_trace.csv").open(encoding="utf8", newline="") as stream:
        trace = list(csv.DictReader(stream))
    expected_trace = [(name, by_id[name] if kind == "iv" else "（结论原文）") for kind, name in used]
    observed_trace = [(row["来源编号"], row["数字或结论"]) for row in trace]
    if expected_trace != observed_trace:
        raise ValueError("report macros and number trace differ")
    product = (ROOT / "report/generated/macros.tex").read_text(encoding="utf8")
    definitions = dict(re.findall(r"\\csname ((?:iv|cv)@[^\\]+)\\endcsname\{(.*)\}", product))
    wanted = {f"iv@{name}": display for name, display in by_id.items()}
    wanted.update({f"cv@{item['id']}": tex_escape(item["statement_en"]) for item in recomputed})
    if definitions != wanted:
        raise ValueError("generated definitions do not reproduce IV/CV")
    print(f"verified {len(expected)} indicators, {len(recomputed)} conclusions and {len(used)} report citations")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-registered", action="store_true")
    if parser.parse_args().verify_registered:
        verify_registered()
        return
    if status_paths():
        raise ValueError("CV requires the clean registered result commit")
    raw = json.loads((ROOT / STAGING / "indicators.json").read_bytes())
    payload = raw["payload"]
    if raw["_meta"]["commit"] != head() or payload_sha256(payload) != raw["_meta"]["payload_sha256"]:
        raise ValueError("staged IV metadata mismatch")
    metrics, _ = plan()
    sources = load_sources({field[0] for metric in metrics for field in metric.fields})
    if [metric.calculate(sources) for metric in metrics] != payload:
        raise ValueError("staged IV cannot be reproduced from registered RVs")
    claims = conclusions(payload)
    verify(payload, claims)
    write_result(f"{STAGING}/conclusions.json", claims, "CV", "CV-mind-final-v0.1", config=CONFIG,
                 extra_meta={"indicator_payload_sha256": raw["_meta"]["payload_sha256"]})


if __name__ == "__main__":
    main()
