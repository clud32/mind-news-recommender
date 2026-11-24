"""Materialize small-dev indicators from registered RVs and describe their source fields."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean, stdev
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.common import git, head, payload_sha256, snapshot, status_paths
from tools.rv import write_result

CONFIG = "configs/mind-final-v0.1.json"
STAGING = "scratch/card-011"
FINAL_OBJECTS = tuple(f"RV-small-dev-s{i}" for i in range(3))


@dataclass(frozen=True)
class Metric:
    id: str
    fields: tuple
    unit: str
    policy: str
    operation: str = "value"

    def calculate(self, sources):
        values = []
        lineage = []
        for obj, *path in self.fields:
            doc = sources[obj]
            value = doc["payload"]
            for field in path:
                value = value[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"expected numeric field at {obj}, {path}")
            values.append(value)
            lineage.append({"registry_object": obj, "rv": doc["_meta"]["version"],
                            "payload_sha256": doc["_meta"]["payload_sha256"], "field": path})
        calculators = {"value": lambda: values[0], "mean": lambda: mean(values),
                       "sample_sd": lambda: stdev(values), "difference": lambda: values[0] - values[1],
                       "mean_difference": lambda: mean(differences(values)),
                       "sd_difference": lambda: stdev(differences(values))}
        value = calculators[self.operation]()
        item = {"id": self.id, "value": value, "unit": self.unit, "policy": self.policy,
                "operation": self.operation, "sources": lineage}
        if self.id.startswith("I-error-"):
            item["digits"] = 12
        return item


def differences(values):
    return [a - b for a, b in zip(values[::2], values[1::2], strict=True)]


def field(seed, *path):
    return (FINAL_OBJECTS[seed], *path)


def load_sources(objects, registry=None, results_dir=None):
    registry = snapshot() if registry is None else registry
    results_dir = ROOT / "results/rv" if results_dir is None else results_dir
    sources = {}
    for obj in sorted(objects):
        row = registry[obj]
        path = results_dir / f"{row['version']}.json"
        doc = json.loads(path.read_bytes())
        meta = doc["_meta"]
        if (meta["kind"], meta["version"], meta["dirty"]) != ("RV", row["version"], False):
            raise ValueError(f"{obj}: metadata mismatch")
        if payload_sha256(doc["payload"]) != row["identity"] or meta["payload_sha256"] != row["identity"]:
            raise ValueError(f"{obj}: payload identity mismatch")
        sources[obj] = doc
    return sources


def plan():
    """Return indicator definitions and tables. This contract is fixed before reading dev."""
    metrics, sections = [], []
    ratio, count = "fraction", "count"
    ndcg = "nDCG@10"
    baseline = {"Popularity": "popularity", "Recent click": "recent_click"}
    official_policy = "Dev, mixed binary labels only, original impression, equal exposure weights"
    recall_policy = "Dev, strictly prior train and dev exposures, exact top-k, logged click occurrences"
    pool_policy = "Dev, frozen tower seed-zero top-k, conditional on at least one hit"
    end_policy = "Dev, mixed labels, ideal includes all logged positives and zero-hit exposures score zero"
    bootstrap_policy = "Paired user-cluster percentile interval over seed-averaged effects"

    def add(id_, fields, unit, policy, operation="value"):
        metric = Metric(f"I-{id_}", tuple(fields), unit, policy, operation)
        metrics.append(metric)
        return metric.id

    def table(title, columns):
        rows = []
        sections.append((title, columns, rows))
        return rows

    def statistics_row(rows, label, id_, fields, unit, policy, control=None):
        if control is None and len(fields) == 1:
            ids = [add(id_, fields, unit, policy)]
            rows.append((label, *ids, None, None, None, None))
            return
        pairs = tuple(zip(fields, control, strict=True)) if control else tuple((f,) for f in fields)
        comparisons = tuple(source for pair in pairs for source in pair)
        ids = [add(f"{id_}-s{i}", pair, unit, policy, "difference" if control else "value")
               for i, pair in enumerate(pairs)]
        for which, operation in (("mean", "mean_difference" if control else "mean"),
                                 ("sd", "sd_difference" if control else "sample_sd")):
            ids.append(add(f"{id_}-{which}", comparisons, unit, policy, operation))
        rows.append((label, *ids))

    def comparison(rows, label, key):
        ids = [add(f"{key}-{suffix}", [field(0, "bootstrap", "comparisons", key, suffix)],
                   count if suffix in ("users", "denominator") else ratio, bootstrap_policy)
               for suffix in ("estimate", "low", "high", "users", "denominator")]
        rows.append((label, *ids))

    official = table("Official impression ranking", ("Method and metric", "Seed 0", "Seed 1", "Seed 2", "Mean", "Sample SD"))
    retrieval = table("Temporal retrieval", ("Method and metric", "Seed 0", "Seed 1", "Seed 2", "Mean", "Sample SD"))
    rerank = table("Fixed-pool ranking", ("Method and metric", "Seed 0", "Seed 1", "Seed 2", "Mean", "Sample SD"))
    deltas = table("Same-sample paired differences", ("Comparison", "Seed 0", "Seed 1", "Seed 2", "Mean", "Sample SD"))
    summaries = table("Evaluation denominators and recall destinations", ("Quantity", "Value"))
    intervals = table("User-cluster bootstrap", ("Comparison", "Effect", "95 percent low", "95 percent high",
                                                  "Users", "Denominator"))
    internal = table("Internal time-validation references", ("Quantity", "Seed 0", "Seed 1", "Seed 2", "Mean", "Sample SD"))

    for model, name, key in (("popularity", "Popularity", "pop"), ("recent_click", "Recent click", "recent"),
                             ("tower", "Tower", "tower"), ("din", "DIN", "din")):
        seeds = range(3) if model in ("tower", "din") else range(1)
        for slug, measure in (("auc", "AUC"), ("mrr", "MRR"), ("ndcg5", "nDCG@5"), ("ndcg10", ndcg)):
            paths = [field(s, "official", f"{model}_s{s}" if model in ("tower", "din") else model,
                           "metrics", measure) for s in seeds]
            statistics_row(official, f"{name} {measure}", f"official-{key}-{slug}", paths, ratio, official_policy)
            if model in ("tower", "din"):
                for control_label, control in baseline.items():
                    refs = [field(s, "official", control, "metrics", measure) for s in seeds]
                    statistics_row(deltas, f"{name} minus {control_label}, {measure}",
                                   f"official-{key}-minus-{control}-{slug}", paths, ratio, official_policy, refs)
                if model == "din":
                    refs = [field(s, "official", "tower_s0", "metrics", measure) for s in seeds]
                    statistics_row(deltas, f"DIN minus Tower seed 0, {measure}",
                                   f"official-din-minus-tower-{slug}", paths, ratio, official_policy, refs)
        if model == "din":
            continue
        for measure in ("clicks", "covered", "hits", "recall", "coverage", "conditional_recall"):
            paths = [field(s, "retrieval", f"tower_s{s}" if model == "tower" else model, measure) for s in seeds]
            unit = count if measure in ("clicks", "covered", "hits") else ratio
            statistics_row(retrieval, f"{name} {measure.replace('_', ' ')}", f"retrieval-{key}-{measure}",
                           paths, unit, recall_policy)
            if model == "tower" and unit == ratio:
                for control_label, control in baseline.items():
                    refs = [field(s, "retrieval", control, measure) for s in seeds]
                    statistics_row(deltas, f"Tower minus {control_label}, {measure.replace('_', ' ')}",
                                   f"retrieval-tower-minus-{control}-{measure}", paths, ratio, recall_policy, refs)

    for metric, policy in (("pool", pool_policy), ("end", end_policy)):
        key = f"{metric}_{ndcg}"
        refs = [field(s, "rerank", "tower_s0", key) for s in range(3)]
        din = [field(s, "rerank", f"din_s{s}", key) for s in range(3)]
        statistics_row(rerank, f"Tower seed 0 {metric} nDCG@10", f"rerank-tower-{metric}", refs[:1], ratio, policy)
        statistics_row(rerank, f"DIN {metric} nDCG@10", f"rerank-din-{metric}", din, ratio, policy)
        statistics_row(deltas, f"DIN minus Tower seed 0, {metric} nDCG@10", f"rerank-din-minus-tower-{metric}",
                       din, ratio, policy, refs)
        summaries.append((f"{metric.capitalize()} ranking denominator",
                          add(f"denom-{metric}", [field(0, "rerank", "tower_s0", f"{metric}_denominator")], count, policy)))
        comparison(intervals, f"DIN minus Tower seed 0, {metric}", f"{metric}_din_minus_tower_s0")

    shared = [
        ("Dev total", "official-total", ("official", "tower_s0", "total"), official_policy),
        ("Scored exposure count", "denom-official", ("official", "tower_s0", "scoring_denominator"), official_policy),
        ("Microsoft scorer denominator", "denom-microsoft", ("microsoft", "denominator"), official_policy)]
    for label in ("empty_labels", "all_positive", "all_negative"):
        shared.append((label.replace("_", " "), f"excluded-{label}",
                       ("official", "tower_s0", "excluded", label), official_policy))
    for dest in ("outside", "missed", "recalled"):
        shared.append((f"Frozen Top-100 click destination: {dest}", f"destination-{dest}",
                       ("recall_destination", dest), recall_policy))
    for label, key, path, policy in shared:
        summaries.append((label, add(key, [field(0, *path)], count, policy)))
    for metric in ("AUC", "MRR", "nDCG@5", ndcg):
        summaries.append((f"Microsoft absolute error {metric}",
                          add(f"error-{metric.replace('@', '')}", [field(0, "microsoft", "absolute_errors", metric)],
                              ratio, official_policy)))

    for label, control in baseline.items():
        comparison(intervals, f"Tower minus {label}, official nDCG@10", f"official_tower_minus_{control}")
        comparison(intervals, f"DIN minus {label}, official nDCG@10", f"official_din_minus_{control}")
        comparison(intervals, f"Tower minus {label}, temporal recall", f"recall_tower_minus_{control}")
    comparison(intervals, "DIN minus Tower seed 0, official nDCG@10", "official_din_minus_tower_s0")

    for method in ("Flat", "HNSWFlat", "IVFPQ"):
        for label, path, unit in (("overlap", ("overlap_all", "overlap"), ratio),
                                  ("recall", ("recall",), ratio), ("p50", ("p50_us",), "microsecond"),
                                  ("p95", ("p95_us",), "microsecond"), ("size", ("bytes",), "byte")):
            fields = [(f"RV-faiss-s{s}", "methods", method, *path) for s in range(3)]
            statistics_row(internal, f"{method} {label}", f"internal-{method}-{label}", fields,
                           unit, "Internal validation, static candidates and frozen CARD-008 scorer")
    for audit_slug, label, paths in (
        ("split", "Random minus time nDCG@10", ("split", "random_minus_time", "ndcg10")),
        ("history", "Donor history DIN nDCG@10", ("history", "branches", "donor_din", "official", ndcg)),
        ("random", "Random AUC expectation", ("random", "expectation", "AUC")),
        ("uniform", "Uniform-five DIN minus tower", ("sampling", "groups", "uniform5", "din_minus_tower")),
        ("weighted", "Weighted-twenty DIN minus tower", ("sampling", "groups", "weighted20", "din_minus_tower"))):
        fields = [(f"RV-evaluation-audit-s{s}", *paths) for s in range(3)]
        statistics_row(internal, label, f"internal-audit-{audit_slug}", fields, ratio,
                       "Internal validation audit, paired exposures under the originating protocol")
    return metrics, sections


CONCLUSIONS = ("C-small-dev-tower", "C-small-dev-din", "C-small-dev-fixed-pool", "C-small-dev-internal")


def render_tex(sections):
    def esc(s):
        return s.replace("_", r"\_").replace("&", r"\&")
    lines = [r"\documentclass{article}", r"\usepackage[margin=0.65in]{geometry}",
             r"\begin{document}", r"\input{generated/macros.tex}", r"\section*{MIND-small final comparison}",
             r"Official metrics are exposure weighted. Temporal recall uses recorded click occurrences.",
             r"Baseline values and the frozen reranking control appear once, in the first value column.",
             r"Intervals resample paired users and retain exposure or click weights.", r"\small"]
    for i, (name, columns, rows) in enumerate(sections):
        if i:
            lines.append(r"\clearpage")
        lines += [rf"\subsection*{{{esc(name)}}}", r"\begin{tabular}{l" + "r" * (len(columns) - 1) + "}",
                  " & ".join(columns) + r" \\", r"\hline"]
        for j, (label, *ids) in enumerate(rows):
            if j and j % 28 == 0:
                lines.extend([r"\end{tabular}", r"\clearpage",
                              r"\begin{tabular}{l" + "r" * (len(columns) - 1) + "}",
                              " & ".join(columns) + r" \\", r"\hline"])
            values = [rf"\iv{{{id_}}}" if id_ else "--" for id_ in ids]
            lines.append(" & ".join((esc(label), *values)) + r" \\")
        lines += [r"\end{tabular}", ""]
    lines += [r"\clearpage", r"\section*{Conclusions}", r"\begin{itemize}",
              *(rf"\item \cv{{{id_}}}" for id_ in CONCLUSIONS), r"\end{itemize}", r"\end{document}"]
    return "\n".join(lines) + "\n"


def pack():
    if status_paths() or head() == git("rev-parse", "card-011", check=True).stdout.strip():
        raise ValueError("IV requires the clean registered result commit")
    metrics, sections = plan()
    if (ROOT / "report/g3-results.tex").read_text(encoding="utf8") != render_tex(sections):
        raise ValueError("report source differs from frozen indicator plan")
    sources = load_sources({source[0] for metric in metrics for source in metric.fields})
    payload = [metric.calculate(sources) for metric in metrics]
    if len({item["id"] for item in payload}) != len(payload):
        raise ValueError("duplicate indicator ID")
    write_result(f"{STAGING}/indicators.json", payload, "IV", "IV-mind-final-v0.1", config=CONFIG)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-tex", type=Path)
    args = parser.parse_args()
    if args.draft_tex:
        args.draft_tex.parent.mkdir(parents=True, exist_ok=True)
        args.draft_tex.write_text(render_tex(plan()[1]), encoding="utf8", newline="\n")
    else:
        pack()


if __name__ == "__main__":
    main()
