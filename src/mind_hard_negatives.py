"""Mine and audit strictly prior, exposed training negatives by title TF cosine."""
from collections import Counter
import hashlib
from itertools import groupby, zip_longest
import json
from math import isclose, sqrt
from pathlib import Path
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import git, head, sha256_file, snapshot, status_paths
from profile_mind_small import require

HARD_CONFIG = "configs/mind-two-tower-hard-v0.1.json"


def title_counts(texts, cfg):
    tokenize = re.compile(cfg["token_pattern"]).findall
    return {n: Counter(tokenize(title.lower())[:cfg["title_tokens"]]) for n, (title, _) in texts.items()}


def select(row, visible, counts, norms, k):
    if not row.history:
        return [], "empty_history"
    query = counts.get(row.history[-1])
    if not query:
        return [], "missing_or_empty_query_title"
    eligible = {n for n, label in zip(row.candidates, row.labels) if label == 0 and n in visible}
    eligible.difference_update(row.history)
    eligible.difference_update(row.clicks())
    if not eligible:
        return [], "no_eligible_candidate"
    matches = []
    have_titles = False
    for n in eligible:
        terms = counts.get(n)
        if terms:
            have_titles = True
            dot = sum(f * terms[w] for w, f in query.items())
            if dot:
                # Equal squared cosine has exactly the same rank; small integer
                # numerators/denominators are determined by at most 32 tokens.
                rank_score = dot * dot / norms[n]
                cosine = dot / sqrt(norms[row.history[-1]] * norms[n])
                matches.append((rank_score, n, cosine))
    matches.sort(key=lambda x: (-x[0], x[1]))
    chosen = [(n, score) for _, n, score in matches[:k]]
    if chosen:
        return chosen, "selected"
    return [], "no_positive_similarity" if have_titles else "no_candidate_title"


def mine(rows, counts, k):
    visible = set()
    norms = {n: sum(f*f for f in terms.values()) for n, terms in counts.items()}
    for _, group in groupby(rows, key=lambda row: row.timestamp):
        additions = set()
        for row in group:
            chosen, reason = select(row, visible, counts, norms, k)
            yield {"id": row.impression_id, "reason": reason, "items": chosen}
            additions.update(row.candidates)
        visible.update(additions)


def verify(rows, counts, path, k):
    """Independently reconstruct eligibility and complete rankings using first-seen times."""
    first = {}
    for row in rows:
        for n in row.candidates:
            first.setdefault(n, row.timestamp)
    squared = {n: sum(x*x for x in v.values()) for n,v in counts.items()}
    histogram, frequencies, reasons, similarities = Counter(), Counter(), Counter(), []
    with path.open(encoding="utf8") as stream:
        for row, line in zip_longest(rows, stream):
            require(row is not None and line is not None, "mined row count mismatch")
            saved = json.loads(line)
            require(saved["id"] == row.impression_id, f"exposure order mismatch {row.impression_id}")
            banned = set(row.history) | set(row.clicks())
            query = counts.get(row.history[-1], {}) if row.history else {}
            scores = {}
            for n,label in zip(row.candidates,row.labels):
                if label or n in banned or first[n] >= row.timestamp or not query or n in scores:
                    continue
                vector = counts.get(n, {})
                dot = sum(v * query.get(w, 0) for w,v in vector.items())
                if dot:
                    scores[n] = (dot*dot / squared[n], dot / sqrt(squared[n]*squared[row.history[-1]]))
            expected = sorted(scores, key=lambda n: (-scores[n][0], n))[:k]
            chosen = saved["items"]
            require([n for n,_ in chosen] == expected and all(isclose(s,scores[n][1],abs_tol=1e-14)
                    for n,s in chosen), f"{row.impression_id}: invalid visibility, eligibility, score or order")
            histogram[len(chosen)] += 1
            reasons[saved["reason"]] += 1
            frequencies.update(n for n,_ in chosen)
            similarities.extend(s for _,s in chosen)
    bins = np.histogram(similarities, bins=np.linspace(0,1,11))[0].tolist()
    return {"impressions":len(rows), "nonempty":len(rows)-histogram[0], "pairs":len(similarities),
            "violations":0, "count_per_exposure":{str(i):histogram[i] for i in range(k+1)},
            "reason_counts":dict(reasons), "news_frequency":dict(sorted(frequencies.items())),
            "similarity_decile_bins":bins, "similarity_min_median_max":(
                np.quantile(similarities,[0,.5,1]).tolist() if similarities else [])}


def load_verified(rows, index, settings):
    path = ROOT / settings["mined_file"]
    audit = json.loads((ROOT / settings["audit_file"]).read_bytes())
    require(sha256_file(path) == audit["list_sha256"] and audit["commit"] == head(), "mining identity mismatch")
    for key,want in settings["expected"].items():
        require(audit[key] == want, f"unexpected mined {key}")
    hard = np.zeros((len(rows), settings["max_negatives"]),dtype=np.int64)
    with path.open(encoding="utf8") as f:
        for i,(row,line) in enumerate(zip(rows,f,strict=True)):
            entry = json.loads(line)
            require(entry["id"] == row.impression_id, "mining exposure mismatch")
            ids = [index[n] for n,_ in entry["items"]]
            hard[i,:len(ids)] = ids
    return hard


def main():
    from run_mind_two_tower import CONFIG, read_sources, registered_ref, split_prefix
    require(not status_paths() and head() == git("rev-parse","card-006",check=True).stdout.strip(),"requires clean card-006")
    cfg = json.loads((ROOT / CONFIG).read_bytes())
    settings = json.loads((ROOT / HARD_CONFIG).read_bytes())
    split = registered_ref(cfg["split_rv"], snapshot())["split"]
    rows, texts = read_sources(cfg)
    train, _ = split_prefix(rows, split, cfg)
    counts = title_counts(texts, cfg)
    path = ROOT / settings["mined_file"]
    audit_path = ROOT / settings["audit_file"]
    require(not path.exists() and not audit_path.exists(), "previous mining artifacts exist")
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open("x",encoding="utf8",newline="\n") as f:
        for line in mine(train, counts, settings["max_negatives"]):
            f.write(json.dumps(line,ensure_ascii=False,separators=(",",":"))+"\n")
    stats = verify(train, counts, path, settings["max_negatives"])
    stats.update(commit=head(), list_sha256=sha256_file(path))
    with audit_path.open("x",encoding="utf8",newline="\n") as f:
        json.dump(stats,f,ensure_ascii=False,sort_keys=True)
        f.write("\n")
    print({k:v for k,v in stats.items() if k!="news_frequency"},flush=True)
    require(all(stats[key]==n for key,n in settings["expected"].items()), "mined cardinalities differ")


if __name__ == "__main__":
    main()
