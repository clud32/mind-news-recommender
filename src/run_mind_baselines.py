"""Run the two frozen baselines on the first complete validation time groups."""
import argparse
from bisect import bisect_left
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
from itertools import groupby
import json
from operator import attrgetter
from pathlib import Path
import random
import sys
from time import perf_counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from mind_evaluation import Impression, evaluate, ranks_from_scores
from profile_mind_small import TIME_FORMAT, fields, parse_impression, require
from temporal_candidates import Exposure, score_by_timestamp

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/mind-baselines-smoke-v0.1.json"
MODELS = ("popularity", "recent_click")
FALLBACKS = ("empty_history", "missing_title", "empty_title", "zero_vector")


@dataclass(frozen=True)
class Event(Exposure):
    labels: bytes
    last_news: str | None

    def clicks(self):
        return (news for news, label in zip(self.candidates, self.labels) if label)


def read_events(stream):
    rows = []
    for where, row in fields(stream, "behaviors.tsv", 5):
        item, history, _ = parse_impression(row, where)
        # The shared parser has checked every token and disallows partial labels.
        labels = b"" if item.group == "empty_labels" else bytes(
            token.endswith("-1") for token in row[4].split())
        rows.append(Event(item.iid, item.time, item.candidates, labels,
                          sys.intern(history[-1]) if history else None))
    return sorted(rows, key=attrgetter("timestamp"))


def split_prefix(rows, split, cfg):
    key = attrgetter("timestamp")
    index = 4 * len(rows) // 5
    boundary = rows[index].timestamp
    cut = bisect_left(rows, boundary, key=key)
    require(index == split["zero_based_index"] and
            boundary == datetime.strptime(split["boundary_source_time"], TIME_FORMAT)
            == datetime.fromisoformat(cfg["boundary"]), "split boundary mismatch")
    require(cut == cfg["train_count"] == split["train"]["impressions"] and
            len(rows) - cut == cfg["validation_count"] == split["validation"]["impressions"],
            "split counts mismatch")
    sample = []
    for _, group in groupby(rows[cut:], key):
        sample.extend(group)
        if len(sample) >= cfg["sample_minimum"]:
            break
    require(len(sample) >= cfg["sample_minimum"], "validation prefix too short")
    return rows[:cut], sample


class ClickWindow:
    def __init__(self, hours):
        self.span = timedelta(hours=hours)
        self.queue = deque()
        self.counts = Counter()

    def update(self, rows):
        for row in rows:
            for news in row.clicks():
                self.queue.append((row.timestamp, news))
                self.counts[news] += 1

    def expire(self, t):
        cutoff = t - self.span
        while self.queue and self.queue[0][0] < cutoff:
            _, news = self.queue.popleft()
            self.counts[news] -= 1
            if not self.counts[news]:
                del self.counts[news]


class TitleIndex:
    def __init__(self, titles, training_ids, params):
        self.titles = titles
        self.encoder = TfidfVectorizer(**params)
        self.encoder.fit([titles.get(news, "") for news in sorted(training_ids)])
        self.index = {news: i for i, news in enumerate(sorted(titles))}
        self.matrix = self.encoder.transform(titles[news] for news in self.index)
        self.empty_row = len(self.index)

    def query(self, last):
        if last is None:
            return None, "empty_history"
        if last not in self.index:
            return None, "missing_title"
        if not self.titles[last].strip():
            return None, "empty_title"
        row = self.matrix[self.index[last]]
        if row.nnz == 0:
            return None, "zero_vector"
        # Rows have L2 norm one. A missing candidate title gets similarity zero.
        scores = (self.matrix @ row.T).toarray().ravel()
        return np.append(scores, 0.0), None

    def scores(self, similarities, ids):
        return [similarities[self.index.get(news, self.empty_row)] for news in ids]


@dataclass
class Retrieval:
    clicks: int = 0
    covered: int = 0
    hits: int = 0

    def add(self, clicks, pool, top):
        for news in clicks:
            self.clicks += 1
            self.covered += news in pool
            self.hits += news in top

    def report(self):
        require(0 <= self.hits <= self.covered <= self.clicks, "recall counts")
        return {"clicks": self.clicks, "covered": self.covered, "hits": self.hits,
                "recall": self.hits / self.clicks if self.clicks else None,
                "coverage": self.covered / self.clicks if self.clicks else None,
                "conditional_recall": self.hits / self.covered if self.covered else None}


def run_sample(train, sample, titles, cfg):
    initial_pool = {news for event in train for news in event.candidates}
    index = TitleIndex(titles, initial_pool, cfg["tfidf"])
    window = ClickWindow(cfg["window_hours"])
    cutoff = sample[0].timestamp - window.span
    window.update(row for row in train if row.timestamp >= cutoff)
    official = {name: [] for name in MODELS}
    retrieval = {name: Retrieval() for name in MODELS}
    fallbacks = Counter()
    k = cfg["top_k"]

    def score(row, pool):
        window.expire(row.timestamp)
        # The pool's lexicographic order resolves score ties for both retrievers.
        ordered_pool = sorted(pool)
        popular = window.counts
        pop_top = set(sorted(ordered_pool, key=popular.__getitem__, reverse=True)[:k])
        clicks = tuple(row.clicks())
        pop_scores = [popular[n] for n in row.candidates]
        similarities, reason = index.query(row.last_news)
        if reason:
            fallbacks[reason] += 1
            recent_top, recent_scores = pop_top, pop_scores
        else:
            recent_scores = index.scores(similarities, row.candidates)
            order = np.argsort(-np.asarray(index.scores(similarities, ordered_pool)), kind="stable")
            recent_top = {ordered_pool[i] for i in order[:k]}
        for name, scores, top in zip(MODELS, (pop_scores, recent_scores), (pop_top, recent_top)):
            ranks = ranks_from_scores(row.candidates, scores)
            official[name].append(Impression(row.labels, ranks))
            retrieval[name].add(clicks, pool, top)

    score_by_timestamp(sample, score, initial_pool, after_group=window.update)
    models = {name: {"official": evaluate(official[name]), "retrieval": retrieval[name].report()}
              for name in MODELS}
    pop, recent = (models[name] for name in MODELS)
    for key in ("total", "scoring_denominator", "excluded"):
        require(pop["official"][key] == recent["official"][key], "different official samples")
    for key in ("clicks", "covered"):
        require(pop["retrieval"][key] == recent["retrieval"][key], "different recall denominators")
    require(pop["official"]["scoring_denominator"] > 0 and pop["retrieval"]["clicks"] > 0,
            "empty official or click denominator")
    return {"models": models, "fallback_counts": {key: fallbacks[key] for key in FALLBACKS},
            "training_candidate_ids": len(initial_pool), "tfidf_features": len(index.encoder.vocabulary_)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, choices=("a", "b"))
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from tools.common import git, head, payload_sha256, snapshot, status_paths
    from tools.rv import write_result

    started = perf_counter()
    version = f"RV-g2-baselines-001-{args.run}"
    out = f"scratch/card-003/{version}.json"
    require(not (ROOT / out).exists(), "output already exists")
    tag = git("rev-parse", "--verify", "refs/tags/card-003^{commit}", check=True).stdout.strip()
    require(not status_paths() and head() == tag, "requires clean card-003")
    cfg = json.loads((ROOT / CONFIG).read_bytes())
    registry = snapshot()
    require(registry["DATA"]["version"] == cfg["data_version"] and
            registry["LOCK"]["version"] == cfg["lock"], "unexpected environment or data")
    ref = json.loads((ROOT / cfg["split_rv"]).read_bytes())
    identity = payload_sha256(ref["payload"])
    require(identity == ref["_meta"]["payload_sha256"] == registry["RV-data-profile"]["identity"]
            and Path(cfg["split_rv"]).stem == registry["RV-data-profile"]["version"],
            "split RV identity mismatch")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    with (ROOT / cfg["behaviors"]).open("rb") as stream:
        train, sample = split_prefix(read_events(stream), ref["payload"]["split"], cfg)
    with (ROOT / cfg["news"]).open("rb") as stream:
        titles = {row[0]: row[3] for _, row in fields(stream, "news.tsv", 8)}
    payload = run_sample(train, sample, titles, cfg)
    # Each source decimal ID is encoded as ASCII followed by LF (including the last).
    seq = "".join(row.impression_id + "\n" for row in sample).encode("ascii")
    payload.update({"split": {"boundary": cfg["boundary"], "train": len(train),
                              "validation": cfg["validation_count"], "profile_payload_sha256": identity},
                    "sample": {"count": len(sample), "start": str(sample[0].timestamp),
                               "end": str(sample[-1].timestamp),
                               "id_sequence_sha256": hashlib.sha256(seq).hexdigest()}})
    write_result(out, payload, kind="RV", version=version, config=CONFIG, seed=cfg["seed"],
                 extra_meta={"run": args.run, "output_file": Path(out).name,
                             "compute_seconds": perf_counter() - started})


if __name__ == "__main__":
    main()
