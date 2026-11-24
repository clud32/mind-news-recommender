"""Run the frozen CARD-011 protocol. Business reads require a clean card-011 tag."""
import argparse
from collections import Counter
from datetime import datetime
import gzip
import importlib.util
from io import StringIO
from itertools import groupby
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

from mind_din import DIN, fixed_order, pool_scores
from mind_evaluation import Impression, METRICS, _measure, evaluate, ranks_from_scores
from mind_two_tower import MeanTowers, article_tokens, exact_top
from profile_mind_small import fields, parse_impression, require
from run_mind_baselines import ClickWindow, Retrieval, TitleIndex, split_prefix
from run_mind_din import batch_logits
from run_mind_two_tower import Event, histories, registered_ref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import git, head, payload_sha256, sha256_file, snapshot, status_paths
from tools.rv import write_result

CONFIG = "configs/mind-final-v0.1.json"
OUT = "scratch/card-011"
BASELINES = ("popularity", "recent_click")
SEEDS = (0, 1, 2)


def read_json(path):
    return json.loads((ROOT / path).read_bytes())


def verify_inputs(cfg):
    snap = snapshot()
    require(tuple(cfg["policy"]["seeds"]) == SEEDS and cfg["policy"]["din_encoder_seed"] == 0,
            "configured model seeds differ from frozen scorer")
    data = cfg["data"]
    require(snap["DATA"]["identity"] == data["manifest_sha256"] ==
            sha256_file(ROOT / data["manifest"]) and snap["DATA"]["version"] == data["version"],
            "DATA manifest identity mismatch")
    registered_tsv = dict(
        (path, identity) for identity, path in
        (line.split(None, 1) for line in (ROOT / data["manifest"]).read_text(encoding="utf8").splitlines()
         if line and not line.startswith("#")))
    for path, identity in data["tsv"].items():
        require(registered_tsv[path] == identity and (ROOT / path).is_file(), f"TSV inventory: {path}")
    for path, identity in (cfg["configs"] | cfg["artifacts"]).items():
        require(sha256_file(ROOT / path) == identity, f"frozen input changed: {path}")
    require(sha256_file(ROOT / cfg["scorer"]["path"]) == cfg["scorer"]["sha256"],
            "Microsoft scorer identity mismatch")
    refs = {}
    for obj, identity in cfg["references"].items():
        require(snap[obj]["identity"] == identity, f"immutable reference changed: {obj}")
        refs[obj] = registered_ref(obj, snap)
    for seed in SEEDS:
        files = refs[f"RV-two-tower-s{seed}"]["artifacts"] | refs[f"RV-din-s{seed}"]["weights"]
        require(all(cfg["artifacts"].get(p) == h for p, h in files.items()),
                f"configured and registered weight identities differ: seed {seed}")
    return refs


def events(path):
    with path.open("rb") as stream:
        for location, row in fields(stream, path.as_posix(), 5):
            item, history, _ = parse_impression(row, location)
            labels = (b"" if item.group == "empty_labels" else
                      bytes(token.endswith("-1") for token in row[4].split()))
            yield Event(item.iid, item.time, item.candidates, labels,
                        tuple(map(sys.intern, history))), sys.intern(row[1])


def merge_news(train_path, dev_path, evidence_path, before_dev=None):
    texts, sources, conflicts = {}, {}, []
    counts = Counter()
    for split, path in (("train", train_path), ("dev", dev_path)):
        if split == "dev" and before_dev:
            before_dev()
        with path.open("rb") as stream:
            for location, row in fields(stream, path.as_posix(), 8):
                news, text = row[0], (row[3], row[4])
                if news in texts and texts[news] != text:
                    conflicts.append({"news_id": news, "previous": sources[news], "current": location,
                                      "previous_text": texts[news], "current_text": text})
                if news not in texts:
                    texts[news] = text
                    sources[news] = {"split": split, "location": location}
                    counts[split] += 1
                elif split == "dev" and texts[news] == text:
                    counts["dev_identical_reuse"] += 1
    if conflicts:
        evidence_path.write_text(json.dumps(conflicts, ensure_ascii=False, indent=2), encoding="utf8")
        raise ValueError(f"{len(conflicts)} news text conflicts, see {evidence_path}")
    return texts, dict(counts), sources


def valid_labels(labels):
    return 0 < sum(labels) < len(labels)


def verify_microsoft(cfg, rows):
    spec = importlib.util.spec_from_file_location("mind_reference_scorer", ROOT / cfg["scorer"]["path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    truth, prediction = StringIO(), StringIO()
    i = 0
    for item in rows:
        if valid_labels(item.labels):
            i += 1
            truth.write(f"{i} {json.dumps(list(item.labels), separators=(',', ':'))}\n")
            prediction.write(f"{i} {json.dumps(item.ranks, separators=(',', ':'))}\n")
    require(i > 0, "empty Microsoft reference denominator")
    truth.seek(0)
    prediction.seek(0)
    reference = dict(zip(METRICS, map(float, module.scoring(truth, prediction)), strict=True))
    project = evaluate(rows)
    require(project["scoring_denominator"] == i, "Microsoft denominator mismatch")
    errors = {name: abs(reference[name] - project["metrics"][name]) for name in METRICS}
    require(all(err <= cfg["scorer"]["tolerance"] for err in errors.values()),
            f"Microsoft scorer exceeds tolerance: {errors}")
    return {"metrics": reference, "absolute_errors": errors, "denominator": i}


def timestamp_replay(train, dev, score):
    """Publish both splits together only after every dev query of a timestamp."""
    tagged = [(event, None) for event, _ in train] + list(dev)
    tagged.sort(key=lambda item: item[0].timestamp)
    yield from replay_groups(tagged, score)


def replay_groups(tagged, score):
    for t, group in groupby(tagged, key=lambda item: item[0].timestamp):
        group = list(group)
        scored = score(t, [(event, user) for event, user in group if user is not None])
        yield t, [event for event, _ in group], scored


def frozen_models(news_ids, texts, base, din_cfg, device):
    index = {news: i for i, news in enumerate(news_ids, 1)}
    pattern = re.compile(base["token_pattern"])
    n_title, n_abstract = base["title_tokens"], base["abstract_tokens"]
    tokens = [article_tokens(texts.get(news, ("", "")), pattern, n_title, n_abstract) for news in news_ids]
    towers, banks, dins = [], [], []
    for seed in SEEDS:
        prefix = f"checkpoints/card-004/s{seed}"
        vocab = read_json(f"{prefix}/vocabulary.json")
        require(set(vocab.values()) == set(range(1, len(vocab) + 1)), "invalid frozen vocabulary IDs")
        table = np.zeros((len(index) + 1, n_title + n_abstract), dtype=np.int32)
        for i, words in enumerate(tokens, 1):
            encoded = [vocab[word] for word in words if word in vocab]
            table[i, :len(encoded)] = encoded
        tower = MeanTowers(len(vocab), base["embedding_dim"]).eval()
        tower.load_state_dict(torch.load(ROOT / prefix / "model.pt", weights_only=True, map_location="cpu"))
        with torch.inference_mode():
            banks.append(torch.cat([tower.news(batch) for batch in torch.from_numpy(table).split(512)]))
        towers.append(tower)
        din = DIN(din_cfg["dim"], din_cfg["attention_widths"], din_cfg["head_width"]).to(device).eval()
        din.load_state_dict(torch.load(ROOT / f"checkpoints/card-009/s{seed}.pt",
                                       weights_only=True, map_location=device))
        dins.append(din)
    return index, towers, banks, dins


@torch.inference_mode()
def score_din(model, bank, hist, lists, index, fallback):
    lengths = np.array([len(ids) for ids in lists], dtype=np.int64)
    flat = np.array([index[news] for ids in lists for news in ids], dtype=np.int64)
    logits = batch_logits(model, bank, hist, lengths, flat).cpu().numpy() if len(flat) else np.array([])
    return [scores if hist[i].any() else fallback[i]
            for i, scores in enumerate(np.split(logits, np.cumsum(lengths)[:-1]))]


def rerank(top, scores, empty_history):
    ordered = fixed_order(top, scores, empty_history)
    require(len(ordered) == len(top) and set(ordered) == set(top), "DIN changed fixed candidate membership")
    return ordered


def log_jsonl(stream, record):
    stream.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")


def paired_bootstrap(users, differences, denominators, resamples, seed):
    """Resample user clusters uniformly; each draw retains all of its exposures."""
    identities, inverse = np.unique(users, return_inverse=True)
    require(len(identities) > 0, "paired bootstrap denominator is zero")
    weight = np.bincount(inverse, weights=denominators)
    values = np.bincount(inverse, weights=differences)
    require(weight.sum() > 0, "paired bootstrap weight is zero")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(resamples):
        sampled = rng.integers(len(identities), size=len(identities))
        denominator = weight[sampled].sum()
        if denominator:
            estimates.append(float(values[sampled].sum() / denominator))
    require(len(estimates) > 0, "no nonempty bootstrap replicate")
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"low": float(low), "high": float(high), "estimate": float(values.sum() / weight.sum()),
            "users": len(identities), "denominator": int(weight.sum()), "valid_replicates": len(estimates)}


class Replay:
    def __init__(self, cfg, base, title_index, news_ids, model_bundle, device):
        self.cfg = cfg
        self.base = base
        self.tfidf = title_index
        self.news = np.array(["", *news_ids], dtype=object)
        self.index, self.towers, self.banks, self.dins = model_bundle
        self.din_bank = self.banks[0].to(device)
        self.visible = np.zeros(len(self.news), dtype=bool)
        self.window = ClickWindow(cfg["policy"]["window_hours"])
        self.k = cfg["policy"]["top_k"]
        self.title_rows = np.array([self.tfidf.index.get(n, self.tfidf.empty_row) for n in self.news])

    def publish(self, observed):
        for row in observed:
            for news in row.candidates:
                self.visible[self.index[news]] = True
        self.window.update(observed)

    @torch.inference_mode()
    def score(self, t, group):
        if not group:
            return []
        self.window.expire(t)
        rows, users = zip(*group)
        hist = histories(rows, self.index, self.base["history_limit"])
        ht = torch.from_numpy(hist)
        lists = [row.candidates for row in rows]
        positions = [np.array([self.index[n] for n in ids], dtype=np.int64) for ids in lists]
        counts = self.window.counts
        hot = np.zeros(len(self.news), dtype=np.int64)
        for news, count in counts.items():
            hot[self.index[news]] = count
        pop_top = self.news[exact_top(hot, self.visible, self.k)].tolist()
        official = {name: [] for name in (*BASELINES, *(f"tower_s{s}" for s in SEEDS))}
        tops = {name: [] for name in official}
        fallbacks = []
        for row, pos in zip(rows, positions):
            official["popularity"].append(hot[pos])
            tops["popularity"].append(pop_top)
            similarities, reason = self.tfidf.query(row.history[-1] if row.history else None)
            fallbacks.append(reason)
            if reason:
                official["recent_click"].append(hot[pos])
                tops["recent_click"].append(pop_top)
                continue
            sim = similarities[self.title_rows]
            official["recent_click"].append(sim[pos])
            tops["recent_click"].append(self.news[exact_top(sim, self.visible, self.k)].tolist())

        # Matrix products are tiled by query batch; ID indices resolve ties.
        for seed, tower, bank in zip(SEEDS, self.towers, self.banks):
            name = f"tower_s{seed}"
            for start in range(0, len(rows), self.cfg["policy"]["score_batch"]):
                limit = start + self.cfg["policy"]["score_batch"]
                used = ht[start:limit]
                queries = tower.aggregate_history(bank[used], used)
                pool_scores_batch = (queries @ bank.T).numpy()
                for offset, (query, pos) in enumerate(zip(queries, positions[start:limit])):
                    # Same rank rule as CARD-004 official impression scoring.
                    official[name].append((bank[pos] * query).sum(-1).numpy())
                    top = exact_top(pool_scores_batch[offset], self.visible, self.k)
                    tops[name].append(self.news[top].tolist())

        fixed = tops["tower_s0"]
        orders = {"tower_s0": fixed}
        for seed, model in zip(SEEDS, self.dins):
            name = f"din_s{seed}"
            original_scores, ranks = [], []
            for start in range(0, len(rows), self.cfg["policy"]["score_batch"]):
                end = start + self.cfg["policy"]["score_batch"]
                original_scores.extend(score_din(model, self.din_bank, hist[start:end], lists[start:end],
                                                 self.index, official["tower_s0"][start:end]))
                fixed_batch = fixed[start:end]
                scores = score_din(model, self.din_bank, hist[start:end], fixed_batch, self.index,
                                   [np.zeros(len(ids)) for ids in fixed_batch])
                ranks.extend(rerank(ids, value, not row.history)
                             for ids, value, row in zip(fixed_batch, scores, rows[start:end]))
            official[name], orders[name] = original_scores, ranks

        records = []
        for i, (row, user) in enumerate(zip(rows, users)):
            clicks = tuple(row.clicks())
            covered = sum(bool(self.visible[self.index[news]]) for news in clicks)
            selected = {name: set(ids[i]) for name, ids in tops.items()}
            hits = {name: sum(n in ids for n in clicks) for name, ids in selected.items()}
            require(all(self.visible[self.index[n]] for ids in selected.values() for n in ids),
                    "retrieval returned a same-time or future candidate")
            recalled = hits["tower_s0"]
            require(all(0 <= hit <= covered <= len(clicks) for hit in hits.values()), "recall counts")
            records.append({
                "id": row.impression_id, "user": user, "time": t.isoformat(),
                "labels": list(row.labels), "candidates": row.candidates,
                "ranks": {name: ranks_from_scores(row.candidates, scores[i]) for name, scores in official.items()},
                "top100": {name: ids[i] for name, ids in tops.items()},
                "fixed": fixed[i], "orders": {name: order[i] for name, order in orders.items()},
                "clicks": len(clicks), "covered": covered, "hits": hits,
                "recall_destination": [len(clicks) - covered, covered - recalled, recalled],
                "rerank": {name: pool_scores(order[i], clicks) for name, order in orders.items()},
                "recent_fallback": fallbacks[i]})
        return records


def aggregate(records, cfg):
    labels = [r["labels"] for r in records]
    eligible = [i for i, row in enumerate(labels) if valid_labels(row)]
    require(bool(eligible) and sum(r["clicks"] for r in records) > 0, "empty evaluation denominator")
    official_rows = {
        name: [Impression(y, r["ranks"][name]) for r, y in zip(records, labels)]
        for name in records[0]["ranks"]}
    official = {name: evaluate(rows) for name, rows in official_rows.items()}
    total, covered = (sum(r[key] for r in records) for key in ("clicks", "covered"))
    require(covered > 0, "zero covered click denominator")
    retrieval = {name: Retrieval(total, covered, sum(r["hits"][name] for r in records)).report()
                 for name in records[0]["hits"]}
    pool_index = [i for i, r in enumerate(records) if r["hits"]["tower_s0"] > 0]
    require(bool(pool_index), "zero fixed-pool denominator")
    ranking = {}
    for name in records[0]["rerank"]:
        ranking[name] = {
            "pool_nDCG@10": float(np.mean([records[i]["rerank"][name][0] for i in pool_index])),
            "end_nDCG@10": float(np.mean([records[i]["rerank"][name][1] for i in eligible])),
            "pool_denominator": len(pool_index), "end_denominator": len(eligible)}
    destinations = {name: sum(r["recall_destination"][i] for r in records)
                    for i, name in enumerate(("outside", "missed", "recalled"))}
    require(sum(destinations.values()) == total and destinations["recalled"] == retrieval["tower_s0"]["hits"],
            "unbalanced recall destinations")
    boot = cfg["bootstrap"]
    report_intervals = {}

    def interval(name, ids, differences, denominators):
        report_intervals[name] = paired_bootstrap(
            [records[i]["user"] for i in ids], differences, denominators, boot["replicates"], boot["seed"])

    unit_weights = np.ones(len(eligible))
    n10 = {name: np.array([_measure(labels[i], rows[i].ranks)["nDCG@10"] for i in eligible])
           for name, rows in official_rows.items()}
    tower_mean = np.mean([n10[f"tower_s{s}"] for s in SEEDS], axis=0)
    din_mean = np.mean([n10[f"din_s{s}"] for s in SEEDS], axis=0)
    click_index = [i for i, r in enumerate(records) if r["clicks"] > 0]
    for name in BASELINES:
        interval(f"official_tower_minus_{name}", eligible, tower_mean - n10[name], unit_weights)
        interval(f"official_din_minus_{name}", eligible, din_mean - n10[name], unit_weights)
        interval(f"recall_tower_minus_{name}", click_index,
                 [np.mean([records[i]["hits"][f"tower_s{s}"] for s in SEEDS]) - records[i]["hits"][name]
                  for i in click_index], [records[i]["clicks"] for i in click_index])
    interval("official_din_minus_tower_s0", eligible, din_mean - n10["tower_s0"], unit_weights)
    for metric, ids, position in (("pool", pool_index, 0), ("end", eligible, 1)):
        differences = [np.mean([records[i]["rerank"][f"din_s{s}"][position] for s in SEEDS])
                       - records[i]["rerank"]["tower_s0"][position] for i in ids]
        interval(f"{metric}_din_minus_tower_s0", ids, differences, np.ones(len(ids)))
    return official_rows, official, retrieval, ranking, destinations, report_intervals


def locked_commit():
    tag = git("rev-parse", "--verify", "refs/tags/card-011^{commit}").stdout.strip()
    require(bool(tag) and head() == tag and not status_paths(),
            "dev business read requires a clean card-011 commit")
    return tag


def final_run(cfg, refs):
    commit = locked_commit()
    folder = ROOT / OUT
    folder.mkdir(parents=True, exist_ok=True)
    require(not (folder / "first-dev-read.json").exists(), "first dev read already recorded; preserve evidence")
    base = read_json(cfg["base_config"])
    din_cfg = read_json(cfg["din_config"])
    baseline = read_json(cfg["baseline_config"])
    data = cfg["data"]
    for path, identity in data["tsv"].items():
        require(sha256_file(ROOT / path) == identity, f"TSV digest mismatch: {path}")

    def begin_dev_read():
        with (folder / "first-dev-read.json").open("x", encoding="utf8") as stream:
            json.dump({"commit": commit, "time": datetime.now().astimezone().isoformat(),
                       "first_file": data["dev_news"], "first_operation": "open for TSV parsing"}, stream)

    texts, news_counts, sources = merge_news(ROOT / data["train_news"], ROOT / data["dev_news"],
                                             folder / "news-conflicts.json", begin_dev_read)
    train = sorted(events(ROOT / data["train_behaviors"]), key=lambda item: item[0].timestamp)
    dev = list(events(ROOT / data["dev_behaviors"]))
    require(len({row.impression_id for row, _ in dev}) == len(dev), "duplicate dev impression IDs")
    used, _ = split_prefix([row for row, _ in train], refs["RV-data-profile"]["split"], baseline)
    fit_ids = {n for row in used for n in row.candidates}
    # Fit on the original training documents, including empty documents for missing IDs.
    train_titles = {n: text[0] for n, text in texts.items() if sources[n]["split"] == "train"}
    all_titles = {n: text[0] for n, text in texts.items()}
    title_index = TitleIndex(train_titles, fit_ids, baseline["tfidf"])
    title_index.titles = all_titles
    title_index.index = {n: i for i, n in enumerate(sorted(all_titles))}
    title_index.matrix = title_index.encoder.transform(all_titles[n] for n in title_index.index)
    title_index.empty_row = len(title_index.index)
    (folder / "news-sources.json").write_text(json.dumps(sources, sort_keys=True, ensure_ascii=False),
                                             encoding="utf8")
    all_ids = sorted(set(texts) | {n for row, _ in train + dev for n in (*row.candidates, *row.history)})
    torch.set_num_threads(cfg["policy"]["cpu_threads"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    replay = Replay(cfg, base, title_index, all_ids, frozen_models(all_ids, texts, base, din_cfg, device), device)
    records = []
    with gzip.open(folder / "dev-evidence.jsonl.gz", "xt", encoding="utf8", newline="\n") as stream:
        for _, observed, scored in timestamp_replay(train, dev, replay.score):
            for record in scored:
                log_jsonl(stream, record)
                records.append(record)
            replay.publish(observed)
    require(len(records) == len(dev), "lost dev impressions in replay")
    official_rows, official, retrieval, ranking, destinations, intervals = aggregate(records, cfg)
    microsoft = verify_microsoft(cfg, official_rows["tower_s0"])
    denominator = official["tower_s0"]["scoring_denominator"]
    require(all(item["scoring_denominator"] == denominator and
                item["excluded"] == official["tower_s0"]["excluded"] for item in official.values()),
            "different official denominators")
    artifacts = {f"{OUT}/{name}": sha256_file(folder / name) for name in
                 ("news-sources.json", "dev-evidence.jsonl.gz", "first-dev-read.json")}
    common = {"data": data, "references": cfg["references"], "inputs": cfg["artifacts"],
              "evidence": artifacts, "news_sources": news_counts, "train_rows_replayed": len(train),
              "dev_rows": len(dev), "recall_destination": destinations, "microsoft": microsoft,
              "bootstrap": {"policy": cfg["bootstrap"], "comparisons": intervals},
              "recent_fallbacks": dict(Counter(r["recent_fallback"] for r in records if r["recent_fallback"])),
              "evaluation_device": str(device), "tfidf_fit_news": len(fit_ids)}
    for seed in SEEDS:
        tower, din = f"tower_s{seed}", f"din_s{seed}"
        result = common | {"seed": seed,
                           "official": {n: official[n] for n in (*BASELINES, tower, "tower_s0", din)},
                           "retrieval": {n: retrieval[n] for n in (*BASELINES, tower, "tower_s0")},
                           "rerank": {n: ranking[n] for n in ("tower_s0", din)}}
        version = f"RV-small-dev-001-s{seed}"
        write_result(f"{OUT}/{version}.json", result, "RV", version, config=CONFIG, seed=seed)



def verify_staged(cfg):
    """Reconcile archived ranks, denominators, visibility and registered input identities."""
    tag = locked_commit()
    folder = ROOT / OUT
    first = read_json(f"{OUT}/first-dev-read.json")
    require(first["commit"] == tag and first["first_file"] == cfg["data"]["dev_news"],
            "dev read marker mismatches code tag or input")
    with gzip.open(folder / "dev-evidence.jsonl.gz", "rt", encoding="utf8") as stream:
        records = [json.loads(line) for line in stream]
    data = cfg["data"]
    train = list(events(ROOT / data["train_behaviors"]))
    dev = list(events(ROOT / data["dev_behaviors"]))
    first_seen = {}
    for row, _ in train + dev:
        for news in row.candidates:
            first_seen[news] = min(first_seen.get(news, row.timestamp), row.timestamp)
    dev.sort(key=lambda item: item[0].timestamp)
    require(len(dev) == len(records), "evidence is missing dev rows")
    for (row, user), record in zip(dev, records, strict=True):
        require((row.impression_id, user, row.timestamp.isoformat(), list(row.candidates), list(row.labels)) ==
                (record["id"], record["user"], record["time"], record["candidates"], record["labels"]),
                f"dev evidence identity mismatch: {row.impression_id}")
        clicks = tuple(row.clicks())
        def prev(news):
            return first_seen[news] < row.timestamp
        coverage = sum(map(prev, clicks))
        require((len(clicks), coverage) == (record["clicks"], record["covered"]), "incorrect click denominator")
        for name, top in record["top100"].items():
            ids = set(top)
            require(len(ids) == len(top) <= cfg["policy"]["top_k"] and all(map(prev, top))
                    and sum(n in ids for n in clicks) == record["hits"][name],
                    f"invalid retrieval evidence: {row.impression_id} {name}")
        fixed = record["fixed"]
        require(fixed == record["top100"]["tower_s0"] == record["orders"]["tower_s0"],
                "fixed reference order mismatch")
        hits = record["hits"]["tower_s0"]
        require(record["recall_destination"] == [len(clicks) - coverage, coverage - hits, hits],
                "recall destination mismatch")
        for name, order in record["orders"].items():
            require(len(order) == len(fixed) and set(order) == set(fixed), "changed fixed candidate set")
            require(list(pool_scores(order, clicks)) == record["rerank"][name], "ranking score mismatch")
    official_rows, official, retrieval, ranking, destinations, intervals = aggregate(records, cfg)
    reference = verify_microsoft(cfg, official_rows["tower_s0"])
    for seed in SEEDS:
        version = f"RV-small-dev-001-s{seed}"
        doc = read_json(f"{OUT}/{version}.json")
        meta, result = doc["_meta"], doc["payload"]
        require(meta["kind"] == "RV" and meta["version"] == version and meta["seed"] == result["seed"] == seed
                and meta["commit"] == tag and not meta["dirty"] and
                meta["config_sha256"] == sha256_file(ROOT / CONFIG) and
                meta["data_version"] == cfg["data"]["version"] and meta["lock"] == snapshot()["LOCK"]["version"] and
                meta["payload_sha256"] == payload_sha256(result), f"{version}: staged metadata mismatch")
        require(result["data"] == cfg["data"] and result["references"] == cfg["references"]
                and result["inputs"] == cfg["artifacts"], f"{version}: frozen inputs mismatch")
        for path, digest in result["evidence"].items():
            require(sha256_file(ROOT / path) == digest, f"changed evidence: {path}")
        tower, din = f"tower_s{seed}", f"din_s{seed}"
        require(result["official"] == {name: official[name] for name in (*BASELINES, tower, "tower_s0", din)}
                and result["retrieval"] == {name: retrieval[name] for name in (*BASELINES, tower, "tower_s0")}
                and result["rerank"] == {name: ranking[name] for name in ("tower_s0", din)}
                and result["recall_destination"] == destinations and result["microsoft"] == reference
                and result["bootstrap"]["comparisons"] == intervals, f"{version}: evidence does not reproduce payload")
        print(f"{version}: evidence, candidate sets, denominators and payload verified")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="verify freeze without opening dev TSV")
    parser.add_argument("--verify-staged", action="store_true", help="audit the staged dev output after A2")
    args = parser.parse_args()
    cfg = read_json(CONFIG)
    refs = verify_inputs(cfg)
    if args.preflight:
        print(f"verified {len(refs)} registered references and {len(cfg['artifacts'])} model artifacts")
    elif args.verify_staged:
        verify_staged(cfg)
    else:
        final_run(cfg, refs)


if __name__ == "__main__":
    main()
