"""Train and evaluate one CARD-004 seed from source, with an exact temporal replay."""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
from itertools import groupby
import json
import math
import os
from pathlib import Path
import sys
from threading import Timer
from time import perf_counter

import numpy as np
import torch

from mind_evaluation import Impression, evaluate, ranks_from_scores
from mind_hard_negatives import HARD_CONFIG, load_verified
from mind_two_tower import (AttentionTowers, MeanTowers, Visibility, build_catalog, click_examples,
                            column_mask, exact_top, hard_loss, inbatch_loss, target_probabilities)
from profile_mind_small import fields, parse_impression, require
from run_mind_baselines import Retrieval, split_prefix
from temporal_candidates import Exposure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import git, head, payload_sha256, sha256_file, snapshot, status_paths
from tools.rv import write_result

ATTENTION_CONFIG = "configs/mind-two-tower-attention-v0.1.json"
CONFIG = "configs/mind-two-tower-v0.1.json"
LOGQ_CONFIG = "configs/mind-two-tower-logq-v0.1.json"


@dataclass(frozen=True)
class Event(Exposure):
    labels: bytes
    history: tuple

    def clicks(self):
        return (n for n, label in zip(self.candidates, self.labels) if label)


def read_sources(cfg):
    with (ROOT / cfg["news"]).open("rb") as ns:
        texts = {row[0]: (row[3], row[4]) for _, row in fields(ns, "news.tsv", 8)}
    rows = []
    with (ROOT / cfg["behaviors"]).open("rb") as bs:
        for where, row in fields(bs, "behaviors.tsv", 5):
            obs, hist, _ = parse_impression(row, where)
            labels = b"" if obs.group == "empty_labels" else bytes(t.endswith("-1") for t in row[4].split())
            rows.append(Event(obs.iid, obs.time, obs.candidates, labels, tuple(map(sys.intern, hist))))
    rows.sort(key=lambda r: r.timestamp)
    return rows, texts


def registered_ref(name, snap):
    record = snap[name]
    ref = json.loads((ROOT / "results/rv" / (record["version"] + ".json")).read_bytes())
    require(payload_sha256(ref["payload"]) == ref["_meta"]["payload_sha256"] == record["identity"],
            f"{name}: payload identity mismatch")
    return ref["payload"]


def histories(events, index, limit):
    array = np.zeros((len(events), limit), dtype=np.int64)
    for i, row in enumerate(events):
        ids = [index[n] for n in row.history[-limit:]]
        array[i, :len(ids)] = ids
    return array


def train_epochs(model, table, rows, index, cfg, device, seed, *, pairing=None, hard_rows=None, coefficient=1.0):
    examples, positives = click_examples(rows, index)
    require(len(examples) > 0, "training has no positive samples")
    history = histories(rows, index, cfg["history_limit"])
    q_summary = None
    if pairing is not None:
        reference, control = pairing
        counts_by_id, q = target_probabilities(examples, len(index) + 1)
        by_name = {name: int(counts_by_id[j]) for name, j in index.items() if counts_by_id[j]}
        q_summary = {"total": len(examples), "support": len(by_name), "min": min(by_name.values()),
                     "max": max(by_name.values()), "counts_sha256": payload_sha256(by_name)}
        require(q_summary == reference["q"], "training Q differs from independent count")
        q_summary["sum"] = math.fsum(q)
        require(math.isclose(q_summary["sum"], 1., abs_tol=1e-14, rel_tol=0), "Q sum differs from one")
        q_table = torch.from_numpy(q).to(device)
        # Encode occurrence index, source impression ID and source target ID together.
        iids = np.asarray([int(r.impression_id) for r in rows], dtype="<u8")
        nids = np.zeros(len(index)+1, dtype="<u8")
        for name, j in index.items():
            nids[j] = int(name[1:])
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    results, durations = [], []
    for epoch in range(cfg["epochs"]):
        start = perf_counter()
        order = rng.permutation(len(examples))
        if pairing is not None:
            sequence = hashlib.sha256()
        counts = Counter(valid_rows=0, skipped_rows=0, duplicate_target_columns=0, other_positive_columns=0)
        total_loss = 0.
        hard_counts = Counter(uses=0, rows=0, overlapping_items=0, overlapping_columns=0)
        hard_sum_epoch = 0.
        for offset in range(0, len(order), cfg["batch_size"]):
            batch = examples[order[offset:offset + cfg["batch_size"]]]
            if pairing is not None:
                ordinals = order[offset:offset + cfg["batch_size"]]
                sequence.update(len(batch).to_bytes(4, "little"))
                sequence.update(np.column_stack((ordinals.astype("<u8"), iids[batch[:,0]],
                                                 nids[batch[:,1]])).astype("<u8").tobytes())
            mask, audit = column_mask(batch, positives)
            counts.update(audit)
            valid = len(batch) - audit["skipped_rows"]
            if not valid:
                continue
            counts["valid_rows"] += valid
            h = torch.as_tensor(history[batch[:, 0]], device=device)
            targets = torch.as_tensor(batch[:, 1], device=device)
            optimizer.zero_grad(set_to_none=True)
            if hard_rows is None:
                logits = model(table, h, targets)
            else:
                negatives = hard_rows[batch[:,0]]
                hard_ids = torch.as_tensor(negatives, device=device)
                logits, hard_logits = model(table, h, targets, hard_ids)
            column_q = None if pairing is None else q_table[targets]
            loss_sum = inbatch_loss(logits, torch.as_tensor(mask, device=device), column_q)
            if hard_rows is not None:
                usable = mask.sum(axis=1) > 1
                term = hard_loss(logits.diagonal(), hard_logits, hard_ids.ne(0),
                                 torch.as_tensor(usable, device=device))
                hard_value = term.item()
                require(math.isfinite(hard_value), f"seed {seed}, epoch {epoch+1}, batch {offset}: nonfinite hard loss")
                hard_sum_epoch += hard_value
                loss_sum = loss_sum + coefficient * term
                used = negatives[usable]
                hard_counts["rows"] += int(np.any(used != 0, axis=1).sum())
                used = used[used != 0]
                hard_counts["uses"] += len(used)
                # Mined IDs exclude all row positives. Every matching batch target
                # is therefore also an eligible in-batch negative for this row.
                targets_sorted = np.sort(batch[:,1])
                copies = np.searchsorted(targets_sorted, used, side="right") - np.searchsorted(targets_sorted, used)
                hard_counts["overlapping_items"] += int(np.count_nonzero(copies))
                hard_counts["overlapping_columns"] += int(copies.sum())
            value = loss_sum.item()
            require(math.isfinite(value), f"epoch {epoch+1}, offset {offset}: nonfinite training loss")
            (loss_sum / valid).backward()
            optimizer.step()
            total_loss += value
        require(counts["valid_rows"] + counts["skipped_rows"] == len(examples) and counts["valid_rows"] > 0,
                "training row accounting mismatch")
        require(torch.isfinite(model.words.weight).all().item(), "nonfinite learned weights")
        summary = {"epoch": epoch + 1, "loss": total_loss / counts["valid_rows"], **counts}
        if hard_rows is not None:
            summary["hard"] = {**hard_counts, "loss": hard_sum_epoch / counts["valid_rows"]}
            if cfg["epochs"] == 3 and pairing is not None:
                require(hard_counts["uses"] > 0, f"seed {seed}, epoch {epoch+1}: no hard samples")
        if pairing is not None:
            summary["batch_sha256"] = sequence.hexdigest()
            require(summary["batch_sha256"] == reference["seeds"][str(seed)][epoch],
                    f"seed {seed} epoch {epoch+1}: batch sequence mismatch")
            prior = control["training"]["epochs"][epoch]
            if hard_rows is not None:
                require(prior["batch_sha256"] == summary["batch_sha256"], "CARD-005 batch differs")
            require(all(counts[k] == prior[k] for k in counts),
                    f"seed {seed} epoch {epoch+1}: row or mask counts differ")
        results.append(summary)
        durations.append(perf_counter() - start)
        print("epoch", summary, "seconds", durations[-1], flush=True)
    result = {"positive_occurrences": len(examples), "epochs": results}
    if q_summary is not None:
        result["q"] = q_summary
    return result, durations


class SliceMetrics:
    def __init__(self):
        self.items = []
        self.retrieval = Retrieval()

    def add(self, item, clicks, covered, hits):
        self.items.append(item)
        self.retrieval.clicks += clicks
        self.retrieval.covered += covered
        self.retrieval.hits += hits

    def report(self):
        return {"official": evaluate(self.items), "retrieval": self.retrieval.report()}


@torch.inference_mode()
def replay(model, table, train, valid, index, cfg, prefix_size, *, lengths=False):
    model.eval()
    batch_size = cfg["evaluation_batch"]
    bank = torch.cat([model.news(chunk) for chunk in table.split(batch_size)])
    require(torch.isfinite(bank).all().item(), "nonfinite news vectors")
    history = histories(valid, index, cfg["history_limit"])
    groups = [(g, row) for g, (_, rows) in enumerate(groupby(valid, key=lambda x: x.timestamp)) for row in rows]
    visibility = Visibility.from_replay(train, groups, index)
    slices = {s: SliceMetrics() for s in ("full", "empty_history", "with_history", "prefix")}
    if lengths:
        slices.update(short_history=SliceMetrics(), long_history=SliceMetrics())
    for offset in range(0, len(valid), batch_size):
        h = torch.as_tensor(history[offset:offset+batch_size], device=table.device)
        users = model.aggregate_history(bank[h], h)
        scores = (users @ bank.T).cpu().numpy()
        require(np.isfinite(scores).all(), "nonfinite validation dot products")
        for i, dots in enumerate(scores):
            g, row = groups[offset+i]
            pool = visibility.before(g)
            top = set(exact_top(dots, pool, cfg["top_k"]))
            ids = [index[n] for n in row.candidates]
            item = Impression(row.labels, ranks_from_scores(row.candidates, dots[ids]))
            clicks = [index[n] for n in row.clicks()]
            covered = sum(bool(pool[n]) for n in clicks)
            hits = sum(n in top for n in clicks)
            keys = ["full", "with_history" if row.history else "empty_history"]
            if lengths and row.history:
                keys.append("short_history" if len(row.history) < 20 else "long_history")
            if offset+i < prefix_size:
                keys.append("prefix")
            for key in keys:
                slices[key].add(item, len(clicks), covered, hits)
    return {name: acc.report() for name, acc in slices.items()}


def weight_identity(model):
    return hashlib.sha256(model.words.weight.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def verify_mean_metrics(actual, reference):
    """Match all legacy slice counts exactly and all legacy floating metrics within 1e-8."""
    def compare(new, old, path):
        if isinstance(old, dict):
            require(new.keys() == old.keys(), f"{path}: metric keys differ")
            for key, value in old.items():
                compare(new[key], value, f"{path}/{key}")
        elif isinstance(old, float):
            require(math.isclose(new, old, rel_tol=0, abs_tol=1e-8), f"{path}: {new} vs {old}")
        else:
            require(new == old, f"{path}: {new} vs {old}")
    for name, old in reference.items():
        compare(actual[name], old, name)


def verify_groups(metrics, expected):
    full = metrics["full"]
    for name, counts in expected.items():
        part = metrics[name]
        observed = [part["official"]["scoring_denominator"], part["retrieval"]["clicks"],
                    part["retrieval"]["covered"]]
        require(observed == counts, f"{name}: unexpected denominator {observed}")
    for section, keys in (("official", ("total", "scoring_denominator")),
                          ("retrieval", ("clicks", "covered", "hits"))):
        for key in keys:
            require(sum(metrics[name][section][key] for name in expected) == full[section][key],
                    f"{section}/{key}: groups do not sum to full")


def load_frozen_negatives(rows, index, hard, control, origin):
    """The frozen list belongs to the CARD-006 files commit, across all later experiments."""
    artifacts = control["mining_artifacts"]
    path = ROOT / hard["mined_file"]
    audit_path = ROOT / hard["audit_file"]
    require(sha256_file(audit_path) == artifacts[hard["audit_file"]], "mining audit hash differs")
    audit = json.loads(audit_path.read_bytes())
    require(sha256_file(path) == artifacts[hard["mined_file"]] == audit["list_sha256"]
            and audit["commit"] == origin, "frozen mining source differs")
    require(all(audit[k] == v for k, v in hard["expected"].items()), "mining counts differ")
    result = np.zeros((len(rows), hard["max_negatives"]), dtype=np.int64)
    with path.open(encoding="utf-8") as stream:
        for i, (event, line) in enumerate(zip(rows, stream, strict=True)):
            entry = json.loads(line)
            require(entry["id"] == event.impression_id, "frozen mining exposure differs")
            selected = [index[news] for news, _ in entry["items"]]
            result[i, :len(selected)] = selected
    return result


def attention_identity(model):
    params = list(model.projection.parameters()) + [model.query]
    require(all(torch.isfinite(p).all().item() for p in params), "nonfinite attention weights")
    return {name: hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, p in model.named_parameters() if name != "words.weight"}


def paired_deltas(mean, attention):
    differences = {}
    for name, part in mean.items():
        other = attention[name]
        require(all(part["official"][k] == other["official"][k]
                    for k in ("total", "excluded", "scoring_denominator"))
                and all(part["retrieval"][k] == other["retrieval"][k] for k in ("clicks", "covered")),
                f"{name}: paired denominators differ")
        differences[name] = {"nDCG@10": other["official"]["metrics"]["nDCG@10"]
                             - part["official"]["metrics"]["nDCG@10"]}
        for key in ("recall", "coverage", "conditional_recall"):
            differences[name][key] = other["retrieval"][key] - part["retrieval"][key]
    return differences


def review_mean(control, seed, device, table, train, valid, index, cfg, prefix, settings, snap, fitted_vocab):
    name = settings["reference_objects"][seed]
    record = snap[name]
    meta = json.loads((ROOT / "results/rv" / (record["version"] + ".json")).read_bytes())["_meta"]
    origin = git("rev-parse", "--verify", "card-006^{commit}", check=True).stdout.strip()
    require(meta["dirty"] is False and meta["seed"] == seed and meta["commit"] == origin
            and meta["config_sha256"] == settings["hard_config_sha256"]
            and meta["lock"] == cfg["lock"] and meta["data_version"] == cfg["data_version"],
            "mean RV provenance differs")
    folder = Path("checkpoints/card-006") / f"s{seed}"
    weights = folder / "model.pt"
    vocabulary = folder / "vocabulary.json"
    for path in (weights, vocabulary):
        require(sha256_file(ROOT / path) == control["artifacts"][path.as_posix()],
                f"mean checkpoint file differs: {path}")
    frozen_vocab = json.loads((ROOT / vocabulary).read_bytes())
    require(frozen_vocab == fitted_vocab, "mean replay token IDs differ from frozen vocabulary")
    mean = MeanTowers(control["vocab_size"], cfg["embedding_dim"]).to(device)
    mean.load_state_dict(torch.load(ROOT / weights, map_location=device, weights_only=True), strict=True)
    require(weight_identity(mean) == control["trained_weight_sha256"], "loaded mean weights differ")
    metrics = replay(mean, table, train, valid, index, cfg, prefix, lengths=True)
    verify_mean_metrics(metrics, control["metrics"])
    verify_groups(metrics, settings["group_counts"])
    return metrics, origin


def run(cfg, seed, dest, started, experiment=None, hard=None, attention=None):
    snap = snapshot()
    require(cfg["data_version"] == snap["DATA"]["version"] and cfg["lock"] == snap["LOCK"]["version"],
            "data or lock version mismatch")
    profile = registered_ref(cfg["split_rv"], snap)
    base_a, base_b = [registered_ref(name, snap) for name in cfg["baseline_rvs"]]
    require(base_a == base_b, "G2 runs disagree")
    print("seed", seed, "phase: parsing and fitting", flush=True)
    rows, texts = read_sources(cfg)
    train, prefix = split_prefix(rows, profile["split"], cfg)
    valid = rows[len(train):]
    sample = {"count": len(prefix), "start": str(prefix[0].timestamp), "end": str(prefix[-1].timestamp),
              "id_sequence_sha256": hashlib.sha256("".join(r.impression_id+"\n" for r in prefix).encode("ascii")).hexdigest()}
    require(sample == base_a["sample"], "G2 prefix mismatch")
    fit_ids = {n for row in train for source in (row.history, row.candidates) for n in source}
    all_ids = fit_ids.union(n for row in valid for source in (row.history, row.candidates) for n in source)
    names, vocab, table = build_catalog(texts, fit_ids, all_ids, cfg)
    index = {n: i for i, n in enumerate(names, 1)}
    checkpoint_root = experiment["checkpoint_root"] if experiment else "checkpoints/card-004"
    folder = ROOT / checkpoint_root / f"s{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    vfile, weights = folder / "vocabulary.json", folder / "model.pt"
    vfile.write_text(json.dumps(vocab, ensure_ascii=False, sort_keys=True), encoding="utf8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("highest")
    if attention:
        model = AttentionTowers(len(vocab), cfg["embedding_dim"], attention["hidden_dim"]).to(device)
        initial_attention = attention_identity(model)
    else:
        model = MeanTowers(len(vocab), cfg["embedding_dim"]).to(device)
    table = table.to(device)
    initial = weight_identity(model)
    pairing = None
    if experiment:
        ref_name = experiment["reference_objects"][seed]
        control = registered_ref(ref_name, snap)
        control_card = "006" if attention else ("005" if hard else "004")
        expected_vocab = control["artifacts"][f"checkpoints/card-{control_card}/s{seed}/vocabulary.json"]
        require(control["seed"] == seed and control["initial_weight_sha256"] == initial
                and sha256_file(vfile) == expected_vocab, f"seed {seed}: initialization or vocabulary mismatch")
        audit = json.loads((ROOT / experiment["audit"]).read_bytes())
        if hard:
            require(sha256_file(ROOT / experiment["audit"]) == control["sampler_reference_sha256"],
                    "CARD-005 sampler reference mismatch")
        pairing = (audit, control)
    print("seed", seed, "phase: training on", device, flush=True)
    if attention:
        started_mean = perf_counter()
        mean_metrics, origin = review_mean(control, seed, device, table, train, valid, index, cfg,
                                         len(prefix), attention, snap, vocab)
        mean_seconds = perf_counter() - started_mean
        hard_rows = load_frozen_negatives(train, index, hard, control, origin)
    else:
        hard_rows = load_verified(train, index, hard) if hard else None
    training, epoch_seconds = train_epochs(model, table, train, index, cfg, device, seed, pairing=pairing,
                                           hard_rows=hard_rows, coefficient=hard["loss_weight"] if hard else 1.)
    final = weight_identity(model)
    require(initial != final, "weights did not change")
    torch.save(model.state_dict(), weights)
    evaluation_start = perf_counter()
    print("seed", seed, "phase: exact validation replay", flush=True)
    metrics = replay(model, table, train, valid, index, cfg, len(prefix), lengths=bool(attention))
    evaluation_seconds = perf_counter() - evaluation_start
    full = metrics["full"]["official"]
    require(full["total"] == len(valid) and full["scoring_denominator"] == profile["split"]["validation"]["scoring_denominator"],
            "validation denominator mismatch")
    for baseline in base_a["models"].values():
        for k in ("total", "excluded", "scoring_denominator"):
            require(metrics["prefix"]["official"][k] == baseline["official"][k], "prefix official denominator mismatch")
        for k in ("clicks", "covered"):
            require(metrics["prefix"]["retrieval"][k] == baseline["retrieval"][k], "prefix recall denominator mismatch")
    for part in metrics.values():
        for value in list((part["official"]["metrics"] or {}).values()) + list(part["retrieval"].values()):
            require(value is None or math.isfinite(value), "nonfinite evaluation")
    payload = {"seed": seed, "training": training, "metrics": metrics, "prefix_sample": sample,
               "references": {k: snap[k]["identity"] for k in [cfg["split_rv"], *cfg["baseline_rvs"]]},
               "train_impressions": len(train), "fit_news_ids": len(fit_ids), "vocab_size": len(vocab),
               "initial_weight_sha256": initial, "trained_weight_sha256": final,
               "artifacts": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in (vfile, weights)}}
    extra = {"device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
             "epoch_seconds": epoch_seconds, "evaluation_seconds": evaluation_seconds,
             "elapsed_seconds": perf_counter() - started}
    require(extra["elapsed_seconds"] < cfg["seconds_per_seed"], "seed time limit reached")
    if experiment:
        require(sample == control["prefix_sample"], "paired prefix differs")
        for name, before in control["metrics"].items():
            section = metrics[name]
            require(all(section["official"][k] == before["official"][k]
                        for k in ("total", "scoring_denominator", "excluded"))
                    and all(section["retrieval"][k] == before["retrieval"][k] for k in ("clicks", "covered")),
                    f"seed {seed}: {name} denominators differ")
        payload["references"][ref_name] = snap[ref_name]["identity"]
        payload["base_config_sha256"] = experiment["base_config_sha256"]
        payload["sampler_reference_sha256"] = sha256_file(ROOT / experiment["audit"])
    config_path = LOGQ_CONFIG if experiment else CONFIG
    if hard:
        config_path = HARD_CONFIG
        require(training["q"] == control["training"]["q"], "paired training Q mismatch")
        payload["mining_artifacts"] = {hard[key]: sha256_file(ROOT / hard[key])
                                       for key in ("mined_file", "audit_file")}
        payload["logq_config_sha256"] = hard["logq_config_sha256"]
    if attention:
        config_path = ATTENTION_CONFIG
        verify_groups(metrics, attention["group_counts"])
        final_attention = attention_identity(model)
        require(all(initial_attention[k] != final_attention[k] for k in final_attention),
                "attention parameters did not all update")
        for epoch, prior in zip(training["epochs"], control["training"]["epochs"], strict=True):
            require(all(epoch["hard"][key] == prior["hard"][key]
                        for key in ("rows", "uses", "overlapping_items", "overlapping_columns")),
                    "hard-negative usage differs")
        payload.update(mean_metrics=mean_metrics, paired_delta=paired_deltas(mean_metrics, metrics),
                       mean_artifacts=control["artifacts"], hard_config_sha256=attention["hard_config_sha256"],
                       initial_attention_sha256=initial_attention, final_attention_sha256=final_attention)
        require(metrics["empty_history"] == mean_metrics["empty_history"], "empty history ranking changed")
        extra["mean_evaluation_seconds"] = mean_seconds
    write_result(dest, payload, "RV", Path(dest).stem, config_path, seed, extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--logq", action="store_true")
    mode.add_argument("--hard", action="store_true")
    mode.add_argument("--attention", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    cfg = json.loads((ROOT / CONFIG).read_bytes())
    experiment = json.loads((ROOT / LOGQ_CONFIG).read_bytes()) if args.logq or args.hard or args.attention else None
    hard = json.loads((ROOT / HARD_CONFIG).read_bytes()) if args.hard or args.attention else None
    attention = json.loads((ROOT / ATTENTION_CONFIG).read_bytes()) if args.attention else None
    if hard:
        require(sha256_file(ROOT / LOGQ_CONFIG) == hard["logq_config_sha256"], "logQ config identity mismatch")
        experiment.update({k: hard[k] for k in ("checkpoint_root","scratch_prefix","reference_objects")})
    if attention:
        require(sha256_file(ROOT / HARD_CONFIG) == attention["hard_config_sha256"],
                "frozen hard-negative configuration differs")
        experiment.update({k: attention[k] for k in ("checkpoint_root", "scratch_prefix", "reference_objects")})
    if experiment:
        require(experiment["base_config"] == CONFIG and sha256_file(ROOT / CONFIG) == experiment["base_config_sha256"],
                "paired base config mismatch")
    def timed_out():
        print(f"BLOCK seed {args.seed}: reached {cfg['seconds_per_seed']} s", flush=True)
        os._exit(124)
    timer = Timer(cfg["seconds_per_seed"], timed_out)
    timer.daemon = True
    timer.start()
    try:
        tag = "card-006" if hard else ("card-005" if experiment else "card-004")
        if attention:
            tag = "card-007"
        require(not status_paths() and head() == git("rev-parse", "--verify", tag+"^{commit}", check=True).stdout.strip(),
                f"requires clean {tag}")
        prefix = experiment["scratch_prefix"] if experiment else "scratch/card-004/RV-two-tower-001-s"
        dest = f"{prefix}{args.seed}.json"
        require(not (ROOT / dest).exists(), "RV already exists")
        run(cfg, args.seed, dest, started, experiment, hard, attention)
    finally:
        timer.cancel()


if __name__ == "__main__":
    main()
