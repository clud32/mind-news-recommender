"""Compare frozen MIND news and query vectors with three CPU inner-product indexes."""
import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter_ns

import faiss
import numpy as np
import torch

from mind_two_tower import MeanTowers, build_catalog
from profile_mind_small import require
from run_mind_baselines import split_prefix
from run_mind_two_tower import histories, read_sources, registered_ref, weight_identity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import git, head, payload_sha256, sha256_file, snapshot, status_paths
from tools.rv import write_result

CONFIG = "configs/mind-faiss-v0.1.json"
METHODS = ("Flat", "HNSWFlat", "IVFPQ")


def read(path):
    return json.loads(path.read_bytes())


def save(path, data):
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.write("\n")


def relative(path):
    return path.relative_to(ROOT).as_posix()


def identities(paths):
    return {relative(p): sha256_file(p) for p in paths}


def encoder_source(cfg, snap):
    base_path = ROOT / cfg["encoder_config"]
    base = read(base_path)
    require(sha256_file(base_path) == cfg["encoder_config_sha256"] and
            base["boundary"] == cfg["boundary"] and base["embedding_dim"] == cfg["dimension"],
            "encoder config or boundary differs")
    for name, digest in ((cfg["encoder_object"], cfg["encoder_payload"]),
                         (cfg["split_object"], cfg["split_payload"])):
        require(snap[name]["identity"] == digest, f"{name} identity differs")
    obj = snap[cfg["encoder_object"]]
    doc = read(ROOT / "results/rv" / (obj["version"] + ".json"))
    meta = doc["_meta"]
    require(meta["commit"] == git("rev-parse", "card-004", check=True).stdout.strip()
            and meta["seed"] == cfg["encoder_seed"] == 0 and meta["dirty"] is False
            and meta["config_sha256"] == sha256_file(base_path)
            and meta["lock"] == snap["LOCK"]["version"]
            and meta["data_version"] == snap["DATA"]["version"], "encoder provenance differs")
    control = registered_ref(cfg["encoder_object"], snap)
    for name, digest in control["artifacts"].items():
        require(sha256_file(ROOT / name) == digest, f"checkpoint changed: {name}")
    return base, control


@torch.inference_mode()
def freeze(cfg, folder):
    snap = snapshot()
    base, control = encoder_source(cfg, snap)
    rows, texts = read_sources(base)
    split = registered_ref(cfg["split_object"], snap)["split"]
    train, _ = split_prefix(rows, split, base)
    valid = rows[len(train):]
    require(train[-1].timestamp < valid[0].timestamp and str(valid[0].timestamp) == cfg["boundary"],
            "chronological split differs")
    corpus = sorted({n for row in train for n in row.candidates})
    require(len(corpus) == cfg["count"] and len(valid) == cfg["queries"], "unexpected input sizes")
    fit_ids = set(corpus).union(n for row in train for n in row.history)
    all_ids = fit_ids.union(n for row in valid for n in row.history)
    names, vocab, tokens = build_catalog(texts, fit_ids, all_ids, base)
    index = {name: i for i, name in enumerate(names, 1)}
    frozen = ROOT / "checkpoints/card-004/s0"
    require(read(frozen / "vocabulary.json") == vocab, "frozen token IDs differ")
    model = MeanTowers(len(vocab), cfg["dimension"]).eval()
    model.load_state_dict(torch.load(frozen / "model.pt", map_location="cpu", weights_only=True))
    require(weight_identity(model) == control["trained_weight_sha256"], "loaded encoder tensor differs")
    bank = torch.cat([model.news(batch) for batch in tokens.split(256)])
    news = bank[[index[n] for n in corpus]].contiguous().numpy()
    history = torch.from_numpy(histories(valid, index, base["history_limit"]))
    queries = torch.cat([model.aggregate_history(bank[h], h) for h in history.split(256)]).numpy()
    require(all(a.dtype == np.float32 and np.isfinite(a).all() for a in (news, queries)), "invalid vectors")
    clicks = [list(row.clicks()) for row in valid]
    ids = [row.impression_id for row in valid]
    require(len(set(ids)) == len(ids) and sum(map(len, clicks)) == cfg["clicks"], "query labels differ")
    corpus_set = set(corpus)
    covered = sum(n in corpus_set for clicked in clicks for n in clicked)
    require(covered == cfg["covered"], "frozen corpus coverage differs")
    folder.mkdir(parents=True, exist_ok=False)
    np.save(folder / "news.npy", news, allow_pickle=False)
    np.save(folder / "queries.npy", queries, allow_pickle=False)
    save(folder / "news-ids.json", corpus)
    save(folder / "queries.json", [{"id": iid, "clicks": c} for iid, c in zip(ids, clicks, strict=True)])
    manifest = dict(commit=head(), config_sha256=sha256_file(ROOT / CONFIG),
                    encoder_payload=cfg["encoder_payload"], encoder_files=control["artifacts"],
                    split_payload=cfg["split_payload"], boundary=cfg["boundary"],
                    train=len(train), validation=len(valid), corpus=len(corpus), clicks=sum(map(len, clicks)),
                    covered=covered, news_id_sha256=payload_sha256(corpus),
                    impression_id_sha256=payload_sha256(ids), artifacts=identities(sorted(folder.iterdir())))
    save(folder / "inputs.json", manifest)
    print("inputs sha256", sha256_file(folder / "inputs.json"), flush=True)


def exact(index, query, k):
    """Rows are inserted in ascending lexical ID order. Fetch the whole kth-score tie."""
    count = index.ntotal
    require(count > 0 and k > 0, "empty index or nonpositive cutoff")
    width = min(count, k + 1)
    while True:
        scores, ids = index.search(query, width)
        scores, ids = scores[0], ids[0]
        require(np.isfinite(scores).all() and (ids >= 0).all(), "invalid Flat result")
        cutoff = scores[min(k, count) - 1]
        if width == count or scores[-1] < cutoff:
            break
        width = min(count, 2 * width)
    order = np.lexsort((ids, -scores))
    tie_count = int(np.count_nonzero(scores == cutoff))
    crosses = len(scores) > k and scores[k] == cutoff
    return ids[order[:k]], tie_count, bool(crosses)


def ann_ids(raw, size):
    ids = raw[raw != -1]
    require(((ids >= 0) & (ids < size)).all() and len(set(ids)) == len(ids), "invalid ANN rows")
    return ids, len(raw) - len(ids)


def build(method, news, cfg, seed):
    d = news.shape[1]
    if method == "Flat":
        index = faiss.IndexFlatIP(d)
    elif method == "HNSWFlat":
        h = cfg["hnsw"]
        index = faiss.IndexHNSWFlat(d, h["M"], faiss.METRIC_INNER_PRODUCT)
        index.hnsw.rng = faiss.RandomGenerator(seed)
        index.hnsw.efConstruction = h["efConstruction"]
        index.hnsw.efSearch = h["efSearch"]
    else:
        iv = cfg["ivfpq"]
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, iv["nlist"], iv["m"], iv["nbits"],
                                 faiss.METRIC_INNER_PRODUCT)
        index.cp.seed = index.pq.cp.seed = seed
        index.nprobe = iv["nprobe"]
    train_ns = 0
    if not index.is_trained:
        start = perf_counter_ns()
        index.train(news)
        train_ns = perf_counter_ns() - start
    start = perf_counter_ns()
    index.add(news)
    return index, train_ns, perf_counter_ns() - start


def measure(index, queries, selected, k, warmup):
    for i in selected[:warmup]:
        index.search(queries[i:i+1], k)
    times = []
    for i in selected:
        query = queries[i:i+1]
        start = perf_counter_ns()
        index.search(query, k)
        times.append(perf_counter_ns() - start)
    require(all(t > 0 for t in times), "invalid latency")
    return times


def ratio(num, den):
    return num / den if den else None


def summarize(records, method, k):
    nonzero = [r for r in records if not r["zero"]]
    untied = [r for r in nonzero if r["boundary_group"] == 1]
    unambiguous = [r for r in nonzero if not r["crosses"]]
    clicks = sum(r["clicks"] for r in records)
    covered = sum(r["covered"] for r in records)
    hits = sum(r[method]["hits"] for r in records)
    require(0 <= hits <= covered <= clicks, "click accounting differs")
    def overlap(rows):
        total = sum(r[method]["intersection"] for r in rows)
        return {"queries": len(rows), "intersection": total, "overlap": ratio(total, k * len(rows))}
    return dict(queries=len(records), nonzero=len(nonzero), zero=len(records)-len(nonzero),
                boundary_tied=len(nonzero)-len(untied), boundary_crossed=len(nonzero)-len(unambiguous),
                overlap_all=overlap(nonzero), overlap_without_ties=overlap(untied),
                overlap_without_crossing=overlap(unambiguous),
                missing=sum(r[method]["missing"] for r in records),
                clicks=clicks, covered=covered, hits=hits, recall=ratio(hits, clicks),
                coverage=ratio(covered, clicks), conditional_recall=ratio(hits, covered))


def quality(indexes, queries, corpus, metadata, k):
    rows = {name: i for i, name in enumerate(corpus)}
    lexical = np.arange(min(len(corpus), k))
    records = []
    for i, (q, info) in enumerate(zip(queries, metadata, strict=True)):
        query = q.reshape(1, -1)
        zero = not np.any(q)
        if zero:
            top, group, crosses = lexical, 0, False
        else:
            top, group, crosses = exact(indexes["Flat"], query, k)
        reference = set(top)
        positives = [rows[n] for n in info["clicks"] if n in rows]
        record = dict(id=info["id"], zero=zero, boundary_group=group, crosses=crosses,
                      clicks=len(info["clicks"]), covered=len(positives))
        for method, index in indexes.items():
            ids, missing = top, 0
            if not zero and method != "Flat":
                _, raw = index.search(query, k)
                ids, missing = ann_ids(raw[0], len(corpus))
            found = set(ids)
            record[method] = dict(intersection=len(found & reference), missing=missing,
                                  hits=sum(n in found for n in positives))
        records.append(record)
    return records


def cpu_name():
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor() or platform.machine()


def benchmark(cfg, folder, seed, expected):
    manifest = folder / "inputs.json"
    require(sha256_file(manifest) == expected, "shared input manifest changed")
    inputs = read(manifest)
    require(inputs["commit"] == head() and inputs["config_sha256"] == sha256_file(ROOT / CONFIG),
            "frozen input provenance changed")
    for name, digest in (inputs["artifacts"] | inputs["encoder_files"]).items():
        require(sha256_file(ROOT / name) == digest, f"frozen input changed: {name}")
    news = np.load(folder / "news.npy", allow_pickle=False)
    queries = np.load(folder / "queries.npy", allow_pickle=False)
    corpus, metadata = read(folder / "news-ids.json"), read(folder / "queries.json")
    require(corpus == sorted(set(corpus)) and news.shape == (cfg["count"], cfg["dimension"])
            and queries.shape == (cfg["queries"], cfg["dimension"]), "input row dimensions differ")
    require(payload_sha256(corpus) == inputs["news_id_sha256"] and
            payload_sha256([r["id"] for r in metadata]) == inputs["impression_id_sha256"],
            "frozen ID order differs")
    chosen = np.flatnonzero(queries.any(axis=1))[:cfg["timed"]].tolist()
    require(len(chosen) == cfg["timed"], "insufficient nonzero timing queries")
    selected_ids = [metadata[i]["id"] for i in chosen]
    dest = folder.parent / f"s{seed}"
    dest.mkdir(exist_ok=False)
    order = list(METHODS[seed:] + METHODS[:seed])
    indexes, stats, durations = {}, {}, {}
    for method in order:
        idx, train_ns, add_ns = build(method, news, cfg, seed)
        f = dest / (method + ".faiss")
        faiss.write_index(idx, str(f))
        loaded = faiss.read_index(str(f))
        require(loaded.ntotal == len(news) and loaded.metric_type == faiss.METRIC_INNER_PRODUCT,
                "index read-back failed")
        probe = queries[chosen[:4]]
        before, after = idx.search(probe, cfg["k"]), loaded.search(probe, cfg["k"])
        require(all(np.array_equal(x, y) for x, y in zip(before, after)), "index roundtrip differs")
        indexes[method] = loaded
        durations[method] = measure(loaded, queries, chosen, cfg["k"], cfg["warmup"])
        p50, p95 = np.percentile(durations[method], [50, 95])
        stats[method] = dict(train_seconds=train_ns / 1e9, add_seconds=add_ns / 1e9,
                             bytes=f.stat().st_size, p50_us=float(p50 / 1000), p95_us=float(p95 / 1000))
    records = quality(indexes, queries, corpus, metadata, cfg["k"])
    save(dest / "query-quality.json", records)
    save(dest / "timings.json", dict(ids=selected_ids, rows=chosen, nanoseconds=durations))
    for method in METHODS:
        stats[method].update(summarize(records, method, cfg["k"]))
        require(stats[method]["clicks"] == cfg["clicks"] and
                stats[method]["covered"] == inputs["covered"] == cfg["covered"],
                "validation click denominators changed")
    require(stats["Flat"]["overlap_all"]["overlap"] == 1, "exact reference mismatch")
    payload = dict(encoder_seed=0, index_seed=seed, inputs_sha256=expected, inputs_path=relative(manifest), inputs=inputs,
                   order=order, timings_id_sha256=payload_sha256(selected_ids),
                   timing_count=len(chosen), warmup=cfg["warmup"], threads=faiss.omp_get_max_threads(),
                   cpu=cpu_name(), faiss_version=faiss.__version__, methods=stats,
                   artifacts=identities(sorted(dest.iterdir())))
    output = f"scratch/card-008/RV-faiss-001-s{seed}.json"
    require(not (ROOT / output).exists(), "existing RV")
    write_result(output, payload, "RV", Path(output).stem, CONFIG, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--inputs-sha256")
    args = parser.parse_args()
    cfg = read(ROOT / CONFIG)
    require(not status_paths() and head() == git("rev-parse", "card-008", check=True).stdout.strip(),
            "requires clean card-008")
    faiss.omp_set_num_threads(cfg["threads"])
    torch.set_num_threads(cfg["threads"])
    folder = ROOT / cfg["folder"] / "inputs"
    if args.freeze:
        freeze(cfg, folder)
    else:
        benchmark(cfg, folder, args.seed, args.inputs_sha256)


if __name__ == "__main__":
    main()
