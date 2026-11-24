"""Profile the registered MIND training split and record its chronological boundary."""
import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/mind-small-split-v0.1.json"
OUTPUT = "results/rv/RV-data-profile-001-r1.json"
GROUPS = ("empty_labels", "all_positive", "all_negative", "mixed")
NEWS_ID = re.compile(r"N[0-9]+")
CANDIDATE = re.compile(r"(N[0-9]+)(?:-([01]))?")
TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"


class ProfileError(ValueError):
    """Input errors carry the source file, line and error type."""


def require(condition, message):
    if not condition:
        raise ProfileError(message)


def fields(lines, name, count):
    for line_no, raw in enumerate(lines, 1):
        where = f"{name}:{line_no}"
        try:
            row = raw.decode("utf-8").rstrip("\r\n").split("\t")
        except UnicodeDecodeError as exc:
            raise ProfileError(f"{where}: invalid_utf8") from exc
        require(len(row) == count, f"{where}: field_count expected {count}, got {len(row)}")
        yield where, row


@dataclass(slots=True)
class Impression:
    iid: str
    user: str
    time: datetime
    source_time: str
    candidates: tuple
    group: str
    where: str


def parse_impression(row, where):
    iid, user, source_time, history_text, candidate_text = row
    require(re.fullmatch(r"[0-9]+", iid) and re.fullmatch(r"U[0-9]+", user),
            f"{where}: invalid_impression_or_user_id")
    try:
        time = datetime.strptime(source_time, TIME_FORMAT)
    except ValueError as exc:
        raise ProfileError(f"{where}: invalid_time {source_time!r}") from exc
    history = history_text.split()
    require(all(NEWS_ID.fullmatch(n) for n in history), f"{where}: invalid_history_news_id")
    candidates = []
    positives = 0
    labeled = 0
    for token in candidate_text.split():
        match = CANDIDATE.fullmatch(token)
        require(match, f"{where}: invalid_candidate_or_label {token!r}")
        news, label = match.groups()
        candidates.append(sys.intern(news))
        if label is not None:
            labeled += 1
            positives += int(label)
    require(labeled in (0, len(candidates)), f"{where}: partially_labeled_impression")
    if not labeled:
        group = "empty_labels"
    elif positives == labeled:
        group = "all_positive"
    elif positives == 0:
        group = "all_negative"
    else:
        group = "mixed"
    return Impression(iid, user, time, source_time, tuple(candidates), group, where), history, positives


def summarize_side(rows, total):
    groups = Counter(row.group for row in rows)
    counts = {name: groups[name] for name in GROUPS}
    require(sum(counts.values()) == len(rows), "label_counts_do_not_partition_side")
    return {
        "impressions": len(rows), "fraction": len(rows) / total,
        "start": rows[0].source_time, "end": rows[-1].source_time,
        "labels": counts, "scoring_denominator": counts["mixed"],
    }


def profile(news_lines, behavior_lines, train_fraction):
    known = set()
    news_rows = empty_abstract = 0
    for where, row in fields(news_lines, "news.tsv", 8):
        require(NEWS_ID.fullmatch(row[0]), f"{where}: invalid_news_id")
        known.add(sys.intern(row[0]))
        news_rows += 1
        empty_abstract += not row[4].strip()

    rows = []
    impression_ids = Counter()
    lengths = Counter()
    missing_candidates = Counter()
    missing_history = Counter()
    duplicate_rows = duplicate_tokens = empty_history = multi_click = 0
    example = None
    for where, row in fields(behavior_lines, "behaviors.tsv", 5):
        obs, history, clicks = parse_impression(row, where)
        rows.append(obs)
        impression_ids[obs.iid] += 1
        lengths[len(obs.candidates)] += 1
        extra = len(obs.candidates) - len(set(obs.candidates))
        duplicate_rows += extra > 0
        duplicate_tokens += extra
        empty_history += not history
        multi_click += clicks > 1
        missing_candidates.update(n for n in obs.candidates if n not in known)
        missing_history.update(n for n in history if n not in known)
        if example is None:
            example = {"impression": "<impression>", "user": "<user>", "time": obs.source_time,
                       "history_items": len(history), "candidate_items": len(obs.candidates),
                       "positive_labels": clicks, "label_group": obs.group}

    require(rows, "empty_behaviors")
    rows.sort(key=lambda r: r.time)
    total = len(rows)
    numerator, denominator = train_fraction
    index = total * numerator // denominator
    boundary = rows[index]
    cut = next(i for i, row in enumerate(rows) if row.time == boundary.time)
    train, valid = rows[:cut], rows[cut:]
    require(train and valid and len(train) + len(valid) == total, "split_size")
    require(train[-1].time < valid[0].time == boundary.time, "split_boundary")
    tied = sum(r.time == boundary.time for r in rows)
    require(sum(r.time == boundary.time for r in valid) == tied, "boundary_group_split")
    train_ids = {r.iid for r in train}
    for row in valid:
        require(row.iid not in train_ids, f"{row.where}: cross_split_impression_id {row.iid}")
    train_news = {n for r in train for n in r.candidates}
    valid_news = {n for r in valid for n in r.candidates}
    return {
        "news": {"rows": news_rows, "distinct_ids": len(known), "empty_abstract": empty_abstract},
        "impressions": {
            "total": total, "start": rows[0].source_time, "end": rows[-1].source_time,
            "empty_history": empty_history, "multiple_positive_labels": multi_click,
            "candidate_count_histogram": {str(size): count for size, count in sorted(lengths.items())},
            "repeated_impression_ids": sum(v > 1 for v in impression_ids.values()),
            "excess_impression_rows": total - len(impression_ids),
            "rows_with_repeated_candidates": duplicate_rows, "excess_candidate_tokens": duplicate_tokens,
            "missing_candidate_ids": len(missing_candidates), "missing_candidate_occurrences": missing_candidates.total(),
            "missing_history_ids": len(missing_history), "missing_history_occurrences": missing_history.total(),
            "example": example,
        },
        "split": {
            "boundary_source_time": boundary.source_time, "zero_based_index": index,
            "impressions_at_boundary": tied, "timezone": "source time; timezone unspecified",
            "train": summarize_side(train, total), "validation": summarize_side(valid, total),
            "distinct_users_in_both": len({r.user for r in train} & {r.user for r in valid}),
            "distinct_candidate_news_ids_in_both": len(train_news & valid_news),
        },
        "first_seen_policy": "First observation in impression candidates. Score every impression at a timestamp before updating the pool.",
    }


def self_test():
    news = [b"N1\tc\ts\ttitle\t\turl\t[]\t[]\n"]
    def line(i, hour, candidates, history=""):
        return f"{i}\tU1\t11/12/2019 {hour}:00:00 AM\t{history}\t{candidates}\n".encode()
    fixture = [line(4, 4, "N1-1 N2-0"), line(1, 1, ""), line(5, 4, "N2-0"),
               line(3, 3, "N1-0"), line(2, 2, "N1-1 N1-1", "N2 N2")]
    result = profile(news, fixture, (4, 5))
    split = result["split"]
    assert split["train"]["impressions"] == 3 and split["validation"]["impressions"] == 2
    assert split["zero_based_index"] == 4 and split["impressions_at_boundary"] == 2
    assert split["validation"]["start"] == split["validation"]["end"] == split["boundary_source_time"]
    assert split["train"]["labels"] == dict(zip(GROUPS, [1, 1, 1, 0]))
    assert split["validation"]["labels"] == dict(zip(GROUPS, [0, 0, 1, 1]))
    assert [split[k]["scoring_denominator"] for k in ("train", "validation")] == [0, 1]
    assert split["distinct_users_in_both"] == split["distinct_candidate_news_ids_in_both"] == 1
    imp = result["impressions"]
    assert imp["excess_candidate_tokens"] == imp["rows_with_repeated_candidates"] == 1
    assert imp["missing_candidate_ids"] == imp["missing_history_ids"] == 1
    assert imp["missing_candidate_occurrences"] == imp["missing_history_occurrences"] == 2
    for sample, reason in [(line(5, 13, "N1-1"), "invalid_time"),
                           (line(5, 4, "N1-7"), "invalid_candidate_or_label"),
                           (line(5, 4, "N1 N2-1"), "partially_labeled_impression"),
                           (line(1, 4, "N1-0"), "cross_split_impression_id")]:
        try:
            profile(news, fixture[:2] + [sample] + fixture[3:], (4, 5))
        except ProfileError as exc:
            assert f"behaviors.tsv:3: {reason}" in str(exc), str(exc)
        else:
            raise AssertionError(f"accepted invalid sample: {reason}")
    print("PASS: 5 synthetic impressions; train=3 validation=2 boundary_group=2; denominators=0,1")
    print("PASS: four label groups, overlaps, duplicate/missing counts and line-numbered errors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    sys.path.insert(0, str(ROOT))
    from tools.common import git, head, snapshot, status_paths
    from tools.rv import write_result
    require(args.config == CONFIG and args.output == OUTPUT, "unexpected_config_or_output")
    cfg = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    require(cfg == {"news": "data/train/news.tsv", "behaviors": "data/train/behaviors.tsv",
                    "train_fraction": [4, 5], "boundary_ties": "validation", "output": OUTPUT},
            "unexpected_input_or_split_contract")
    require(not status_paths() and head() == git("rev-parse", "--verify", "refs/tags/card-901^{commit}", check=True).stdout.strip(),
            "requires_clean_card_901_commit")
    require(not (ROOT / OUTPUT).exists(), "result_already_exists")
    snap = snapshot()
    require(snap["DATA"]["version"] == "MIND-small-train-dev-v0.1" and snap["LOCK"]["version"] == "LOCK-v0.1",
            "unexpected_data_or_lock_version")
    with (ROOT / cfg["news"]).open("rb") as news, (ROOT / cfg["behaviors"]).open("rb") as behaviors:
        payload = profile(news, behaviors, cfg["train_fraction"])
    write_result(args.output, payload, kind="RV", version="RV-data-profile-001-r1", config=args.config)


if __name__ == "__main__":
    try:
        main()
    except ProfileError as exc:
        sys.exit(str(exc))
