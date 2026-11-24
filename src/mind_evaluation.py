"""Offline MIND metrics on the original impression positions, with equal impression weights."""
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import fsum, isfinite, log2
from numbers import Integral

METRICS = ("AUC", "MRR", "nDCG@5", "nDCG@10")


def ranks_from_scores(news_ids: Sequence[str], scores: Sequence[float]) -> list[int]:
    """Return a 1-based rank for each candidate in input order."""
    if len(news_ids) != len(scores):
        raise ValueError("candidate and score counts differ")
    if not all(isfinite(score) for score in scores):
        raise ValueError("candidate scores must be finite")
    if not all(isinstance(news, str) for news in news_ids):
        raise ValueError("news IDs must be strings")
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), news_ids[i], i))
    ranks = [0] * len(order)
    for rank, position in enumerate(order, 1):
        ranks[position] = rank
    return ranks


@dataclass(frozen=True)
class Impression:
    labels: Sequence[int]
    ranks: Sequence[int]


def _ndcg(positive_ranks: list[int], k: int) -> float:
    dcg = fsum(1 / log2(1 + rank) for rank in positive_ranks if rank <= k)
    ideal = fsum(1 / log2(1 + rank) for rank in range(1, min(len(positive_ranks), k) + 1))
    return dcg / ideal


def _measure(labels, ranks):
    # Give each positive credit for each negative below it. The other
    # positives below it do not belong in the pairwise AUC denominator.
    positive = sorted(rank for label, rank in zip(labels, ranks) if label == 1)
    size = len(labels)
    positives = len(positive)
    negatives = size - positives
    wins = sum(size - rank - (positives - index - 1) for index, rank in enumerate(positive))
    return {
        "AUC": wins / (positives * negatives),
        "MRR": fsum(1 / rank for rank in positive) / positives,
        "nDCG@5": _ndcg(positive, 5),
        "nDCG@10": _ndcg(positive, 10),
    }


def evaluate(impressions: Iterable[Impression]) -> dict:
    """Score valid impressions and report the number excluded for each reason."""
    counts = Counter()
    sums = dict.fromkeys(METRICS, 0.0)
    total = 0
    for item in impressions:
        total += 1
        labels, ranks = item.labels, item.ranks
        if len(labels) == 0:
            counts["empty_labels"] += 1
            continue
        size = len(labels)
        if not all(label in (0, 1) for label in labels):
            raise ValueError("expected binary labels")
        if (len(ranks) != size or not all(isinstance(r, Integral) for r in ranks)
                or set(ranks) != set(range(1, size + 1))):
            raise ValueError("expected a permutation of 1-based ranks in original positions")
        positives = sum(int(label) for label in labels)
        if positives == size:
            counts["all_positive"] += 1
        elif positives == 0:
            counts["all_negative"] += 1
        else:
            counts["valid"] += 1
            for key, value in _measure(labels, ranks).items():
                sums[key] += value

    denominator = counts["valid"]
    return {
        "total": total,
        "scoring_denominator": denominator,
        "excluded": {key: counts[key] for key in ("empty_labels", "all_positive", "all_negative")},
        "metrics": {key: value / denominator for key, value in sums.items()} if denominator else None,
        "status": "已评分" if denominator else "无可评分曝光",
    }
