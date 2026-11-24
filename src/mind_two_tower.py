"""Mean news/history encoders and uncorrected in-batch dot-product loss."""
from collections import Counter
from dataclasses import dataclass
import re

import numpy as np
import torch
from torch import nn


def article_tokens(text, pattern, title_limit, abstract_limit):
    title, abstract = text
    return pattern.findall(title.lower())[:title_limit] + pattern.findall(abstract.lower())[:abstract_limit]


def build_catalog(texts, training_ids, all_ids, cfg):
    """Use lexical IDs for deterministic pool ties; zero is reserved for padding."""
    pattern = re.compile(cfg["token_pattern"])
    ids = sorted(all_ids)
    tokens = {n: article_tokens(texts.get(n, ("", "")), pattern,
                                cfg["title_tokens"], cfg["abstract_tokens"]) for n in ids}
    words = sorted({w for n in training_ids for w in tokens[n]})
    vocabulary = {w: i for i, w in enumerate(words, 1)}
    table = np.zeros((1 + len(ids), cfg["title_tokens"] + cfg["abstract_tokens"]), dtype=np.int32)
    for i, n in enumerate(ids, 1):
        encoded = [vocabulary[w] for w in tokens[n] if w in vocabulary]
        table[i, :len(encoded)] = encoded
    return ids, vocabulary, torch.from_numpy(table)


def history_mean(news_vectors, history_ids):
    mask = history_ids.ne(0)
    return (news_vectors * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)


class MeanTowers(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.words = nn.EmbeddingBag(vocab_size + 1, dim, mode="mean", padding_idx=0)

    def news(self, token_rows):
        return self.words(token_rows)

    def aggregate_history(self, news_vectors, history_ids):
        return history_mean(news_vectors, history_ids)

    def forward(self, token_table, histories, targets, hard_ids=None):
        # Each news article is encoded once per batch; gradients from all uses add.
        parts = [histories.flatten(), targets]
        if hard_ids is not None:
            parts.append(hard_ids.flatten())
        selected, inverse = torch.unique(torch.cat(parts), return_inverse=True)
        vectors = self.news(token_table[selected])
        count = histories.numel()
        hist_vectors = vectors[inverse[:count]].reshape(*histories.shape, -1)
        users = self.aggregate_history(hist_vectors, histories)
        end = count + targets.numel()
        logits = users @ vectors[inverse[count:end]].T
        if hard_ids is None:
            return logits
        negatives = vectors[inverse[end:]].reshape(*hard_ids.shape, -1)
        return logits, (users[:, None] * negatives).sum(dim=-1)


class AttentionTowers(MeanTowers):
    """Use one masked additive attention module in training and both ranking tasks."""

    def __init__(self, vocab_size, dim, hidden_dim):
        super().__init__(vocab_size, dim)
        self.projection = nn.Linear(dim, hidden_dim)
        self.query = nn.Parameter(torch.zeros(hidden_dim))

    def history_weights(self, vectors, history_ids):
        valid = history_ids.ne(0)
        scores = torch.tanh(self.projection(vectors)) @ self.query
        scores = scores.masked_fill(~valid, -torch.inf)
        # Give the all-padding row finite inputs to softmax, then zero its mass.
        scores = torch.where(valid.any(dim=1, keepdim=True), scores, 0.0)
        return scores.softmax(dim=1).masked_fill(~valid, 0.0)

    def aggregate_history(self, news_vectors, history_ids):
        weights = self.history_weights(news_vectors, history_ids)
        return (weights.unsqueeze(-1) * news_vectors).sum(dim=1)


def hard_loss(positive, negative, present, valid):
    """Raw (uncorrected) softmax over the positive and the row's mined articles."""
    active = present.any(dim=1) & valid
    columns = torch.cat((positive[:, None], negative.masked_fill(~present, -torch.inf)), dim=1)
    return (torch.logsumexp(columns, dim=1) - positive)[active].sum()


def click_examples(events, news_index):
    """One (event index, target news) per positive occurrence, including repeats."""
    examples = []
    positives = []
    for i, event in enumerate(events):
        clicks = [news_index[n] for n in event.clicks()]
        positives.append(frozenset(clicks))
        examples.extend((i, n) for n in clicks)
    return np.asarray(examples, dtype=np.int64).reshape(-1, 2), positives


def column_mask(batch, positives):
    """Keep own column. Mask off-diagonal copies and other known positives."""
    events, targets = batch.T
    duplicates = targets[:, None] == targets[None, :]
    np.fill_diagonal(duplicates, False)
    other = np.zeros_like(duplicates)
    for i, (event, target) in enumerate(zip(events, targets)):
        if len(positives[event]) > 1:
            for news in positives[event]:
                if news != target:
                    other[i] |= targets == news
    allowed = ~(duplicates | other)
    stats = Counter(duplicate_target_columns=int(duplicates.sum()),
                    other_positive_columns=int(other.sum()),
                    skipped_rows=int((allowed.sum(axis=1) == 1).sum()))
    return allowed, stats


def target_probabilities(examples, catalog_size):
    """Column marginal before any row-specific mask. Count every click occurrence."""
    if len(examples) == 0:
        raise ValueError("no positive training targets")
    counts = np.bincount(examples[:, 1], minlength=catalog_size)
    return counts, counts / len(examples)


def inbatch_loss(logits, allowed, column_q=None):
    """Return a sum over usable rows for correct epoch-level weighting."""
    if column_q is not None:
        if column_q.shape != (logits.size(1),):
            raise ValueError("expected one probability per column")
        if not torch.all(torch.isfinite(column_q) & (column_q > 0)):
            raise ValueError("column probabilities must be finite and positive")
        logits = logits - column_q.log().to(logits.dtype)
    valid_rows = allowed.sum(dim=1) > 1
    losses = torch.logsumexp(logits.masked_fill(~allowed, -torch.inf), dim=1) - logits.diagonal()
    return losses[valid_rows].sum()


def exact_top(scores, visible, k):
    """Scores are exact dots in lexical news-ID order. Resolve the kth tie explicitly."""
    eligible = np.flatnonzero(visible)
    if len(eligible) > k:
        values = scores[eligible]
        boundary = np.partition(values, len(values) - k)[len(values) - k]
        higher = eligible[values > boundary]
        equal = eligible[values == boundary]
        eligible = np.concatenate((higher, equal[:k - len(higher)]))
    return eligible[np.lexsort((eligible, -scores[eligible]))]


@dataclass
class Visibility:
    first_time: np.ndarray

    @classmethod
    def from_replay(cls, train, valid, news_index):
        # Strict comparison at query time makes all members of a timestamp see
        # the same pool. Unobserved news retain the sentinel and never enter it.
        first = np.full(len(news_index) + 1, np.iinfo(np.int64).max, dtype=np.int64)
        for row in train:
            for n in row.candidates:
                first[news_index[n]] = -1
        for group, row in valid:
            for n in row.candidates:
                j = news_index[n]
                if group < first[j]:
                    first[j] = group
        return cls(first)

    def before(self, group):
        return self.first_time < group
