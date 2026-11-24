"""Candidate-conditioned DIN reranker on frozen news embeddings."""
from math import fsum, log2

import torch
from torch import nn
from torch.nn import functional as F

from mind_evaluation import ranks_from_scores


class DIN(nn.Module):
    def __init__(self, dim, widths=(64, 32), head=64):
        super().__init__()
        self.local = nn.Sequential(nn.Linear(4 * dim, widths[0]), nn.ReLU(),
                                   nn.Linear(widths[0], widths[1]), nn.ReLU(),
                                   nn.Linear(widths[1], 1), nn.Sigmoid())
        self.rank = nn.Sequential(nn.Linear(3 * dim, head), nn.ReLU(), nn.Linear(head, 1))

    def weights(self, candidate, history, mask):
        query = candidate[:, None].expand_as(history)
        features = torch.cat((query, history, query - history, query * history), dim=-1)
        return self.local(features).squeeze(-1).masked_fill(~mask, 0.)

    def forward(self, candidate, history, mask):
        weights = self.weights(candidate, history, mask)
        interest = (weights.unsqueeze(-1) * history).sum(dim=1)
        return self.rank(torch.cat((candidate, interest, candidate * interest), dim=-1)).squeeze(-1)


def impression_loss(logits, labels, lengths):
    per_candidate = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (per_candidate / torch.repeat_interleave(lengths, lengths, output_size=len(logits))).sum() / len(lengths)


def fixed_order(names, scores, empty_history):
    if empty_history:
        return list(names)
    ranks = ranks_from_scores(names, scores)
    return [news for _, news in sorted(zip(ranks, names))]


def pool_scores(ordered, clicked):
    """One gain per selected news. End-to-end ideal includes every original positive."""
    positive = set(clicked)
    ranks = [i for i, n in enumerate(ordered, 1) if n in positive]
    gain = fsum(1/log2(1+i) for i in ranks if i <= 10)
    def ideal(n):
        return fsum(1/log2(1+i) for i in range(1, min(n, 10)+1))
    return (gain/ideal(len(ranks)) if ranks else None, gain/ideal(len(clicked)) if clicked else 0.)
