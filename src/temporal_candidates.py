"""Candidate-pool visibility at the decision time of each impression."""
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import groupby
from operator import attrgetter
from typing import TypeVar


@dataclass(frozen=True)
class Exposure:
    impression_id: str
    timestamp: datetime
    candidates: tuple[str, ...]


E = TypeVar("E", bound=Exposure)


def score_by_timestamp(exposures: Iterable[E], score: Callable[[E, frozenset[str]], None],
                       initial_pool: Iterable[str] = (), *,
                       after_group: Callable[[Sequence[E]], None] | None = None) -> None:
    """Score the complete group on one snapshot, then publish its observations."""
    observed = set(initial_pool)
    timestamp = attrgetter("timestamp")
    for _, group in groupby(sorted(exposures, key=timestamp), key=timestamp):
        batch = tuple(group)
        visible = frozenset(observed)
        for exposure in batch:
            score(exposure, visible)
        for exposure in batch:
            observed.update(exposure.candidates)
        if after_group is not None:
            after_group(batch)
