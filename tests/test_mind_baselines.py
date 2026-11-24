from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_mind_baselines import (ClickWindow, Event, Retrieval, TitleIndex,
                                read_events, run_sample, split_prefix)
from temporal_candidates import score_by_timestamp

CFG = json.loads((ROOT / "configs/mind-baselines-smoke-v0.1.json").read_bytes())
T = datetime(2019, 11, 13, 12)
DAY = timedelta(days=1)
SECOND = timedelta(seconds=1)


def event(i, t, ids, labels, last=None):
    return Event(str(i), t, tuple(ids), bytes(labels), last)


class BaselineTests(unittest.TestCase):
    def test_window_and_visibility(self):
        old = event(1, T-DAY-SECOND, ["old"], [1])
        edge = event(2, T-DAY, ["edge"], [1])
        first = event(3, T, ["a"], [1])
        second = event(4, T, ["b"], [1])
        later = event(5, T+SECOND, ["c"], [0])
        for order in ((first, second), (second, first)):
            window = ClickWindow(24)
            window.update([old, edge])
            seen = {}

            def record(row, pool):
                window.expire(row.timestamp)
                seen[row.impression_id] = (set(pool), dict(window.counts))

            score_by_timestamp([later, *order], record, {"edge", "old"}, after_group=window.update)
            for row in order:
                self.assertEqual(seen[row.impression_id], ({"edge", "old"}, {"edge": 1}))
            self.assertEqual(seen["5"], ({"edge", "old", "a", "b"}, {"a": 1, "b": 1}))

    def test_title_fit_and_fallback(self):
        titles = {"N1": "red apple", "N2": "blue apple", "N3": "apple unseenword",
                  "N4": " ", "N5": "unseenword"}
        index = TitleIndex(titles, {"N1", "N2"}, CFG["tfidf"])
        self.assertEqual(set(index.encoder.vocabulary_), {"red", "blue", "apple"})
        self.assertAlmostEqual(index.encoder.idf_[index.encoder.vocabulary_["apple"]], 1)
        for last, reason in [(None, "empty_history"), ("absent", "missing_title"),
                             ("N4", "empty_title"), ("N5", "zero_vector")]:
            self.assertEqual(index.query(last), (None, reason))
        sims, reason = index.query("N3")
        self.assertIsNone(reason)
        self.assertGreater(index.scores(sims, ["N1"])[0], 0)
        self.assertEqual(index.scores(sims, ["absent", "N4"]), [0, 0])

    def test_end_to_end_fallback_and_ranking(self):
        train = [event(1, T-DAY, ["N1", "N2"], [0, 1])]
        titles = {"N1": "apple", "N2": "pear", "N3": "", "N4": "unknown"}
        sample = [event(i, T, ["N1", "N2"], [0, 1], last)
                  for i, last in enumerate((None, "absent", "N3", "N4"), 2)]
        result = run_sample(train, sample, titles, dict(CFG, top_k=1))
        self.assertEqual(set(result["fallback_counts"].values()), {1})
        pop, recent = result["models"].values()
        self.assertEqual(pop, recent)
        self.assertEqual(pop["official"]["metrics"]["nDCG@10"], 1)
        self.assertEqual(pop["retrieval"]["hits"], 4)
        changed = run_sample(train, [event(6, T, ["N1", "N2"], [1, 0], "N1")],
                             titles, dict(CFG, top_k=1))["models"]
        self.assertEqual(changed["recent_click"]["official"]["metrics"]["AUC"], 1)
        self.assertEqual(changed["popularity"]["official"]["metrics"]["AUC"], 0)
        self.assertEqual(changed["recent_click"]["retrieval"]["hits"], 1)

    def test_recall_denominators(self):
        rec = Retrieval()
        self.assertIsNone(rec.report()["recall"])
        self.assertIsNone(rec.report()["coverage"])
        rec.add(["unseen"], {"a", "b"}, {"a"})
        self.assertIsNone(rec.report()["conditional_recall"])
        rec.add(["a", "a", "b"], {"a", "b"}, {"a"})
        self.assertEqual(rec.report(), {"clicks": 4, "covered": 3, "hits": 2,
                         "recall": .5, "coverage": .75, "conditional_recall": 2/3})

    def test_split_prefix_preserves_whole_groups_and_history_order(self):
        def line(i, second):
            return f"{i}\tU1\t11/13/2019 12:00:{second:02d} PM\tN8 N2\tN9-0 N2-1\n".encode()
        rows = read_events([line(i, t) for i, t in enumerate([2, 1, 5, 4, 3, 6, 7, 8, 8, 9], 1)])
        self.assertEqual(rows[0].last_news, "N2")
        split = {"zero_based_index": 8, "boundary_source_time": "11/13/2019 12:00:08 PM",
                 "train": {"impressions": 7}, "validation": {"impressions": 3}}
        train, sample = split_prefix(rows, split, dict(CFG, train_count=7, validation_count=3,
                                     boundary="2019-11-13 12:00:08", sample_minimum=1))
        self.assertLess(train[-1].timestamp, sample[0].timestamp)
        self.assertEqual([r.impression_id for r in sample], ["8", "9"])
        self.assertEqual(list(sample[0].clicks()), ["N2"])


if __name__ == "__main__":
    unittest.main()
