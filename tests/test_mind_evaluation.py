import csv
from datetime import datetime, timedelta
import hashlib
import importlib.util
from io import StringIO
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mind_evaluation import Impression, METRICS, evaluate, ranks_from_scores
from temporal_candidates import Exposure, score_by_timestamp

TOLERANCE = 1e-8
# Deliberately store candidates out of rank order. Scores determine ranks.
CASES = {
    "A": ((1, 0, 1, 0, 0, 0), (4, 1, 2, 6, 3, 5),
          (0.625, 0.375, 0.6509209298071326, 0.6509209298071326)),
    "B": ((0, 1, 0, 0, 1, 0, 0), (3, 7, 1, 5, 2, 4, 6),
          (0.4, 0.3214285714285714, 0.38685280723454163, 0.5912352048230277)),
}


def make_impression(name):
    labels, expected_ranks, _ = CASES[name]
    news = [f"N{i}" for i in range(len(labels))]
    scores = [len(labels) - rank for rank in expected_ranks]
    return Impression(labels, ranks_from_scores(news, scores))


def load_scorer():
    sources = Path.cwd() / "资料"
    path = sources / "MIND-官方评测.py"
    with (sources / "来源与校验.tsv").open(encoding="utf-8-sig", newline="") as f:
        row, = [r for r in csv.DictReader(f, delimiter="\t") if r["文件"] == path.name]
    contents = path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    if digest != row["SHA256"] or len(contents) != int(row["字节数"]):
        raise ValueError("Microsoft scorer identity differs from source inventory")
    print(f"Microsoft scorer SHA-256 {digest}", flush=True)
    spec = importlib.util.spec_from_file_location("mind_reference_scorer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.scoring


def reference_metrics(scoring, rows):
    truth, submission = StringIO(), StringIO()
    for i, item in enumerate(rows, 1):
        truth.write(f"{i} {json.dumps(list(item.labels), separators=(',', ':'))}\n")
        submission.write(f"{i} {json.dumps(item.ranks, separators=(',', ':'))}\n")
    truth.seek(0)
    submission.seek(0)
    return scoring(truth, submission)


class EvaluationTests(unittest.TestCase):
    def test_hand_and_reference(self):
        scoring = load_scorer()
        a, b = make_impression("A"), make_impression("B")
        self.assertEqual(a.ranks, list(CASES["A"][1]))
        self.assertEqual(b.ranks, list(CASES["B"][1]))
        mean = [(x + y) / 2 for x, y in zip(CASES["A"][2], CASES["B"][2])]
        print("sample metric hand project microsoft abs(project-hand) abs(project-microsoft)")
        for name, rows, expected in (("A", [a], CASES["A"][2]),
                                     ("B", [b], CASES["B"][2]), ("A+B", [a, b], mean)):
            result = evaluate(rows)
            self.assertEqual(result["scoring_denominator"], len(rows))
            reference = reference_metrics(scoring, rows)
            for metric, hand, official in zip(METRICS, expected, reference, strict=True):
                with self.subTest(sample=name, metric=metric):
                    actual = result["metrics"][metric]
                    e_hand, e_ref = abs(actual - hand), abs(actual - official)
                    print(name, metric, repr(hand), repr(actual), repr(float(official)),
                          repr(e_hand), repr(float(e_ref)))
                    self.assertLessEqual(e_hand, TOLERANCE)
                    self.assertLessEqual(abs(official - hand), TOLERANCE)
                    self.assertLessEqual(e_ref, TOLERANCE)
        for metric in METRICS:
            average = (evaluate([a])["metrics"][metric] + evaluate([b])["metrics"][metric]) / 2
            self.assertEqual(evaluate([a, b])["metrics"][metric], average)

    def test_ties_and_original_positions(self):
        ranks = ranks_from_scores(["N2", "N10", "N10"], [0.5] * 3)
        self.assertEqual(ranks, [3, 1, 2])
        print("tie ranks in original positions:", ranks)

    def test_denominator(self):
        valid = [make_impression("A"), make_impression("B")]
        invalid = [Impression([], [2, 1]), Impression([1, 1], [2, 1]), Impression([0], [1])]
        result = evaluate(valid + invalid)
        self.assertEqual(result["total"], 5)
        self.assertEqual(result["scoring_denominator"], 2)
        self.assertEqual(result["excluded"], {"empty_labels": 1, "all_positive": 1, "all_negative": 1})
        self.assertEqual(result["metrics"], evaluate(valid)["metrics"])
        empty = evaluate(invalid)
        self.assertEqual(empty["scoring_denominator"], 0)
        self.assertIsNone(empty["metrics"])
        self.assertEqual(empty["status"], "无可评分曝光")
        print("denominator:", result["total"], result["scoring_denominator"], result["excluded"])
        print("zero denominator:", empty["status"])

    def test_time_group(self):
        start = datetime(2019, 11, 10)
        t1, t2 = start + timedelta(seconds=1), start + timedelta(seconds=2)
        early = Exposure("prior", start, ("N0",))
        a, b = Exposure("a", t1, ("N1",)), Exposure("b", t1, ("N2",))
        later = Exposure("later", t2, ("N3",))
        for order in [(a, b), (b, a)]:
            seen = {}
            def record(row, pool):
                self.assertNotIn(row.impression_id, seen)
                seen[row.impression_id] = pool
            score_by_timestamp([later, *order, early], record)
            self.assertEqual(list(seen), ["prior", order[0].impression_id, order[1].impression_id, "later"])
            self.assertEqual(seen, {"prior": set(), "a": {"N0"}, "b": {"N0"}, "later": {"N0", "N1", "N2"}})
            for key, pool in seen.items():
                print(f"order={order[0].impression_id},{order[1].impression_id} query={key} pool={sorted(pool)}")


if __name__ == "__main__":
    unittest.main()
