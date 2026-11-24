from datetime import datetime, timedelta
from pathlib import Path
import json
import math
import sys
import unittest

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
from mind_two_tower import (MeanTowers, Visibility, build_catalog, click_examples, column_mask,
                            exact_top, history_mean, inbatch_loss)
from run_mind_two_tower import Event, histories, replay, train_epochs, weight_identity

CFG = json.loads((BASE / "configs/mind-two-tower-v0.1.json").read_bytes())
T = datetime(2019, 11, 13)


class TowerTests(unittest.TestCase):
    def test_aggregation_and_vocabulary(self):
        texts = {"N1": ("alpha beta ignored", "beta stop"), "N2": ("heldout", "alpha"),
                 "N3": ("", "history")}
        cfg = dict(CFG, title_tokens=2, abstract_tokens=1)
        ids, vocab, table = build_catalog(texts, {"N1", "N3"}, texts, cfg)
        self.assertEqual(ids, ["N1", "N2", "N3"])
        self.assertEqual(set(vocab), {"alpha", "beta", "history"})
        model = MeanTowers(len(vocab), 2)
        with torch.no_grad():
            model.words.weight.copy_(torch.tensor([[0.,0.], [3.,0.], [0.,3.], [6.,3.]]))
        news = model.news(table)
        torch.testing.assert_close(news, torch.tensor([[0.,0.],[1.,2.],[3.,0.],[6.,3.]]))
        h = torch.tensor([[0, 0, 0], [1, 0, 2]])
        torch.testing.assert_close(history_mean(news[h], h), torch.tensor([[0.,0.],[2.,1.]]))
        evt = Event("x", T, (), b"", tuple(["N3"] + ["N1"]*19 + ["N2"]))
        self.assertEqual(histories([evt], dict(zip(ids, range(1,4))), 20).tolist(), [[1]*19 + [2]])

    def test_negative_columns_and_loss(self):
        # Row 0: own A, foreign A, another positive B from the same impression, negative C.
        batch = np.array([[0,1],[1,1],[0,2],[2,3]])
        mask, counts = column_mask(batch, [{1,2},{1},{3}])
        self.assertEqual(mask[0].tolist(), [True,False,False,True])
        self.assertEqual(mask[2].tolist(), [False,False,True,True])
        self.assertEqual(mask[3].tolist(), [True,True,True,True])  # duplicate negative A kept twice
        self.assertEqual((counts["duplicate_target_columns"], counts["other_positive_columns"]), (2,3))
        logits = torch.tensor([[2.,9.,8.,1.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]], dtype=torch.float64)
        # Other rows get only their diagonal, so this tests the production reduction on exactly row 0.
        allowed = np.eye(4, dtype=bool)
        allowed[0] = mask[0]
        expected = math.log1p(math.exp(-1))
        actual = inbatch_loss(logits, torch.from_numpy(allowed)).item()
        self.assertAlmostEqual(actual, expected, places=14)
        print("row=0 targets=A,A,B,C positives=A,B logits=2,9,8,1 keep=1,0,0,1",
              "expected", expected, "actual", actual)
        mask, counts = column_mask(np.array([[0,1],[0,2],[1,1]]), [{1,2},{1}])
        self.assertEqual(counts["skipped_rows"], 2)
        self.assertTrue(mask[2,1])
        self.assertEqual(inbatch_loss(logits, torch.eye(4,dtype=torch.bool)).item(), 0)

    def test_exact_top_and_boundary_ties(self):
        rng = np.random.default_rng(7)
        # Row indexes correspond to lexicographically sorted news IDs.
        names = sorted(f"N{i}" for i in range(145))
        bank = rng.integers(-1, 2, size=(len(names), 4))
        for q in (np.zeros(4, dtype=int), np.array([1,0,1,-1])):
            dots = bank @ q
            for limit in (0, 17, 145):
                visible = np.arange(len(names)) < limit
                brute = sorted(range(limit), key=lambda i: (-sum(q[j]*bank[i,j] for j in range(4)), names[i]))
                self.assertEqual(exact_top(dots, visible, 100).tolist(), brute[:100])

    def test_train_and_temporal_replay(self):
        idx = {"N0":1, "N1":2, "N2":3, "N3":4}
        train = [Event("x", T, ("N0","N1","N1"), bytes([0,1,1]), ("N0",)),
                 Event("y", T, ("N0","N1"), bytes([1,0]), ("N1",))]
        examples, positives = click_examples(train, idx)
        self.assertEqual(examples.tolist(), [[0,2],[0,2],[1,1]])
        self.assertEqual(positives, [{2},{1}])
        torch.manual_seed(0)
        model = MeanTowers(4, 4)
        table = torch.tensor([[0,0],[1,0],[2,0],[3,0],[4,0]])
        initial = weight_identity(model)
        report, _ = train_epochs(model, table, train, idx, dict(CFG, epochs=1), "cpu", 0)
        self.assertNotEqual(initial, weight_identity(model))
        self.assertEqual(report["epochs"][0]["valid_rows"], 3)
        valid = [Event("a", T+timedelta(seconds=1), ("N2","N1"), bytes([1,0]), ()),
                 Event("b", T+timedelta(seconds=1), ("N3","N0"), bytes([1,0]), ("N2",)),
                 Event("c", T+timedelta(seconds=2), ("N2","N3"), bytes([0,1]), ("N1",))]
        for ordered in (valid, [valid[1],valid[0],valid[2]]):
            v = Visibility.from_replay(train, [(0,ordered[0]),(0,ordered[1]),(1,ordered[2])], idx)
            self.assertEqual(np.flatnonzero(v.before(0)).tolist(), [1,2])
            self.assertEqual(np.flatnonzero(v.before(1)).tolist(), [1,2,3,4])
            result = replay(model, table, train, ordered, idx, CFG, 2)
            self.assertEqual(result["full"]["official"]["scoring_denominator"], 3)
            self.assertEqual(result["full"]["retrieval"]["covered"], 1)
            self.assertEqual(result["full"]["retrieval"]["hits"], 1)
            self.assertEqual(result["prefix"]["retrieval"]["covered"], 0)
            self.assertEqual(result["empty_history"]["official"]["metrics"]["MRR"], .5)
            self.assertEqual(result["with_history"]["official"]["total"], 2)


if __name__ == "__main__":
    unittest.main()
