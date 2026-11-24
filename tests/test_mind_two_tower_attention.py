"""Synthetic controls for the shared training and exact-replay history encoder."""
from datetime import datetime
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mind_two_tower import AttentionTowers, MeanTowers, history_mean
from run_mind_two_tower import Event, paired_deltas, replay, verify_groups, verify_mean_metrics


class AttentionTests(unittest.TestCase):
    def test_initial_word_identity_mask_and_empty_backward(self):
        torch.manual_seed(21)
        control = MeanTowers(8, 4)
        torch.manual_seed(21)
        model = AttentionTowers(8, 4, 32)
        torch.testing.assert_close(model.words.weight, control.words.weight, rtol=0, atol=0)
        features = torch.randn(3, 4, 4, requires_grad=True)
        ids = torch.tensor([[0, 0, 0, 0], [0, 2, 0, 0], [2, 0, 3, 1]])
        weights = model.history_weights(features, ids)
        self.assertTrue((weights[ids == 0] == 0).all())
        self.assertEqual(weights[1, 1].item(), 1)
        users = model.aggregate_history(features, ids)
        torch.testing.assert_close(users, history_mean(features, ids))
        self.assertTrue((users[0] == 0).all())
        users.sum().backward()
        self.assertTrue((features.grad[ids == 0] == 0).all())
        for parameter in (model.query, model.projection.weight, model.projection.bias):
            self.assertTrue(parameter.grad.isfinite().all())
        self.assertGreater(model.query.grad.norm().item(), 0)

    def test_forward_and_exact_replay_share_learned_attention(self):
        model = AttentionTowers(2, 1, 32)
        with torch.no_grad():
            model.words.weight.copy_(torch.tensor([[0.], [-1.], [1.]]))
            model.projection.weight.fill_(1.)
            model.projection.bias.zero_()
        table = torch.arange(3).view(-1, 1)
        ids = {"N1": 1, "N2": 2}
        t = datetime(2019, 11, 11)
        row = Event("2", t, tuple(ids), bytes([0, 1]), tuple(ids))
        cfg = {"evaluation_batch": 4, "history_limit": 20, "top_k": 1}

        def measure():
            forward = model(table, torch.tensor([[1, 2]]), torch.tensor([1, 2]))
            report = replay(model, table, [row], [row], ids, cfg, 1)["full"]
            return forward, report["official"]["metrics"]["nDCG@10"], report["retrieval"]["hits"]

        uniform, before, hits = measure()
        self.assertEqual(uniform[0, 0].item(), uniform[0, 1].item())
        self.assertEqual(hits, 0)
        with torch.no_grad():
            model.query.fill_(.2)
        focused, after, hits = measure()
        self.assertGreater(focused[0, 1].item(), focused[0, 0].item())
        self.assertGreater(after, before)
        self.assertEqual(hits, 1)
        focused[0, 1].backward()
        self.assertGreater(model.projection.weight.grad.norm().item(), 0)
        print(f"official nDCG@10 {before:.9f} -> {after:.9f}, exact Top-1 hits 0 -> 1")

    def test_length_groups_use_untruncated_history_and_sum(self):
        t = datetime(2019, 11, 11)
        ids = {"N1": 1, "N2": 2}
        rows = [Event(str(i), t, tuple(ids), bytes([1, 0]), ("N2",) * n)
                for i, n in enumerate((0, 1, 19, 20, 21))]
        model = MeanTowers(2, 4)
        cfg = {"evaluation_batch": 2, "history_limit": 2, "top_k": 1}
        table = torch.arange(3).view(-1, 1)
        metrics = replay(model, table, rows[:1], rows, ids, cfg, 2, lengths=True)
        verify_groups(metrics, {"empty_history": [1]*3, "short_history": [2]*3, "long_history": [2]*3})
        legacy = replay(model, table, rows[:1], rows, ids, cfg, 2)
        verify_mean_metrics(metrics, legacy)
        self.assertTrue(all(v == 0 for delta in paired_deltas(metrics, metrics).values() for v in delta.values()))
        legacy["full"]["retrieval"]["hits"] += 1
        with self.assertRaises(ValueError):
            verify_mean_metrics(metrics, legacy)


if __name__ == "__main__":
    unittest.main()
