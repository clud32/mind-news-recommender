import json
from datetime import datetime, timedelta
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
import unittest

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from mind_din import DIN, pool_scores
from mind_evaluation import Impression, METRICS, evaluate
from mind_two_tower import MeanTowers
from run_mind_baselines import TitleIndex
from run_mind_two_tower import Event
import run_mind_final_eval as final
from src.judge.mind_final import conclusions, verify
from src.present.mind_final import Metric, load_sources, plan, render_tex
from tools.common import payload_sha256


def event(iid, timestamp, candidates, labels, history=()):
    return Event(iid, timestamp, tuple(candidates), bytes(labels), history)


class FinalEvaluationTests(unittest.TestCase):
    def test_combined_replay_and_fixed_membership(self):
        names = ["early", "future", "same", "seen_after"]
        titles = {name: "headline matching " + name for name in names}
        settings = {"policy": {"window_hours": 24, "top_k": 100, "score_batch": 4},
                    "bootstrap": {"seed": 1, "replicates": 64}}
        t = datetime(2019, 11, 14)
        train = [
            (event("y", t - timedelta(hours=24), ["early"], [1]), "train"),
            (event("same", t, ["same"], [1]), "train"),
            (event("f", t + timedelta(seconds=2), ["future"], [1]), "train")]
        # Impression IDs may be shared across train and dev. Stable timestamp order matters.
        dev = [
            (event("same", t, ["same", "early"], [1, 0]), "new"),
            (event("both", t, ["early", "seen_after"], [1, 0], ("early",)), "other"),
            (event("end", t + timedelta(seconds=1), ["same", "future"], [1, 0]), "other")]
        title_index = TitleIndex(titles, {"early"}, {"norm": "l2"})
        self.assertNotIn("future", title_index.encoder.vocabulary_)
        index = {name: i for i, name in enumerate(names, 1)}
        torch.manual_seed(2)
        towers = [MeanTowers(3, 2) for _ in range(3)]
        banks = [torch.arange(10, dtype=torch.float32).reshape(5, 2) for _ in range(3)]
        dins = [DIN(2, (4, 2), 4) for _ in range(3)]
        replay = final.Replay(settings, {"history_limit": 2}, title_index, names,
                              (index, towers, banks, dins), torch.device("cpu"))
        result = []
        for _, observed, scored in final.timestamp_replay(train, dev, replay.score):
            result.extend(scored)
            if scored and scored[0]["id"] == "same":
                self.assertEqual(replay.window.counts, {"early": 1})
            replay.publish(observed)
        a, b, c = result
        self.assertEqual([r["fixed"] for r in result], [["early"], ["early"], ["early", "same", "seen_after"]])
        self.assertEqual(a["recall_destination"], [1, 0, 0])
        self.assertEqual(a["rerank"]["din_s0"], (None, 0.))
        self.assertEqual(b["recall_destination"], [0, 0, 1])
        self.assertEqual(c["recall_destination"], [0, 0, 1])
        for r in result:
            for name in ("din_s0", "din_s1", "din_s2"):
                self.assertEqual(set(r["orders"][name]), set(r["fixed"]))
        _, official, retrieval, ranking, destinations, bootstrap = final.aggregate(result, settings)
        self.assertEqual(official["tower_s0"]["scoring_denominator"], 3)
        self.assertEqual(retrieval["tower_s0"]["clicks"], 3)
        self.assertEqual(retrieval["tower_s0"]["hits"], 2)
        self.assertEqual(ranking["din_s0"]["end_denominator"], 3)
        self.assertEqual(ranking["din_s0"]["pool_denominator"], 2)
        self.assertEqual(destinations, {"outside": 1, "missed": 0, "recalled": 2})
        self.assertEqual(bootstrap["end_din_minus_tower_s0"]["users"], 2)

    def test_reference_ranks_and_zero_hits(self):
        cfg = {"scorer": {"path": str(Path.cwd() / "资料/MIND-官方评测.py"), "tolerance": 1e-8}}
        rows = [Impression((1, 0, 1, 0, 0, 0), [4, 1, 2, 6, 3, 5]),
                Impression([1, 1], [2, 1]), Impression([], []), Impression([0, 0], [2, 1])]
        result = final.verify_microsoft(cfg, rows)
        self.assertEqual(result["denominator"], 1)
        self.assertEqual(result["metrics"]["MRR"], 0.375)
        self.assertEqual(list(result["metrics"]), list(METRICS))
        self.assertEqual(evaluate(rows)["excluded"], {"empty_labels": 1, "all_positive": 1, "all_negative": 1})
        self.assertEqual(pool_scores(["x"], ["a", "b"]), (None, 0.))

    def test_news_conflict_stops_with_identity_and_evidence(self):
        with TemporaryDirectory() as temp:
            train, dev, evidence = [Path(temp) / n for n in ("train.tsv", "dev.tsv", "conflicts.json")]
            def row(name, title):
                return "\t".join((name, "cat", "subcat", title, "abstract", "url", "[]", "[]")) + "\n"
            train.write_text(row("N1", "original"), encoding="utf8")
            dev.write_text(row("N1", "original") + row("N2", "new"), encoding="utf8")
            _, counts, sources = final.merge_news(train, dev, evidence)
            self.assertEqual(counts, {"train": 1, "dev": 1, "dev_identical_reuse": 1})
            self.assertEqual(sources["N1"]["split"], "train")
            dev.write_text(row("N1", "changed"), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "news text conflicts"):
                final.merge_news(train, dev, evidence)
            self.assertEqual(json.loads(evidence.read_bytes())[0]["news_id"], "N1")

    def test_registered_sources_metric_differences_and_report_references(self):
        metrics, sections = plan()
        specs = [field for metric in metrics for field in metric.fields]
        docs = {obj: {"_meta": {"version": obj, "dirty": False, "kind": "RV"}, "payload": {}}
                for obj, *_ in specs}
        for obj, *path in specs:
            target = docs[obj]["payload"]
            for segment in path[:-1]:
                target = target.setdefault(segment, {})
            target[path[-1]] = 0.25
        with TemporaryDirectory() as temp:
            registry = {}
            for obj, doc in docs.items():
                digest = payload_sha256(doc["payload"])
                doc["_meta"]["payload_sha256"] = digest
                registry[obj] = {"version": obj, "identity": digest}
                (Path(temp) / f"{obj}.json").write_text(json.dumps(doc), encoding="utf8")
            verified = load_sources(docs, registry, Path(temp))
            bad = next(iter(registry))
            registry[bad]["identity"] = "invalid"
            with self.assertRaisesRegex(ValueError, "payload identity mismatch"):
                load_sources(docs, registry, Path(temp))
        iv = [metric.calculate(verified) for metric in metrics]
        cv = conclusions(iv)
        verify(iv, cv)
        tex = render_tex(sections)
        self.assertEqual(set(re.findall(r"\\iv\{([^}]+)\}", tex)), {entry["id"] for entry in iv})
        self.assertEqual(set(re.findall(r"\\cv\{([^}]+)\}", tex)), {entry["id"] for entry in cv})
        self.assertIn("higher than 24-hour popularity", cv[0]["statement_en"])
        row = Metric("I-pair", (specs[0], specs[0]), "fraction", "same exposure", "difference").calculate(verified)
        self.assertEqual(row["value"], 0.)
        self.assertEqual(row["sources"][0], row["sources"][1])
        self.assertEqual(row["sources"][0]["field"], list(specs[0][1:]))

    def test_cluster_score_uses_exposure_weights(self):
        result = final.paired_bootstrap(["u", "u", "v"], [1, 1, -1], [1, 1, 1], 128, 5)
        self.assertAlmostEqual(result["estimate"], 1/3)
        self.assertEqual(result["denominator"], 3)
        self.assertEqual(result["users"], 2)
        self.assertEqual((result["low"], result["high"]), (-1., 1.))


if __name__ == "__main__":
    unittest.main()
