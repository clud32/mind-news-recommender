"""Artificial vectors exercise the same index, rank, coverage and timing paths as the run."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import faiss
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
from run_mind_faiss import METHODS, ann_ids, build, exact, measure, quality, summarize


class FaissTests(unittest.TestCase):
    def setUp(self):
        faiss.omp_set_num_threads(1)

    def test_full_cutoff_tie_lexical_mapping_and_roundtrip(self):
        corpus = sorted(f"N{i}" for i in range(300))
        news = np.ones((300, 2), dtype="float32")
        news[0] = [2, 10]
        news[1] = [-1, 90]
        q = np.array([[1, 0]], dtype="float32")
        idx = faiss.IndexFlatIP(2)
        idx.add(news)
        expected = sorted(range(len(news)), key=lambda i: (-float(news[i] @ q[0]), corpus[i]))[:100]
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "index.faiss")
            faiss.write_index(idx, path)
            for candidate in (idx, faiss.read_index(path)):
                ids, ties, crossed = exact(candidate, q, 100)
                self.assertEqual(ids.tolist(), expected)
                self.assertEqual([corpus[i] for i in ids], [corpus[i] for i in expected])
                self.assertEqual(ties, 298)
                self.assertTrue(crossed)
                scores, raw = candidate.search(q, 1)
                self.assertEqual((scores.item(), raw.item()), (2, 0))

    def test_zero_vectors_missing_slots_and_click_denominators(self):
        corpus = ["N1", "N10", "N2", "N3"]
        news = np.array([[1], [2], [3], [4]], dtype="float32")
        idx = faiss.IndexFlatIP(1)
        idx.add(news)
        class Partial:
            def search(self, q, k):
                if not np.any(q):
                    raise AssertionError("zero query reached ANN search")
                return None, np.array([[3, -1]])
        qs = np.array([[0], [1]], dtype="float32")
        meta = [{"id":"q0","clicks":["N1","N2","new"]},
                {"id":"q1","clicks":["N3","N3","N10"]}]
        items = quality({"Flat":idx, "HNSWFlat":Partial()}, qs, corpus, meta, 2)
        self.assertEqual([r["id"] for r in items], ["q0","q1"])
        s = summarize(items,"HNSWFlat",2)
        self.assertEqual((s["clicks"],s["covered"],s["hits"],s["missing"]), (6,5,3,1))
        self.assertEqual((s["recall"],s["coverage"],s["conditional_recall"]), (.5,5/6,3/5))
        self.assertEqual(s["overlap_all"], {"queries":1,"intersection":1,"overlap":.5})
        self.assertEqual(s["overlap_without_ties"],s["overlap_all"])
        self.assertIsNone(summarize(items[:1],"Flat",2)["overlap_all"]["overlap"])
        for invalid in ([2,2],[-2,1],[4,-1]):
            with self.assertRaises(ValueError):
                ann_ids(np.array(invalid),len(corpus))

    def test_build_read_search_and_timing(self):
        rng = np.random.default_rng(41)
        news = rng.integers(-4,5,size=(512,8)).astype("float32")
        qs = np.ascontiguousarray(news[:7])
        cfg = {"hnsw":{"M":8,"efConstruction":40,"efSearch":20},
               "ivfpq":{"nlist":8,"m":2,"nbits":3,"nprobe":4}}
        with TemporaryDirectory() as tmp:
            for method in METHODS:
                idx, train, add = build(method, news, cfg, 2)
                self.assertEqual(idx.metric_type, faiss.METRIC_INNER_PRODUCT)
                self.assertEqual(train > 0, method == "IVFPQ")
                self.assertGreater(add, 0)
                path = Path(tmp) / f"{method}.faiss"
                faiss.write_index(idx, str(path))
                loaded = faiss.read_index(str(path))
                self.assertEqual(loaded.ntotal, len(news))
                for x,y in zip(idx.search(qs,100),loaded.search(qs,100)):
                    np.testing.assert_array_equal(x,y)
                if method == "Flat":
                    for q in qs:
                        dots = news @ q
                        want = np.lexsort((np.arange(len(news)), -dots))[:100]
                        ids, _, _ = exact(loaded,q[None],100)
                        np.testing.assert_array_equal(ids, want)
                times = measure(loaded,qs,list(range(7)),100,2)
                self.assertEqual(len(times),7)
                self.assertGreater(min(times),0)


if __name__ == "__main__":
    unittest.main()
