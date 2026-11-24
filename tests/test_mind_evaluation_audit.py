from dataclasses import replace
from datetime import datetime, timedelta
from itertools import permutations
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from mind_din import DIN
from mind_evaluation import Impression, METRICS, evaluate
from mind_two_tower import MeanTowers
from run_mind_two_tower import Event
from run_mind_evaluation_audit import (Scorer, expected_random, invariant, pair_donors, permuted,
    rng_stream, sample_replay, sampling_row, split_ids, negatives)
from test_mind_evaluation import load_scorer, reference_metrics


def event(iid,second,history=("N1",),candidates=("N1","N2","N3"),labels=(1,0,0)):
    return Event(iid,datetime(2019,11,10)+timedelta(seconds=second),tuple(candidates),bytes(labels),tuple(history))


class AuditTests(unittest.TestCase):
    def test_split_original_order_and_holdout(self):
        raw=list("abcdefghij")
        chronological=list("hebfidgcaj")
        result=split_ids(raw,chronological,6,0)
        perm=np.random.default_rng(0).permutation(raw)
        self.assertEqual(result["random_train"],perm[:6].tolist())
        self.assertEqual(result["random_valid"],perm[6:].tolist())
        expected=[n for n in chronological[6:] if n in perm[6:]]
        self.assertEqual(result["common"],expected)
        for arm in ("random","time"):
            self.assertTrue(set(expected).isdisjoint(result[arm+"_train"]))
        with self.assertRaises(ValueError):
            split_ids(["a","a"],["a","b"],1,0)

    def test_donors_require_prior_other_user_same_hour_and_length(self):
        rows=[event("a",3599),event("b",3600),event("c",3601),event("d",3601),
              event("e",3602,("N1","N2")),event("f",3603,("N2","N3","N4")),event("g",3604,())]
        users=dict(zip("abcdefg",["A","X","X","D","Y","F","G"]))
        result=pair_donors(rows,users,set(users),2,rng_stream(0,10))
        self.assertEqual(result,dict(a=None,b=None,c=None,d="b",e=None,f="e"))

    def test_label_invariance_independent_ranks_and_official_expectations(self):
        torch.manual_seed(7)
        model=DIN(3,(4,2),4).eval()
        bank=torch.randn(4,3)
        tower=MeanTowers(4,3).eval()
        scorer=Scorer(tower,model,bank,{"N1":1,"N2":2,"N3":3},2)
        scorer.cpu_bank=bank
        row=event("17",0,labels=(1,0,1))
        alt=permuted(row,rng_stream(1,20))
        before=scorer.score([row],probe=True)[0]
        after=scorer.score([alt],probe=True)[0]
        invariant("17",before,after)
        self.assertEqual(sum(row.labels),sum(alt.labels))
        self.assertEqual(row.candidates,alt.candidates)
        other=(rng_stream(1,21).permutation(3)+1).tolist()
        self.assertNotEqual(rng_stream(1,20).bytes(16),rng_stream(1,21).bytes(16))
        sample=[Impression(alt.labels,after[2]["din"]),Impression(row.labels,other)]
        official=reference_metrics(load_scorer(),sample)
        for k,v in zip(METRICS,official):
            self.assertAlmostEqual(v,evaluate(sample)["metrics"][k],places=10)
        for n,p in ((3,1),(4,2),(5,4)):
            labels=[1]*p+[0]*(n-p)
            avg=evaluate(Impression(labels,order) for order in permutations(range(1,n+1)))["metrics"]
            for k,v in expected_random(n,p).items():
                self.assertAlmostEqual(v,avg[k],places=12)
        empty=scorer.score([replace(row,history=())],probe=True)[0]
        self.assertEqual(empty[2]["tower"],empty[2]["din"])
        self.assertTrue(np.array_equal(empty[1]["candidate_only"],before[1]["candidate_only"]))
        changed=(before[0],dict(before[1],din=before[1]["din"]+1),before[2])
        with self.assertRaises(ValueError):
            invariant("17",before,changed)

    def test_same_second_sampling_visibility_conservation_and_seed(self):
        neg=[f"N{i}" for i in range(20,45)]
        names=["P","Q",*neg]
        labels=[1,1]+[0]*len(neg)
        prior=event("before",0,(),["P",*neg],[1]+[0]*len(neg))
        a=event("a",1,(),names,labels)
        b=event("b",1,(),names,labels)
        c=event("c",2,(),names,labels)
        for same in ([a,b],[b,a]):
            def run():
                return list(sample_replay([prior,*same,c],{"a","b","c"},[rng_stream(1,i) for i in (30,31,32,33)]))
            rows=run()
            self.assertEqual(rows,run())
            self.assertEqual([r.impression_id for r,_ in rows],["c"])
            groups=rows[0][1]
            scores=np.arange(len(names),dtype=np.float32)
            ranks=list(range(len(names),0,-1))
            scored=(tuple(names),dict(tower=scores,din=scores),dict(tower=ranks,din=ranks))
            res=sampling_row(c,scored,groups)
            for key,group in groups.items():
                n=5 if key.endswith("5") else 20
                self.assertEqual(len(group),len(set(group)))
                self.assertEqual(len(group),n+2)
                self.assertTrue({"P","Q"}<=set(group)<=set(names))
                self.assertEqual(res[key]["delta"],0.)
            self.assertIsNone(negatives(c,set(names)-{"Q"}))
            self.assertIsNone(negatives(c,{"P","Q",*neg[:19]}))


if __name__=="__main__":
    unittest.main()
