from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE / "src"))
from mind_two_tower import MeanTowers, hard_loss, inbatch_loss
from mind_hard_negatives import mine, title_counts, verify
from run_mind_two_tower import Event, train_epochs

T = datetime(2019,11,10)
S = timedelta(seconds=1)
CFG = {"token_pattern":r"\w+", "title_tokens":32}


class HardTests(unittest.TestCase):
    def test_mining_visibility_exclusions_and_order(self):
        title = {n:("RED","") for n in ("H","Old","P","N10","N2","Same","Future","Other","N3","N4","N5")}
        title["N2"] = ("red red BLUE","")
        title["Cut"] = (" ".join(["blue"]*32+["red"]),"red")
        counts = title_counts(title,CFG)
        first = Event("0",T,("N10","N2","P","Old","N3","N4","N5","Cut"),bytes([0]*8),())
        same = Event("1",T+S,("Same",),bytes([0]),())
        ids=("Old","P","P","N2","N2","N10","Same","Future","Cut")
        row = Event("2",T+S,ids,bytes([0,1,0,0,0,0,0,0,0]),tuple(["Old"] + ["H"]*20))
        nextrow = Event("3",T+2*S,("N5","N4","N3","N2","N10","Same","Future"),bytes([0]*7),("H",))
        for middle in ((row,same),(same,row)):
            rows = [first,*middle,nextrow]
            lines=list(mine(rows,counts,4))
            byid={r["id"]:r for r in lines}
            self.assertEqual([n for n,_ in byid["2"]["items"]],["N10","N2"])
            self.assertAlmostEqual(byid["2"]["items"][1][1],2/math.sqrt(5))
            self.assertEqual([n for n,_ in byid["3"]["items"]],["Future","N10","N3","N4"])
            with TemporaryDirectory() as tmp:
                path=Path(tmp)/"list.jsonl"
                path.write_text("".join(json.dumps(e)+"\n" for e in lines),encoding="utf8")
                audit=verify(rows,counts,path,4)
                self.assertEqual((audit["pairs"],audit["nonempty"],audit["violations"]),(6,2,0))
                byid["2"]["items"].append(("Same",1.))
                path.write_text("".join(json.dumps(e)+"\n" for e in lines),encoding="utf8")
                with self.assertRaises(ValueError):
                    verify(rows,counts,path,4)

    def test_empty_reasons(self):
        rows=[Event(str(i), T+i*S, ("N1",),bytes([0]),h) for i,h in enumerate(
            [(),("missing",),("M",)])]
        reasons=[e["reason"] for e in mine(rows,title_counts({"M":("blue",""),"N1":("red","")},CFG),4)]
        self.assertEqual(reasons,["empty_history","missing_or_empty_query_title","no_positive_similarity"])

    def test_two_losses_and_empty_gradient(self):
        logits=torch.tensor([[2.,1.],[0.,0.]],dtype=torch.float64,requires_grad=True)
        mask=torch.tensor([[True,True],[False,True]])
        q=torch.tensor([.5,.25],dtype=torch.float64)
        negative=torch.tensor([[1.,99.],[100.,100.]],dtype=torch.float64,requires_grad=True)
        present=torch.tensor([[True,False],[True,True]])
        valid=mask.sum(1)>1
        base=inbatch_loss(logits,mask,q)
        aux=hard_loss(logits.diagonal(),negative,present,valid)
        combined=base+aux
        self.assertAlmostEqual(combined.item(),math.log1p(2/math.e)+math.log1p(1/math.e),places=13)
        grad=torch.autograd.grad(combined,negative,retain_graph=True)[0]
        self.assertTrue((grad[1]==0).all() and grad[0,1]==0)
        empty=hard_loss(logits.diagonal(),negative,torch.zeros_like(present),valid)
        self.assertEqual((base+empty).item(),base.item())
        torch.testing.assert_close(torch.autograd.grad(base+empty,logits,retain_graph=True)[0],
                                   torch.autograd.grad(base,logits)[0],rtol=0,atol=0)
        print("logQ",base.item(),"raw hard",aux.item(),"combined",combined.item())

    def test_training_row_usage_and_shared_updates(self):
        ids={"N1":1,"N2":2,"N3":3}
        events=[Event("1",T,("N1","N2"),bytes([1,0]),("N3",)),
                Event("2",T,("N2","N1"),bytes([1,0]),("N3",))]
        table=torch.tensor([[0],[1],[2],[3]])
        cfg={"epochs":1,"history_limit":2,"batch_size":2,"lr":.001}
        def run(hard):
            torch.manual_seed(7)
            model=MeanTowers(3,4)
            result,_=train_epochs(model,table,events,ids,cfg,"cpu",0,hard_rows=hard)
            return model.words.weight.detach(),result["epochs"][0]
        raw,_=run(None)
        empty,_=run(np.zeros((2,4),dtype=np.int64))
        torch.testing.assert_close(raw,empty,atol=0,rtol=0)
        _,report=run(np.array([[2,0,0,0],[0,0,0,0]],dtype=np.int64))
        self.assertEqual(report["valid_rows"],2)
        self.assertEqual([report["hard"][k] for k in ("rows","uses","overlapping_items","overlapping_columns")],[1]*4)
        self.assertGreater(report["hard"]["loss"],0)


if __name__=="__main__":
    unittest.main()
