"""Synthetic loss and inference controls; --build-reference replays CARD-004 for A1."""
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE / "src"))
from mind_two_tower import MeanTowers, click_examples, column_mask, inbatch_loss, target_probabilities
from run_mind_two_tower import Event, replay


class LogQTests(unittest.TestCase):
    def test_card_two_column_example(self):
        logits = torch.tensor([[2., 1.], [0., 0.]], dtype=torch.float64)
        allowed = torch.tensor([[True, True], [False, True]])
        q = torch.tensor([.5, .25], dtype=torch.float64)
        raw = inbatch_loss(logits, allowed).item()
        corrected = inbatch_loss(logits, allowed, q).item()
        self.assertAlmostEqual(raw, math.log1p(1 / math.e), places=14)
        self.assertAlmostEqual(corrected, math.log1p(2 / math.e), places=14)
        print("card [2,1] Q=[1/2,1/4] raw, logQ", repr(raw), repr(corrected))

    def test_count_occurrences_before_mask(self):
        now = datetime(2019, 11, 10)
        events = [Event("1", now, ("N1","N1","N2","N3"), bytes([1,1,1,0]), ()),
                  Event("2", now, ("N2","N3"), bytes([0,1]), ())]
        examples, _ = click_examples(events, {"N1":1,"N2":2,"N3":3})
        counts, q = target_probabilities(examples, 5)
        np.testing.assert_array_equal(counts, [0,2,1,1,0])
        np.testing.assert_array_equal(q,[0,.5,.25,.25,0])
        self.assertEqual(sum(q), 1)

    def test_nonuniform_hand_and_mask(self):
        batch = np.array([[0,1],[1,1],[0,2],[2,3],[3,3]])
        allowed, audit = column_mask(batch,[{1,2},{1},{3},{3}])
        self.assertEqual(allowed[0].tolist(), [True,False,False,True,True])
        self.assertEqual(allowed[3].tolist(), [True,True,True,True,False])
        # Isolate row 0; the last two allowed columns are two occurrences of C.
        mask=torch.eye(5,dtype=torch.bool)
        mask[0]=torch.from_numpy(allowed[0])
        logits=torch.zeros(5,5,dtype=torch.float64)
        logits[0]=torch.tensor([2.,99.,80.,1.,1.])
        logits.requires_grad_()
        q=torch.tensor([.5,.5,.25,.25,.25],dtype=torch.float64)
        actual=inbatch_loss(logits,mask,q)
        hand=math.log1p(2*math.exp(-1)*(.5/.25))
        self.assertAlmostEqual(actual.item(),hand,places=14)
        actual.backward()
        self.assertTrue((logits.grad[0,1:3]==0).all())
        self.assertTrue((logits.grad[1:]==0).all())
        self.assertEqual(logits.grad[0,3],logits.grad[0,4])
        print("nonuniform Q hand",repr(hand),"measured",repr(actual.item()),"abs",abs(hand-actual.item()))

    def test_uniform_q_logits_and_parameter_gradients(self):
        vectors = torch.tensor([[1.,2.],[0.,-2.],[1.,3.]],dtype=torch.float64,requires_grad=True)
        mask=torch.tensor([[1,1,0],[1,1,1],[0,0,1]],dtype=torch.bool)
        dots=vectors @ vectors.T
        raw = inbatch_loss(dots,mask)
        corrected=inbatch_loss(dots,mask,torch.full((3,),.2,dtype=torch.float64))
        a=torch.autograd.grad(raw,(dots,vectors),retain_graph=True)
        b=torch.autograd.grad(corrected,(dots,vectors))
        errors=[abs(raw.item()-corrected.item()),*((x-y).abs().max().item() for x,y in zip(a,b))]
        self.assertLess(max(errors),1e-12)
        print("uniform Q absolute errors [loss, logits gradient, parameter gradient]",errors)

    def test_invalid_probabilities(self):
        logits=torch.eye(2)
        mask=torch.ones(2,2,dtype=torch.bool)
        for bad in (0., -1., float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                inbatch_loss(logits,mask,torch.tensor([.5,bad]))
        with self.assertRaises(ValueError):
            inbatch_loss(logits,mask,torch.ones(3))

    def test_replay_uses_raw_dots(self):
        t=datetime(2019,11,13)
        ids={"N1":1,"N2":2}
        train=[Event("1",t,("N1","N2"),bytes([1,0]),())]
        valid=[Event("2",t,("N1","N2"),bytes([1,0]),("N1",))]
        model=MeanTowers(2,1)
        with torch.no_grad():
            model.words.weight.copy_(torch.tensor([[0.],[1.],[.9]]))
        cfg={"history_limit":20,"evaluation_batch":128,"top_k":1}
        # logQ would reverse these two scores if Q=[.9,.1].
        self.assertLess(1-math.log(.9),.9-math.log(.1))
        r=replay(model,torch.tensor([[0],[1],[2]]),train,valid,ids,cfg,1)["full"]
        self.assertEqual(r["official"]["metrics"]["nDCG@10"],1)
        self.assertEqual(r["retrieval"]["hits"],1)


def build_reference():
    import ast
    from collections import Counter
    from datetime import datetime
    import hashlib
    import json
    from pathlib import Path
    import struct
    import subprocess
    import sys
    import types
    import numpy as np

    root = Path.cwd()
    sys.path.insert(0, str(root))
    from tools.common import payload_sha256, sha256_file

    def frozen(path):
        return subprocess.check_output(["git", "show", f"card-004:{path}"]).decode("utf8")

    cfgpath = "configs/mind-two-tower-v0.1.json"
    cfg = json.loads(frozen(cfgpath))
    experiment = json.loads(Path("configs/mind-two-tower-logq-v0.1.json").read_bytes())
    assert sha256_file(cfgpath) == experiment["base_config_sha256"]
    cut = datetime.fromisoformat(cfg["boundary"])
    ordered = []
    counts = Counter()
    with Path(cfg["behaviors"]).open(encoding="utf8") as f:
        for line in f:
            iid, _, timestamp, _, labels = line.rstrip("\n\r").split("\t")
            ts = datetime.strptime(timestamp, "%m/%d/%Y %I:%M:%S %p")
            if ts < cut:
                clicks = [v.rsplit("-",1)[0] for v in labels.split() if v.endswith("-1")]
                counts.update(clicks)
                ordered.append((ts, iid, clicks))
    ordered.sort(key=lambda row: row[0])
    assert len(ordered) == cfg["train_count"]
    q = dict(total=counts.total(), support=len(counts), min=min(counts.values()), max=max(counts.values()))
    assert q == experiment["q_expected"]
    assert divmod(q["total"], cfg["batch_size"]) == (1467, 80)
    q["counts_sha256"] = payload_sha256(dict(counts))

    # Execute the frozen click-occurrence constructor and the original RNG, epoch
    # permutation, batch loop and batch selection from train_epochs, without fitting.
    source = frozen("src/mind_two_tower.py")
    module = types.ModuleType("card004_frozen_model")
    sys.modules[module.__name__] = module
    exec(compile(source, "<card-004 model>", "exec"), module.__dict__)
    class ClickRow:
        def __init__(self, clicks):
            self.clicks = lambda: iter(clicks)
    examples,_ = module.click_examples([ClickRow(c) for _,_,c in ordered], {n:int(n[1:]) for n in counts})
    train_source = frozen("src/run_mind_two_tower.py")
    train = next(f for f in ast.parse(train_source).body if getattr(f, "name", None) == "train_epochs")
    def assignment(body, name):
        return next(s for s in body if isinstance(s,ast.Assign) and ast.unparse(s.targets[0]) == name)
    epoch_loop = next(s for s in train.body if isinstance(s, ast.For) and ast.unparse(s.target) == "epoch")
    batch_loop = next(s for s in epoch_loop.body if isinstance(s, ast.For) and ast.unparse(s.target) == "offset")
    selection = assignment(batch_loop.body,"batch")
    slice_expr = ast.unparse(selection.value.slice)
    rng = assignment(train.body,"rng")
    permutation = assignment(epoch_loop.body,"order")
    replay_src = (
        "def sampler(examples,cfg,seed):\n    "+ast.unparse(rng)+"\n    for epoch in "+ast.unparse(epoch_loop.iter)+":\n"
        "        "+ast.unparse(permutation)+"\n        for offset in "+ast.unparse(batch_loop.iter)+":\n"
        "            "+ast.unparse(selection)+"\n            yield epoch, "+slice_expr+", batch\n"
    )
    context={"np":np}
    exec(replay_src,context)
    result={"q":q, "seeds":{}, "control_commit":subprocess.check_output(["git","rev-parse","card-004"]).decode().strip()}
    for seed in cfg["seeds"]:
        digests=[hashlib.sha256() for _ in range(cfg["epochs"])]
        for epoch, ordinal, batch in context["sampler"](examples,cfg,seed):
            h=digests[epoch]
            h.update(struct.pack("<I",len(batch)))
            h.update(b"".join(struct.pack("<QQQ",int(pos),int(ordered[i][1]),int(target))
                              for pos,(i,target) in zip(ordinal,batch)))
        result["seeds"][str(seed)] = [h.hexdigest() for h in digests]
    out = Path(experiment["audit"])
    with out.open("x",encoding="utf8",newline="\n") as f:
        json.dump(result,f,indent=2)
        f.write("\n")
    print(replay_src, json.dumps(result,indent=2), "reference file sha256",sha256_file(out),sep="\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["--build-reference"]:
        build_reference()
    else:
        unittest.main()
