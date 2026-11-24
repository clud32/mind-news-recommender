import math
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from mind_din import DIN, fixed_order, impression_loss, pool_scores


class DINTests(unittest.TestCase):
    def test_candidate_weights_mask_and_empty_history(self):
        model=DIN(1,(1,1),1)
        with torch.no_grad():
            for p in model.local.parameters(): p.zero_()
            model.local[0].weight[0,-1]=1.
            model.local[2].weight.fill_(1.)
            model.local[4].weight.fill_(1.)
        h=torch.tensor([[[1.],[99.]],[[1.],[99.]],[[2.],[3.]]])
        mask=torch.tensor([[1,0],[1,0],[0,0]],dtype=torch.bool)
        c=torch.tensor([[1.],[2.],[1.]])
        w=model.weights(c,h,mask)
        self.assertGreater(w[1,0],w[0,0])
        self.assertTrue(torch.equal(w[:,1],torch.zeros(3)))
        self.assertEqual(w[2].sum().item(),0.)
        self.assertTrue(torch.isfinite(model(c,h,mask)).all())

    def test_pool_denominators(self):
        a,b=pool_scores(["A","B"],["B","C"])
        self.assertAlmostEqual(a,.6309297536,places=9)
        self.assertAlmostEqual(b,.3868528072,places=9)
        self.assertEqual(pool_scores(["A","B"],["X"]),(None,0.))
        self.assertAlmostEqual((b+0)/2,.1934264036,places=9)
        print("one of two clicks ranked second",a,b,"zero hits",0.)

    def test_permutation_and_fallback(self):
        names=["N2","N10","N3"]
        out=fixed_order(names,[1,1,3],False)
        self.assertEqual(out,["N3","N10","N2"])
        self.assertCountEqual(names,out)
        self.assertEqual(fixed_order(names,[1,1,3],True),names)

    def test_equal_impression_weight_multi_positive(self):
        logits=torch.tensor([2.,0.,-1.,2.],requires_grad=True)
        labels=torch.tensor([1.,1.,0.,0.])
        element=F.binary_cross_entropy_with_logits(logits,labels,reduction="none")
        expected=(element[:3].mean()+element[3])/2
        actual=impression_loss(logits,labels,torch.tensor([3,1]))
        torch.testing.assert_close(actual,expected)
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
