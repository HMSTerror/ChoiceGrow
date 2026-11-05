"""Check fixed shares against dense CTMC kernels and the existing Root policy."""
from copy import deepcopy
import math
import unittest
import torch

from configuration import load_config
from diffusion import ChoiceDiffusion
from transition import FixedShares, UserContext


class FixedShareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, shares):
        config=load_config('config.json')
        config['diffusion'].update(dynamic_rate_gate=False,fixed_shares=shares)
        return config

    def artifact(self, root=False):
        n=6
        base=torch.arange(1,n+1,dtype=torch.float64);base/=base.sum()
        ids=torch.arange(n)
        blocks=torch.zeros((1,n),dtype=torch.long) if root else torch.stack((ids//2,torch.zeros(n,dtype=torch.long)))
        return {'base':base,'neighbor_ids':ids[:,None],
            'neighbor_weights':torch.ones(n,1,dtype=torch.float64),'blocks':blocks}

    def test_fixed_kernel_dense_and_zero_share_root_equivalence(self):
        for s in (0.,.3,.5,.8,1.):
            config=self.config([s,1-s])
            process=ChoiceDiffusion(self.artifact(),config).double()
            gen=process.generator
            self.assertFalse(gen.rate_policy.is_learned)
            self.assertFalse(list(gen.rate_policy.parameters()))
            ctx=gen.context(torch.tensor([[6,0,2]]))
            expanded=UserContext(ctx.rho.expand(6,-1),tuple(v.expand(6,-1) for v in ctx.block_mass),ctx.level_rates)
            eye=torch.eye(6,dtype=torch.float64)
            q=.035*(gen.base[None]-eye)
            for level,rate in enumerate(ctx.level_rates):
                q+=rate*(gen.partition.projection(torch.arange(6),level,expanded.rho,expanded.block_mass)-eye)
            torch.testing.assert_close(q.sum(-1),torch.zeros(6,dtype=torch.float64),atol=2e-14,rtol=0)
            for sigma in (.001,1.,10.):
                actual=process.kernel.probability(torch.arange(6),torch.full((6,),sigma,dtype=torch.float64),expanded)
                torch.testing.assert_close(actual,torch.matrix_exp(sigma*q),atol=2e-12,rtol=2e-11)
                if s==0:
                    root=ChoiceDiffusion(self.artifact(root=True),self.config([1.])).double()
                    root_ctx=root.generator.context(torch.tensor([[6,0,2]]).expand(6,-1))
                    reference=root.kernel.probability(torch.arange(6),torch.full((6,),sigma,dtype=torch.float64),root_ctx)
                    torch.testing.assert_close(actual,reference,atol=2e-12,rtol=2e-11)

    def test_explicit_root_is_bitwise_equal_to_legacy_root(self):
        fixed=self.config([1.])
        legacy=deepcopy(fixed);legacy['diffusion'].pop('fixed_shares')
        legacy['diffusion']['level_rates']=[legacy['diffusion']['total_budget']]
        a=ChoiceDiffusion(self.artifact(root=True),fixed).double()
        b=ChoiceDiffusion(self.artifact(root=True),legacy).double()
        sequence=torch.tensor([[6,0,2],[6,1,3]])
        ca,cb=a.generator.context(sequence),b.generator.context(sequence)
        self.assertEqual(ca.level_rates,cb.level_rates)
        x=torch.tensor([1,4]);sigma=torch.tensor([.5,10.],dtype=torch.float64)
        torch.testing.assert_close(a.kernel.probability(x,sigma,ca),b.kernel.probability(x,sigma,cb),atol=0,rtol=0)

    def test_validation_and_no_gate_rng_side_effect(self):
        config=self.config([.3,.7]);settings=config['diffusion']
        for shares in ([],[.3,.6],[-.1,1.1],[float('nan'),.5]):
            with self.subTest(shares=shares),self.assertRaises(ValueError):
                FixedShares({**settings,'fixed_shares':shares})
        config['diffusion']['dynamic_rate_gate']=True
        with self.assertRaisesRegex(ValueError,'dynamic_rate_gate=false'):
            ChoiceDiffusion(self.artifact(),config)
        config['diffusion']['dynamic_rate_gate']=False
        torch.manual_seed(100);before=torch.random.get_rng_state()
        ChoiceDiffusion(self.artifact(),config)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))


if __name__=='__main__':
    unittest.main()
