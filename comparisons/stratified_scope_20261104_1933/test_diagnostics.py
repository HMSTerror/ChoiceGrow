"""Small CPU checks for statistical meaning, capacities, ranks, and edge deletion."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import unittest
from collections import Counter
import torch
import numpy as np
from scipy import sparse

from neighbors import count_transitions, build_table, history_weights, remove_own_final_edges
from scope import FrequencyStrata, balanced_assignment
from diagnostics_step0 import target_mass_and_rank, frequencies


class DiagnosticMeaningTests(unittest.TestCase):
    def test_history_only_and_no_repeated_prefix_edges(self):
        seq=np.array([[5,0,1,2,3]])
        target=np.array([4])
        clean=count_transitions(seq,target,5,'nn_clean')
        full=count_transitions(seq,target,5,'nn')
        old=count_transitions(seq,target,5,'no')
        self.assertFalse(any(4 in row for row in clean.values()))
        self.assertEqual(clean[0],Counter({1:1.,2:.5,3:1/3}))
        self.assertEqual(full[0][1],1.)
        self.assertEqual(full[3][4],1.)
        self.assertEqual(old,{3:Counter({4:1.}),2:Counter({4:.5}),1:Counter({4:1/3})})
        _,_,_,counts,_=frequencies({'seq':seq,'target':target},5)
        np.testing.assert_array_equal(counts,[0,1,1,1,1])

    def test_paired_deletion_equals_rebuilding_without_one_label(self):
        seq=np.array([[0,1,1,2],[1,2,3,4],[0,3,4,5]])
        targets=np.array([3,5,1])
        for mode in ('nn','no'):
            counts=count_transitions(seq,targets,6,mode)
            table=build_table(counts,6)
            removed=remove_own_final_edges(seq,targets,counts,table,6)
            for row in range(len(seq)):
                independent={i:Counter(c) for i,c in counts.items()}
                for lag,source in enumerate(reversed(seq[row][-3:]),1):
                    independent[int(source)][int(targets[row])]-=1/lag
                rebuilt=build_table(independent,6)
                expected=history_weights(seq[row:row+1],6)@rebuilt
                np.testing.assert_allclose(removed.getrow(row).toarray(),expected.toarray(),atol=1e-13)

    def test_sparse_rank_equals_dense_with_ties_and_gamma(self):
        base=np.array([.1,.1,.15,.15,.2,.3])
        rng=np.random.default_rng(123)
        values=rng.random((12,6))
        values[values<.5]=0
        for row in np.flatnonzero(values.sum(1)==0):
            values[row,row%6]=1.
        values/=values.sum(1,keepdims=True)
        destination=sparse.csr_matrix(values)
        targets=np.arange(12)%6
        for gamma in (0.,.02,.10,1.):
            _,actual=target_mass_and_rank(destination,targets,base,gamma)
            scores=(1-gamma)*base+gamma*values
            expected=[]
            for row,target in enumerate(targets):
                order=np.lexsort((np.arange(6),-scores[row]))
                expected.append(int(np.flatnonzero(order==target)[0])+1)
            np.testing.assert_array_equal(actual,expected)

    def test_paired_deletion_preserves_exact_zero_target_and_tied_rank(self):
        # Catalog-sized background ties make cancellation residues observable.
        n=11924
        seq=np.array([[13,18,13,19],[16,20,28,4],[16,13,10,7],[1,12,24,2]])
        targets=np.array([12,29,28,6])
        counts=count_transitions(seq,targets,n,'nn')
        actual=remove_own_final_edges(seq,targets,counts,build_table(counts,n),n)
        self.assertEqual(float(actual[0,12]),0.)
        _,ranks=target_mass_and_rank(actual,targets,np.ones(n)/n)
        np.testing.assert_array_equal(ranks,[17,30,8,9])

    def test_partial_strata_match_random_and_balance_total_sizes(self):
        rng=np.random.default_rng(5)
        for n,k,m in ((17,4,2),(24,4,2),(31,5,2),(7,3,20)):
            base=np.linspace(1,2,n)
            base/=base.sum()
            strata=FrequencyStrata.build(base,k,m)
            random=strata.random_labels()
            features=rng.normal(size=(n,5))
            features/=np.linalg.norm(features,axis=1,keepdims=True)
            actual,_,_=balanced_assignment(features,features[:k],strata,2)
            for items,slots in zip(strata.items,strata.slots):
                expected=np.bincount(slots,minlength=k)
                np.testing.assert_array_equal(np.bincount(actual[items],minlength=k),expected)
                np.testing.assert_array_equal(np.bincount(random[items],minlength=k),expected)
            sizes=np.bincount(actual,minlength=k)
            self.assertLessEqual(sizes.max()-sizes.min(),1)

    def test_large_balanced_transport_integral_capacity(self):
        rng=np.random.default_rng(8)
        n,k=641,3
        features=rng.normal(size=(n,4))
        features/=np.linalg.norm(features,axis=1,keepdims=True)
        strata=FrequencyStrata.build(np.ones(n)/n,k,300)
        labels,_,_=balanced_assignment(features,features[:k],strata,1)
        np.testing.assert_array_equal(np.bincount(labels,minlength=k),np.bincount(strata.slots[0],minlength=k))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
