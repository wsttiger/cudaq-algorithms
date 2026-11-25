import cudaq
import numpy as np
import pytest
from cudaq_algorithms.thc_prepare import THCPairDomain,TriangularIndex,THCPairPrepare


@pytest.mark.parametrize('n,rank',[(2,1),(2,2),(3,2),(2,3)])
def test_valid_pair_domain_amplitudes_and_inverse(n,rank):
    domain=THCPairDomain(n,rank);k=domain.pair_width;s=domain.num_scratch
    pre=domain.kernel();undo=domain.adjoint_kernel()
    @cudaq.kernel
    def run(inverse:bool):
        pair=cudaq.qvector(2*k)
        work=cudaq.qvector(s)
        pre(pair,work)
        if inverse:undo(pair,work)
    actual=np.array(cudaq.get_state(run,False));expected=np.zeros_like(actual)
    for mu,nu in domain.pairs:expected[mu|(nu<<k)]=1/np.sqrt(domain.num_valid)
    np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=0)
    expected[:]=0;expected[0]=1
    np.testing.assert_allclose(np.array(cudaq.get_state(run,True)),expected,atol=1e-9)


def test_triangular_index_nonzero_output_and_inverse():
    indexing=TriangularIndex(1,2);index=indexing.kernel()
    @cudaq.kernel
    def run(twice:bool):
        pair=cudaq.qvector(2)
        output=cudaq.qvector(2)
        scratch=cudaq.qvector(0)
        h(pair[0]);h(pair[1]);x(output[0])
        index(pair,output,scratch)
        if twice:index(pair,output,scratch)
    for twice in [False,True]:
        actual=np.array(cudaq.get_state(run,twice));expected=np.zeros_like(actual)
        for mu in range(2):
            for nu in range(2):
                word=1 if twice else 1^(nu*(nu+1)//2+mu)
                expected[mu|(nu<<1)|(word<<2)]=.5
        np.testing.assert_allclose(actual,expected,atol=1e-10)


def test_generic_triangular_index_coherent_xor_and_clean_workspace():
    obj=TriangularIndex(2,4);index=obj.kernel();s=obj.num_scratch
    @cudaq.kernel
    def run(twice:bool):
        pair=cudaq.qvector(4)
        output=cudaq.qvector(4)
        scratch=cudaq.qvector(s)
        for bit in range(4):h(pair[bit])
        x(output[0]);x(output[2])
        index(pair,output,scratch)
        if twice:index(pair,output,scratch)
    for twice in [False,True]:
        actual=np.array(cudaq.get_state(run,twice));expected=np.zeros_like(actual)
        for mu in range(4):
            for nu in range(4):
                word=5 if twice else 5^(nu*(nu+1)//2+mu)
                expected[mu|(nu<<2)|(word<<4)]=.25
        np.testing.assert_allclose(actual,expected,atol=1e-10)


def test_uniform_specialization_uses_realized_alias_distribution():
    # A tiny nonuniformity rounds away at finite keep precision. At higher
    # precision it must disable the specialization, however small the error.
    one=[1.,1.+1e-13];z=np.array([[2.]])
    assert THCPairPrepare(one,z,keep_bits=2).uniform_nonnegative
    assert not THCPairPrepare(one,z,keep_bits=48).uniform_nonnegative


def test_pair_alias_valid_source_histories_signs_and_clean_scratch():
    obj=THCPairPrepare([-.17,.41],np.array([[.63]]),keep_bits=2)
    k=obj.pair_width;g=obj.num_garbage;s=obj.num_scratch;b=obj.keep_bits
    pre=obj.kernel();undo=obj.adjoint_kernel()
    @cudaq.kernel
    def run(inverse:bool):
        pair=cudaq.qvector(2*k)
        garbage=cudaq.qvector(g)
        scratch=cudaq.qvector(s)
        pre(pair,garbage,scratch)
        if inverse:undo(pair,garbage,scratch)
    actual=np.array(cudaq.get_state(run,False));expected=np.zeros_like(actual)
    d=len(obj.pairs)
    for source,(mu,nu) in enumerate(obj.pairs):
        for coin in range(1<<b):
            flag=int(coin>=obj.keep[source]);alt=obj.alternate[source]
            selected=alt if flag else source
            chosen_mu,chosen_nu=obj.pairs[selected]
            alt_mu,alt_nu=(mu,nu) if flag else obj.pairs[alt]
            sign=int(obj.weights[selected]<0)
            alt_sign=int(obj.weights[source if flag else alt]<0)
            garbage=(alt_mu|(alt_nu<<k)|(obj.keep[source]<<(2*k))|
                     (sign<<(2*k+b))|(alt_sign<<(2*k+b+1))|
                     (coin<<(2*k+b+2))|(flag<<(2*k+2*b+2)))
            expected[chosen_mu|(chosen_nu<<k)|(garbage<<(2*k))]+=1/np.sqrt(d*(1<<b))
    np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=0)
    expected[:]=0;expected[0]=1
    np.testing.assert_allclose(np.array(cudaq.get_state(run,True)),expected,atol=1e-9)


def test_pair_alias_zero_weights_and_keep_threshold_edges():
    obj=THCPairPrepare([0.,.5],np.array([[0.]]),keep_bits=2)
    assert 0 in obj.keep and (1<<obj.keep_bits)-1 in obj.keep
    np.testing.assert_allclose(obj.table_probabilities,[0.,0.,1.],atol=1e-14)
    pre=obj.kernel();undo=obj.adjoint_kernel();g=obj.num_garbage;s=obj.num_scratch
    @cudaq.kernel
    def run(inverse:bool):
        pair=cudaq.qvector(2);garbage=cudaq.qvector(g);scratch=cudaq.qvector(s)
        pre(pair,garbage,scratch)
        if inverse:undo(pair,garbage,scratch)
    state=np.array(cudaq.get_state(run,False))
    marginal=(abs(state)**2).reshape(-1,4).sum(axis=0)
    np.testing.assert_allclose(marginal,[0.,0.,0.,1.],atol=1e-9)
    assert np.linalg.norm(state[1<<(2+g):])<1e-9
    actual=np.array(cudaq.get_state(run,True));expected=np.zeros_like(actual);expected[0]=1
    np.testing.assert_allclose(actual,expected,atol=1e-9)
