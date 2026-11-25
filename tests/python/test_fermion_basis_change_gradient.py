import cudaq
import numpy as np
import pytest
from cudaq_algorithms.basis_change import fermion_basis_change, GivensStructure
from cudaq_algorithms.primitives import PhaseGradient


@pytest.mark.parametrize('bits',[2,3])
def test_gradient_basis_matches_complete_complex_fock_phases(bits):
    cudaq.set_target('qpp-cpu')
    rng=np.random.default_rng(418)
    unitary,_=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))
    plan=fermion_basis_change(unitary,programmable=True,angle_bits=bits)
    resource=PhaseGradient(bits);nw=plan.num_qubits-2;ns=resource.num_scratch
    pre=resource.prepare_kernel();undo=resource.unprepare_kernel()
    forward=plan.phase_gradient_kernel();inverse=plan.phase_gradient_kernel(adjoint=True)
    packed=sum(word << (bits*k) for k,word in enumerate(plan.encoded_words))
    @cudaq.kernel
    def run(occupation:int,roundtrip:bool):
        system=cudaq.qvector(2)
        words=cudaq.qvector(nw)
        gradient=cudaq.qvector(bits)
        scratch=cudaq.qvector(ns)
        for k in range(2):
            if (occupation>>k)&1:x(system[k])
        for k in range(nw):
            if (packed>>k)&1:x(words[k])
        pre(gradient)
        forward(system,words,gradient,scratch)
        if roundtrip:inverse(system,words,gradient,scratch)
        undo(gradient)
        for k in range(nw):
            if (packed>>k)&1:x(words[k])
    u=plan.schedule.orbital_matrix(plan.represented_parameters)
    fock=np.zeros((4,4),complex);fock[0,0]=1;fock[1:3,1:3]=u;fock[3,3]=np.linalg.det(u)
    for occupation in range(4):
        actual=np.array(cudaq.get_state(run,occupation,False))
        expected=np.zeros_like(actual);expected[:4]=fock[:,occupation]
        np.testing.assert_allclose(actual,expected,atol=1e-9)
    actual=np.array(cudaq.get_state(run,3,True));expected=np.zeros_like(actual);expected[3]=1
    np.testing.assert_allclose(actual,expected,atol=1e-9)
