import cudaq
import numpy as np
from cudaq_algorithms.basis_change import fermion_basis_change
from cudaq_algorithms.primitives import PhaseGradient


def test_one_particle_gradient_restriction_retains_scalar_phases():
    plan=fermion_basis_change(np.array([.6,.8j]),programmable=True,angle_bits=3)
    resource=PhaseGradient(3);nw=plan.num_qubits-2
    pre=resource.prepare_kernel();undo=resource.unprepare_kernel()
    fw=plan.one_particle_gradient_kernel();adj=plan.one_particle_gradient_kernel(adjoint=True)
    packed=sum(word << (3*k) for k,word in enumerate(plan.encoded_words))
    @cudaq.kernel
    def run(occupation:int,roundtrip:bool):
        system=cudaq.qvector(1)
        words=cudaq.qvector(nw)
        gradient=cudaq.qvector(3)
        scratch=cudaq.qvector(4)
        if occupation==1:x(system[0])
        for k in range(nw):
            if (packed>>k)&1:x(words[k])
        pre(gradient);fw(system,words,gradient,scratch)
        if roundtrip:adj(system,words,gradient,scratch)
        undo(gradient)
        for k in range(nw):
            if (packed>>k)&1:x(words[k])
    matrix=plan.schedule.orbital_matrix(plan.represented_parameters)
    for occupation in range(2):
        actual=np.array(cudaq.get_state(run,occupation,False));expected=np.zeros_like(actual)
        expected[:2]=matrix[:,occupation]
        np.testing.assert_allclose(actual,expected,atol=1e-9)
    actual=np.array(cudaq.get_state(run,1,True));expected=np.zeros_like(actual);expected[1]=1
    np.testing.assert_allclose(actual,expected,atol=1e-9)
