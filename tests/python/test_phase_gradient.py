import cudaq
import numpy as np
import pytest
from cudaq_algorithms.primitives import PhaseGradient


@pytest.mark.parametrize('bits',[1,2,3,4])
def test_variable_phase_entangled_words_catalyst_and_inverse(bits):
    cudaq.set_target('qpp-cpu')
    resource=PhaseGradient(bits)
    pre=resource.prepare_kernel();undo=resource.unprepare_kernel()
    phase=resource.phase_kernel();inverse=resource.phase_kernel(adjoint=True)
    width=resource.num_scratch
    @cudaq.kernel
    def run(roundtrip:bool,unload:bool):
        target=cudaq.qubit()
        word=cudaq.qvector(bits)
        gradient=cudaq.qvector(bits)
        scratch=cudaq.qvector(width)
        h(target)
        for k in range(bits):h(word[k])
        pre(gradient)
        phase(word,target,gradient,scratch)
        if roundtrip:inverse(word,target,gradient,scratch)
        if unload:undo(gradient)
    N=1<<bits
    expected=np.zeros(1<<(1+2*bits+width),complex)
    for word in range(N):
        for target in range(2):
            for g in range(N):
                expected[target|(word<<1)|(g<<(bits+1))]=np.exp(2j*np.pi*(word*target-g)/N)/np.sqrt(2*N*N)
    np.testing.assert_allclose(np.array(cudaq.get_state(run,False,False)),expected,atol=1e-10)
    expected[:]=0
    expected[:2*N]=1/np.sqrt(2*N)
    np.testing.assert_allclose(np.array(cudaq.get_state(run,True,True)),expected,atol=1e-10)


@pytest.mark.parametrize('bits',[1,2,3,5])
def test_phase_toffoli_price_matches_compiler(bits):
    resource=PhaseGradient(bits);phase=resource.phase_kernel();width=resource.num_scratch
    @cudaq.kernel
    def run():
        target=cudaq.qubit()
        word=cudaq.qvector(bits)
        gradient=cudaq.qvector(bits)
        scratch=cudaq.qvector(width)
        phase(word,target,gradient,scratch)
    estimate=cudaq.estimate_resources(run)
    assert estimate.count_controls('x',2)==resource.phase_toffoli_count
