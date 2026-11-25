import cudaq
import numpy as np
import pytest
from cudaq_algorithms import PauliLCU
from cudaq_algorithms.unary_phase_estimation import UnaryPhaseEstimation


@cudaq.kernel
def plus_input(system:cudaq.qview):
    h(system[0])


@pytest.mark.parametrize('queries',[1,2,4])
def test_signed_sine_window_histogram_against_physical_spectrum(queries):
    enc=PauliLCU({'I':.19,'Z':.37,'X':-.23})
    qpe=UnaryPhaseEstimation(enc,queries)
    state=np.array(cudaq.get_state(qpe.kernel(state_prep=plus_input)))
    p=qpe.num_phase;m=qpe.num_unary;na=enc.num_ancilla
    observed=(abs(state)**2).reshape(1<<(m+na),1<<p,2).sum(axis=(0,2))
    H=np.array([[.56,-.23],[-.23,-.18]])
    values,vectors=np.linalg.eigh(H)
    overlap=abs(vectors.T@np.ones(2)/np.sqrt(2))**2
    N=1<<p;expected=np.zeros(N)
    for value,probability in zip(values,overlap):
        angle=np.arccos(-value/enc.alpha)
        for sign in [-1,1]:
            amplitudes=np.zeros(N,complex)
            amplitudes[:qpe.num_states]=qpe.amplitudes*np.exp(1j*sign*angle*np.arange(qpe.num_states))
            expected+=.5*probability*abs(np.fft.fft(amplitudes)/np.sqrt(N))**2
    np.testing.assert_allclose(observed,expected,atol=1e-9,rtol=0)
    assert qpe.resources['total_select_queries']==queries+1
    # The low parity bit preserves sign; equal absolute block eigenvalues
    # map to different grid halves, without exact energies in the decoder.
    assert qpe.phase_to_energy(0)==-enc.alpha
    assert abs(qpe.phase_to_energy(N//2)-enc.alpha)<1e-12


def test_control_preparation_zero_padding_and_normalization():
    qpe=UnaryPhaseEstimation(PauliLCU({'Z':1.}),2)
    control=qpe.control_kernel();p=qpe.num_phase
    @cudaq.kernel
    def run():
        q=cudaq.qvector(p)
        control(q)
    actual=np.array(cudaq.get_state(run))
    np.testing.assert_allclose(actual[:qpe.num_states],qpe.amplitudes,atol=1e-12)
    np.testing.assert_allclose(actual[qpe.num_states:],0.,atol=1e-12)


@pytest.mark.parametrize('queries',[2,3])
def test_coherent_selected_powers_against_explicit_small_walk(queries):
    from cudaq_algorithms import state_from
    enc=PauliLCU([(.4,'Z'),(-.6,'X')]);qpe=UnaryPhaseEstimation(enc,queries)
    p,m=qpe.num_phase,qpe.num_unary;total=1+p+m+enc.num_ancilla
    multiplex=qpe.multiplex_kernel()
    @cudaq.kernel
    def run(state:cudaq.State):
        register=cudaq.qvector(state)
        multiplex(register.front(1),register[1:total])
    # Independent 4x4 zero-frame block and signed walk, system bit first.
    P=np.array([[np.sqrt(.4),-np.sqrt(.6)],[np.sqrt(.6),np.sqrt(.4)]])
    selected=np.zeros((4,4));selected[:2,:2]=np.diag([1.,-1.]);selected[2:,2:]=[[0.,-1.],[-1.,0.]]
    U=np.kron(P.T,np.eye(2))@selected@np.kron(P,np.eye(2))
    W=np.diag([-1.,-1.,1.,1.])@U
    rng=np.random.default_rng(113+queries)
    branches=rng.normal(size=(qpe.num_states,4))+1j*rng.normal(size=(qpe.num_states,4))
    branches/=np.linalg.norm(branches)
    initial=np.zeros(1<<total,complex);expected=np.zeros_like(initial)
    for r,branch in enumerate(branches):
        transformed=np.linalg.matrix_power(W,r-queries)@branch
        for physical in range(4):
            offset=(physical&1)|(r<<1)|((physical>>1)<<(1+p+m))
            initial[offset]=branch[physical];expected[offset]=transformed[physical]
    np.testing.assert_allclose(np.array(cudaq.get_state(run,state_from(initial))),expected,atol=1e-9,rtol=0)
