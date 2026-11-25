"""Known-zero word omission must preserve full-Fock phases and coherent data."""
import cudaq
import numpy as np
from cudaq_algorithms.basis_change import fermion_basis_change
from cudaq_algorithms.primitives import PhaseGradient,QROM


def test_compact_gradient_words_in_coherent_full_fock_superposition():
    plan=fermion_basis_change(np.array([.6,.8]),programmable=True,angle_bits=3)
    resource=PhaseGradient(3);pre=resource.prepare_kernel();undo=resource.unprepare_kernel()
    # Schedule fixed by [.6,.8]; quantum table selects independent theta codes.
    fw=plan.phase_gradient_kernel(parameter_slots=(0,))
    inverse=plan.phase_gradient_kernel(adjoint=True,parameter_slots=(0,))
    table=QROM([1,3],1,3);lookup=table.kernel;nl=table.num_ladder
    @cudaq.kernel
    def run(roundtrip:bool):
        system=cudaq.qvector(2)
        address=cudaq.qvector(1)
        words=cudaq.qvector(3)
        gradient=cudaq.qvector(3)
        scratch=cudaq.qvector(4)
        h(system[0]);h(system[1]);r1(.37,system[1]);h(address[0])
        pre(gradient);lookup(address,scratch.front(nl),words)
        fw(system,words,gradient,scratch)
        if roundtrip:inverse(system,words,gradient,scratch)
        lookup(address,scratch.front(nl),words);undo(gradient)
    initial=np.array([1,1,np.exp(.37j),np.exp(.37j)])/np.sqrt(8)
    for roundtrip in [False,True]:
        actual=np.array(cudaq.get_state(run,roundtrip));expected=np.zeros_like(actual)
        for a,code in enumerate([1,3]):
            theta=code*np.pi/4;c,s=np.cos(theta),np.sin(theta)
            fock=np.eye(4,dtype=complex);fock[1:3,1:3]=[[c,-s],[s,c]]
            expected[(a<<2)+np.arange(4)]=initial if roundtrip else fock@initial
        np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=0)
