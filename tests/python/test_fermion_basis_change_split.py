"""Separate register composition needed by block-encoding callbacks."""
import cudaq
import numpy as np
from cudaq_algorithms.basis_change import GivensStructure, fermion_basis_change
from cudaq_algorithms.primitives import QROM


def test_split_views_consume_loaded_words_and_clear_scratch():
    cudaq.set_target('qpp-cpu')
    structure=GivensStructure.partial(2,mode_map=(0,2))
    plan=fermion_basis_change(structure=structure,programmable=True,angle_bits=2)
    codes=[(1,0,0),(3,2,0)]
    table=QROM([sum(c << (2*k) for k,c in enumerate(row)) for row in codes],
               address_bits=1,output_bits=6)
    lookup=table.kernel;fw=plan.split_kernel();adj=plan.split_kernel(adjoint=True)
    ladder_width=table.num_ladder
    @cudaq.kernel
    def run(roundtrip:bool):
        system=cudaq.qvector(3)
        words=cudaq.qvector(6)
        address=cudaq.qvector(1)
        scratch=cudaq.qvector(ladder_width)
        for q in range(3):
            h(system[q])
        h(address[0])
        lookup(address,scratch,words)
        fw(system,words)
        if roundtrip:
            adj(system,words)
        lookup(address,scratch,words)
    expected=np.zeros(1 << (3+6+1+table.num_ladder),complex)
    for branch,(mix,phase0,phase1) in enumerate(codes):
        theta=mix*np.pi/2
        u=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        u=np.diag(np.exp(1j*np.array([phase0,phase1])*np.pi/2))@u
        orbital=np.eye(3,dtype=complex);orbital[np.ix_([0,2],[0,2])]=u
        mat=np.zeros((8,8),complex)
        for row in range(8):
            rows=[i for i in range(3) if row>>i&1]
            for col in range(8):
                cols=[i for i in range(3) if col>>i&1]
                if len(rows)==len(cols):mat[row,col]=np.linalg.det(orbital[np.ix_(rows,cols)])
        expected[(branch << 9)+np.arange(8)]=mat@np.ones(8)/4
    np.testing.assert_allclose(np.array(cudaq.get_state(run,False)),expected,atol=1e-10)
    zero_expected=np.zeros_like(expected)
    for branch in range(2):zero_expected[(branch<<9)+np.arange(8)]=.25
    np.testing.assert_allclose(np.array(cudaq.get_state(run,True)),zero_expected,atol=1e-10)
