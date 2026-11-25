import cudaq
import numpy as np
import pytest
from cudaq_algorithms.primitives import QROM,QROMErase


@pytest.mark.parametrize('words',[(.5,1),(True,1),(-1,1),(0,4)])
def test_erasure_rejects_invalid_words(words):
    with pytest.raises(ValueError):QROMErase(words,2,2)


@pytest.mark.parametrize('block',[1,2])
@pytest.mark.parametrize('seed',[3,8,21])
def test_measured_erasure_preserves_entangled_address_phases(block,seed):
    rom=QROM([1,3,2],address_bits=2,output_bits=2,variant='select_swap',block_size=2)
    erase=rom.measurement_eraser(block_size=block)
    lookup=rom.kernel;unload=erase.kernel();scratch=max(rom.num_ladder,erase.num_scratch)
    @cudaq.kernel
    def run():
        address=cudaq.qvector(2)
        output=cudaq.qvector(2)
        work=cudaq.qvector(scratch)
        reference=cudaq.qubit()
        h(address[0]);h(address[1])
        r1(.31,address[0]);r1(-.43,address[1])
        x.ctrl(address[0],reference)
        lookup(address,work,output)
        unload(address,output,work)
    cudaq.set_random_seed(seed)
    actual=np.array(cudaq.get_state(run));expected=np.zeros_like(actual)
    for a in range(4):
        expected[a|((a&1)<<(4+scratch))]=np.exp(1j*(.31*(a&1)-.43*((a>>1)&1)))/2
    np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=0)


@pytest.mark.parametrize('outcome',[0,1,2,3])
def test_every_measurement_branch_phase_correction(outcome):
    data=[1,3,2];erase=QROMErase(tuple(data),2,2,2)
    correction=erase.correction_kernel();n=erase.num_scratch
    values=[bool(outcome>>i&1) for i in range(2)]
    @cudaq.kernel
    def run(measured:list[bool]):
        address=cudaq.qvector(2)
        scratch=cudaq.qvector(n)
        h(address[0]);h(address[1])
        correction(address,scratch,measured)
    actual=np.array(cudaq.get_state(run,values));expected=np.zeros_like(actual)
    for a in range(4):
        word=data[a] if a<len(data) else 0
        expected[a]=(-1)**((word&outcome).bit_count())/2
    np.testing.assert_allclose(actual,expected,atol=1e-9)


def test_erasure_price_independent_of_word_width_and_compiler_checked():
    for width in [1,5]:
        erase=QROMErase(tuple((7*a+3)%(1<<width) for a in range(64)),6,width,4)
        correction=erase.correction_kernel();n=erase.num_scratch
        @cudaq.kernel
        def run(measured:list[bool]):
            address=cudaq.qvector(6)
            scratch=cudaq.qvector(n)
            correction(address,scratch,measured)
        estimate=cudaq.estimate_resources(run,[True]*width)
        assert estimate.count_controls('x',2)==erase.toffoli_count
        assert erase.toffoli_count==23
