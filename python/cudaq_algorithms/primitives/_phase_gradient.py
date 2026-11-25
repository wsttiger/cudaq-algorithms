# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Catalytic variable phases from modular addition into a Fourier resource.

The resource is |G_b> = 2^(-b/2) sum_j exp(-2*pi*i*j/2^b)|j>.
Adding a quantum word a to it kicks back exp(+2*pi*i*a/2^b).
Masking the word with the target occupation implements an exact R1 phase
on the represented grid without an arbitrary target rotation. The coherent
CDKM implementation costs 4*b-2 Toffolis per phase (including masking),
not the lower measurement-assisted adder price quoted in the paper.
Resource preparation uses b Hadamards and b specified dyadic R1 gates;
its synthesis cost is separate and must not be omitted from total estimates.
"""
from dataclasses import dataclass
from functools import cached_property
import math
import cudaq
from ._arithmetic import add_register, subtract_register


@dataclass(frozen=True)
class PhaseGradient:
    """Caller-owned little-endian catalytic register of ``width`` qubits.

    ``phase_kernel`` has signature ``(word, target, gradient, scratch)``.
    Word and gradient each have width b. Scratch has b+1 clean qubits
    (masked word, then carry). The operation restores all three views on
    the promised resource sector; only the target phase changes. The word
    may be entangled with other systems. Width/aliasing are caller contracts
    because CUDA-Q does not dynamically bounds-check qviews.
    """
    width: int

    def __post_init__(self):
        if isinstance(self.width,bool) or int(self.width)!=self.width or not 1<=self.width<=52:
            raise ValueError('phase-gradient width must be an integer in [1,52]')
        object.__setattr__(self,'width',int(self.width))

    @property
    def num_scratch(self):return self.width+1

    @property
    def phase_toffoli_count(self):return 4*self.width-2

    @property
    def preparation_angles(self):
        return tuple(-math.tau*2.**(k-self.width) for k in range(self.width))

    @cached_property
    def _prepare(self):
        b=self.width;angles=list(self.preparation_angles)
        @cudaq.kernel
        def prepare_gradient(gradient:cudaq.qview):
            for k in range(b):
                h(gradient[k])
                r1(angles[k],gradient[k])
        return prepare_gradient

    @cached_property
    def _unprepare(self):
        b=self.width;angles=list(self.preparation_angles)
        @cudaq.kernel
        def unprepare_gradient(gradient:cudaq.qview):
            for j in range(b):
                k=b-1-j
                r1(-angles[k],gradient[k])
                h(gradient[k])
        return unprepare_gradient

    def prepare_kernel(self):return self._prepare
    def unprepare_kernel(self):return self._unprepare

    def _phase(self,adjoint):
        b=self.width
        arithmetic=subtract_register if adjoint else add_register
        @cudaq.kernel
        def phase_from_word(word:cudaq.qview,target:cudaq.qubit,
                            gradient:cudaq.qview,scratch:cudaq.qview):
            for k in range(b):
                x.ctrl(target,word[k],scratch[k])
            arithmetic(scratch.front(b),gradient,scratch.back(1))
            for j in range(b):
                k=b-1-j
                x.ctrl(target,word[k],scratch[k])
        return phase_from_word

    @cached_property
    def _forward(self):return self._phase(False)
    @cached_property
    def _inverse(self):return self._phase(True)

    def phase_kernel(self,*,adjoint=False):
        if not isinstance(adjoint,bool):raise ValueError('adjoint must be a boolean')
        return self._inverse if adjoint else self._forward
