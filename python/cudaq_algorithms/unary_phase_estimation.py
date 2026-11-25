# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sine-window phase estimation with shared unary controlled reflections.

For q>=1, the shared unary circuit selects W**(2*t-q), t=0,...,q,
with q unconditional Hermitian SELECT calls. Internal reflections are
skipped at the selected boundary; endpoints have their own reflection.
One additional controlled W on a parity bit b makes the powers
W**(2*t+b-q) consecutive. This is essential for a signed decoder:
the even-power-only circuit aliases opposite signed block eigenvalues.
The control amplitudes are the optimal sine window over 2*(q+1) states,
zero padded for the inverse QFT. Non-power-of-two query counts are valid.

There are q SELECT queries plus one controlled-W query; actual unary
ladder and reflection costs are reported separately. Coherent ladder
uncomputation has its actual library cost, not a measurement-assisted
paper price. State preparation currently uses the library rotation tree.
"""
import math
import cudaq
import numpy as np
from .phase_estimation import PhaseEstimation
from .pauli_lcu import prepare
from .primitives import iqft, unary_iteration_kernels
from .primitives._unary_iteration import _retain
from .common_kernels import controlled_reflect_about_zero


@cudaq.kernel
def _empty_frame(ancilla:cudaq.qview):
    pass


class UnaryPhaseEstimation:
    """Optimized reflection schedule with an explicit signed-phase parity bit.

    Encodings may provide ``select_kernel()`` and
    ``controlled_reflection_kernel()`` for the prepared frame. The latter
    takes [enable, ancilla...] and includes inverse preparation/reflection/
    preparation, excluding reusable catalysts from the reflected signal.
    If absent, the generic adapter uses the Hermitian full block U_A as
    SELECT and a zero reflection. This has additional preparation cost.
    """
    def __init__(self,encoding,queries):
        if isinstance(queries,bool) or int(queries)!=queries or queries<1:
            raise ValueError('queries must be a positive integer')
        self.encoding=encoding;self.queries=int(queries)
        self.num_states=2*(self.queries+1)
        self.num_phase=(self.num_states-1).bit_length()
        self.num_unary=self.num_phase-1
        self.amplitudes=np.sqrt(2/(self.num_states+1))*np.sin(
            np.pi*(np.arange(self.num_states)+1)/(self.num_states+1))
        leaves=np.zeros(1<<self.num_phase)
        for j,amplitude in enumerate(self.amplitudes):
            leaves[int(format(j,f'0{self.num_phase}b')[::-1],2)]=amplitude**2
        self._angles=[]
        for depth in range(self.num_phase):
            for segment in np.split(leaves,1<<depth):
                left,right=np.split(segment,2)
                self._angles.append(float(2*np.arctan2(np.sqrt(right.sum()),np.sqrt(left.sum()))))
        self.unary=unary_iteration_kernels(self.num_unary,self.queries+1,
                                          lambda k:[('z',k)],include_adjoint=False)
        self._decoder=PhaseEstimation(encoding)
        self._circuits={}
        if hasattr(encoding,'select_kernel') and hasattr(encoding,'controlled_reflection_kernel'):
            self._select=encoding.select_kernel()
            self._reflection=encoding.controlled_reflection_kernel()
            self._prep=encoding.prepare_kernel();self._undo=encoding.unprepare_kernel()
            self._controlled_step=encoding.controlled_walk_step_kernel()
            self.frame='prepared'
        else:
            self._select=encoding.apply_kernel()
            self._reflection=controlled_reflect_about_zero
            self._prep,self._undo=_empty_frame,_empty_frame
            controlled_block=encoding.controlled_apply_kernel()
            @cudaq.kernel
            def controlled_zero_frame(ca:cudaq.qview,system:cudaq.qview):
                controlled_block(ca,system)
                controlled_reflect_about_zero(ca)
            _retain(controlled_zero_frame)
            self._controlled_step=controlled_zero_frame
            self.frame='zero'

    @property
    def num_work(self):
        return self.num_phase+self.num_unary+self.encoding.num_ancilla

    @property
    def resources(self):
        return dict(select_queries=self.queries,controlled_walk_queries=1,
                    total_select_queries=self.queries+1,
                    unary_toffolis=self.unary.toffoli_count,
                    phase_qubits=self.num_phase,unary_qubits=self.num_unary,
                    control_states=self.num_states,frame=self.frame,
                    control_state_preparation='floating-point rotation tree; synthesis not included',
                    erasure='coherent unary ladder reversal')

    def phase_to_energy(self,index):
        return self._decoder.phase_to_energy(index,self.num_phase)

    def energies_from_counts(self,counts,*,top=None):
        return self._decoder.energies_from_counts(counts,self.num_phase,top=top)

    def control_kernel(self):
        p=self.num_phase;angles=self._angles
        @cudaq.kernel
        def sine_control(control:cudaq.qview):
            prepare(control.front(p),angles)
        _retain(sine_control)
        return sine_control

    def multiplex_kernel(self):
        if hasattr(self,'_multiplex'):return self._multiplex
        p,m,na,q=self.num_phase,self.num_unary,self.encoding.num_ancilla,self.queries
        op=[row[0] for row in self.unary.ops]
        aa=[row[1] for row in self.unary.ops]
        bb=[row[2] for row in self.unary.ops]
        cc=[row[3] for row in self.unary.ops]
        select,reflection,controlled=self._select,self._reflection,self._controlled_step
        @cudaq.kernel
        def select_powers(system:cudaq.qview,work:cudaq.qview):
            # Temporarily route the parity qubit adjacent to the oracle.
            swap(work[0],work[p+m-1])
            controlled(work[p+m-1:p+m+na],system)
            swap(work[0],work[p+m-1])
            for k in range(len(op)):
                if op[k]==0:
                    x(work[1+aa[k]])
                elif op[k]==1:
                    x(work[p+aa[k]])
                elif op[k]==2:
                    x.ctrl(work[1+aa[k]],work[p+bb[k]])
                elif op[k]==3:
                    x.ctrl(work[p+aa[k]],work[p+bb[k]])
                elif op[k]==4:
                    x.ctrl(work[p+aa[k]],work[1+bb[k]],work[p+cc[k]])
                elif op[k]==18:
                    x.ctrl(work[1+aa[k]],work[1+bb[k]])
                elif op[k]==19:
                    x.ctrl(work[1+aa[k]],work[1+bb[k]],work[p+cc[k]])
                elif op[k]==7:
                    if bb[k]>0 and bb[k]<q:
                        x(work[p+m-1])
                    reflection(work[p+m-1:p+m+na])
                    if bb[k]>0 and bb[k]<q:
                        x(work[p+m-1])
                    if bb[k]<q:
                        select(work[p+m:p+m+na],system)
        _retain(select_powers)
        self._multiplex=select_powers
        return select_powers

    def kernel(self,*,state_prep=None,measure=False,return_measurements=False):
        if state_prep is None:
            raise ValueError('supply a gate-based state_prep(system) kernel')
        if return_measurements and not measure:
            raise ValueError('return_measurements requires measure=True')
        key=(id(state_prep),measure,return_measurements)
        if key in self._circuits:return self._circuits[key]
        p,m,na,ns=self.num_phase,self.num_unary,self.encoding.num_ancilla,self.encoding.num_system
        control=self.control_kernel();multiplex=self.multiplex_kernel()
        prep,undo=self._prep,self._undo
        @cudaq.kernel
        def body(system:cudaq.qview,work:cudaq.qview):
            state_prep(system)
            control(work.front(p))
            prep(work[p+m:p+m+na])
            multiplex(system,work)
            undo(work[p+m:p+m+na])
            # primitives.iqft omits reversal swaps and expects its input
            # reversed. Our exponent index 2*t+b is ordinary little-endian.
            for k in range(p//2):
                swap(work[k],work[p-1-k])
            iqft(work.front(p))
        if return_measurements:
            @cudaq.kernel
            def readout()->list[bool]:
                system=cudaq.qvector(ns)
                work=cudaq.qvector(p+m+na)
                body(system,work)
                return mz(work.front(p))
            estimate=readout
        else:
            @cudaq.kernel
            def sample_or_state():
                system=cudaq.qvector(ns)
                work=cudaq.qvector(p+m+na)
                body(system,work)
                if measure:mz(work.front(p), register_name='phase')
            estimate=sample_or_state
        _retain(body,estimate)
        self._circuits[key]=estimate
        return estimate
