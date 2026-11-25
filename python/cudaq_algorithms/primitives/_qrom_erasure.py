# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measurement-assisted QROM erasure with block phase correction.

For an output known to equal f(address), X-basis measurement gives y and
the phase (-1)**dot(y,f(address)). A block of k Fourier flags, selected
coherently by the low address, lets one high-address unary walk correct
that phase. The flags are then restored. The cost is W(high,ceil(N/k))
+2*W(low,k), independent of the output word width; W is the actual
coherent unary ladder price. This is not an inverse on arbitrary XOR
targets. It uses measurements and classical feed-forward and cannot be
wrapped in cudaq.control or cudaq.adjoint.
"""
from dataclasses import dataclass
from functools import cached_property
import math
from numbers import Integral
import cudaq
from ._unary_iteration import unary_iteration_kernels, _walk_toffoli_count, _retain


@dataclass(frozen=True)
class QROMErase:
    data: tuple
    address_bits: int
    output_bits: int
    block_size: int | None = None

    def __post_init__(self):
        raw=tuple(self.data)
        if any(isinstance(v,bool) or not isinstance(v,Integral) for v in raw):
            raise ValueError('QROM erasure words must be integers')
        data=tuple(int(v) for v in raw)
        m,b=self.address_bits,self.output_bits
        if (isinstance(m,bool) or not isinstance(m,int) or m<1 or
                isinstance(b,bool) or not isinstance(b,int) or b<1 or
                not data or len(data)>1<<m or min(data)<0 or max(data)>=1<<b):
            raise ValueError('invalid QROM erasure table or widths')
        def price(block):
            low=block.bit_length()-1;high=m-low
            return _walk_toffoli_count(high,math.ceil(len(data)/block)) + (
                2*_walk_toffoli_count(low,block) if low else 0)
        block=self.block_size
        if block is None:block=min((1<<l for l in range(m)),key=price)
        if (isinstance(block,bool) or not isinstance(block,int) or
                block<1 or block>1<<(m-1) or block&(block-1)):
            raise ValueError('block_size must be a power of two in [1,2**(address_bits-1)]')
        object.__setattr__(self,'data',data)
        object.__setattr__(self,'block_size',block)

    @property
    def num_scratch(self):return self.block_size+self.address_bits

    @property
    def toffoli_count(self):
        low=self.block_size.bit_length()-1;high=self.address_bits-low
        return _walk_toffoli_count(high,math.ceil(len(self.data)/self.block_size))+(
            2*_walk_toffoli_count(low,self.block_size) if low else 0)

    @cached_property
    def _correction(self):
        m,width,block=self.address_bits,self.output_bits,self.block_size
        low=block.bit_length()-1;high=m-low
        count=math.ceil(len(self.data)/block)
        high_walk=unary_iteration_kernels(high,count,
            lambda h:[('x',h*block+l) for l in range(block)],include_adjoint=False)
        op=[r[0] for r in high_walk.ops];aa=[r[1] for r in high_walk.ops]
        bb=[r[2] for r in high_walk.ops];cc=[r[3] for r in high_walk.ops]
        words=self.data+(0,)*(count*block-len(self.data))
        bits=[(word>>bit)&1 for word in words for bit in range(width)]
        if low:
            low_walk=unary_iteration_kernels(low,block,lambda k:[('z',k)],include_adjoint=False)
            choose=low_walk.kernel
        else:
            @cudaq.kernel
            def choose(address:cudaq.qview,ladder:cudaq.qview,target:cudaq.qview):
                z(target[0])
        @cudaq.kernel
        def correct(address:cudaq.qview,scratch:cudaq.qview,measured:list[bool]):
            for k in range(block):h(scratch[k])
            choose(address.front(low),scratch[block+high:block+m],scratch.front(block))
            for k in range(len(op)):
                if op[k]==0:
                    x(address[low+aa[k]])
                elif op[k]==1:
                    x(scratch[block+aa[k]])
                elif op[k]==2:
                    x.ctrl(address[low+aa[k]],scratch[block+bb[k]])
                elif op[k]==3:
                    x.ctrl(scratch[block+aa[k]],scratch[block+bb[k]])
                elif op[k]==4:
                    x.ctrl(scratch[block+aa[k]],address[low+bb[k]],scratch[block+cc[k]])
                elif op[k]==18:
                    x.ctrl(address[low+aa[k]],address[low+bb[k]])
                elif op[k]==19:
                    x.ctrl(address[low+aa[k]],address[low+bb[k]],scratch[block+cc[k]])
                elif op[k]==5:
                    parity=0
                    for bit in range(width):
                        if bits[bb[k]*width+bit]==1:
                            if measured[bit]:parity=1-parity
                    if parity==1:
                        x.ctrl(scratch[block+aa[k]],scratch[bb[k]%block])
            choose(address.front(low),scratch[block+high:block+m],scratch.front(block))
            for k in range(block):h(scratch[k])
        _retain(choose,correct)
        return correct

    def correction_kernel(self):
        """``(address, scratch, measured: list[bool])`` phase fixup only."""
        return self._correction

    @cached_property
    def _erase(self):
        width=self.output_bits;correct=self._correction
        @cudaq.kernel
        def erase(address:cudaq.qview,output:cudaq.qview,scratch:cudaq.qview):
            for bit in range(width):h(output[bit])
            measured=mz(output)
            for bit in range(width):
                if measured[bit]:x(output[bit])
            correct(address,scratch,measured)
        _retain(erase)
        return erase

    def kernel(self):return self._erase
