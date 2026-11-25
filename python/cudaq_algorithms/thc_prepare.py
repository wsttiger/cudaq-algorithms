# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""THC triangular-domain preparation and coherent pair alias sampling.

Pairs are zero based: 0<=mu<=nu<M, or nu=M and 0<=mu<n.
The contiguous index is nu*(nu+1)//2+mu. A condition-controlled
amplitude amplification prepares the valid domain without a table of
domain amplitudes. Alias preprocessing uses d valid bins, not 2**m.
All scratch is caller owned and reusable between preparation and SELECT.
"""
from dataclasses import dataclass
from functools import cached_property
import math
import cudaq
import numpy as np
from .primitives import QROM,cmp_ge_register,cmp_ge_constant,add_register,subtract_register,add_constant,subtract_constant
from .primitives._alias_sampling import _vose_tables
from .primitives._unary_iteration import _retain


@cudaq.kernel
def _match(register:cudaq.qview,value:int):
    for bit in range(register.size()):
        if ((value>>bit)&1)==0:x(register[bit])


@cudaq.kernel
def _minus_one(q:cudaq.qubit):
    x(q);z(q);x(q);z(q)


@dataclass(frozen=True)
class THCPairDomain:
    num_spatial: int
    rank: int

    def __post_init__(self):
        if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in (self.num_spatial,self.rank)):
            raise ValueError('positive integer spatial count and THC rank required')

    @property
    def pair_width(self):return (max(self.rank+1,self.num_spatial)-1).bit_length()

    @property
    def pairs(self):
        return tuple((mu,nu) for nu in range(self.rank) for mu in range(nu+1))+tuple(
            (mu,self.rank) for mu in range(self.num_spatial))

    @property
    def num_valid(self):return self.rank*(self.rank+1)//2+self.num_spatial

    @property
    def num_scratch(self):
        return 2 if (self.rank,self.num_spatial)==(1,2) else self.pair_width+7

    @property
    def amplification(self):
        p=self.num_valid/(1<<(2*self.pair_width))
        rounds=max(0,math.ceil(math.pi/(4*math.asin(math.sqrt(p)))-.5-1e-14))
        success=math.sin(math.pi/(4*rounds+2))**2
        return rounds,2*math.asin(math.sqrt(min(1.,success/p)))

    @cached_property
    def _predicate(self):
        k,n,M=self.pair_width,self.num_spatial,self.rank
        if (M,n)==(1,2):
            @cudaq.kernel
            def valid_pair(pair:cudaq.qview,scratch:cudaq.qview):
                x(pair[1])
                x.ctrl(pair[0],pair[1],scratch[1])
                x(pair[1])
                x(scratch[1])
            _retain(valid_pair)
            return valid_pair
        bitsM=[((1<<k)-M)>>i&1 for i in range(k)]
        bitsN=[((1<<k)-n)>>i&1 for i in range(k)]
        order,large,outside,sentinel=k+3,k+4,k+5,k+6
        @cudaq.kernel
        def valid_pair(pair:cudaq.qview,scratch:cudaq.qview):
            cmp_ge_register(pair[k:2*k],pair.front(k),scratch[k+2:k+3],scratch[order:order+1])
            cmp_ge_constant(pair[k:2*k],bitsM,0,scratch[2:k+2],scratch[k+2:k+3],scratch[large:large+1])
            cmp_ge_constant(pair.front(k),bitsN,0,scratch[2:k+2],scratch[k+2:k+3],scratch[outside:outside+1])
            _match(pair[k:2*k],M)
            x.ctrl(pair[k:2*k],scratch[sentinel])
            _match(pair[k:2*k],M)
            x(scratch[large])
            x.ctrl(scratch[order],scratch[large],scratch[1])
            x(scratch[large])
            x(scratch[outside])
            x.ctrl(scratch[sentinel],scratch[outside],scratch[1])
            x(scratch[outside])
            _match(pair[k:2*k],M)
            x.ctrl(pair[k:2*k],scratch[sentinel])
            _match(pair[k:2*k],M)
            cmp_ge_constant(pair.front(k),bitsN,0,scratch[2:k+2],scratch[k+2:k+3],scratch[outside:outside+1])
            cmp_ge_constant(pair[k:2*k],bitsM,0,scratch[2:k+2],scratch[k+2:k+3],scratch[large:large+1])
            cmp_ge_register(pair[k:2*k],pair.front(k),scratch[k+2:k+3],scratch[order:order+1])
        _retain(valid_pair)
        return valid_pair

    def _build(self,adjoint):
        k=self.pair_width;rounds,angle=self.amplification;valid=self._predicate
        @cudaq.kernel
        def base(pair:cudaq.qview,scratch:cudaq.qview):
            for bit in range(2*k):h(pair[bit])
            ry(angle,scratch[0])
        @cudaq.kernel
        def inverse_base(pair:cudaq.qview,scratch:cudaq.qview):
            ry(-angle,scratch[0])
            for bit in range(2*k):h(pair[bit])
        @cudaq.kernel
        def zero_reflection(pair:cudaq.qview,scratch:cudaq.qview):
            for bit in range(2*k):x(pair[bit])
            x(scratch[0])
            z.ctrl(pair,scratch[0])
            x(scratch[0])
            for bit in range(2*k):x(pair[bit])
        @cudaq.kernel
        def success_reflection(pair:cudaq.qview,scratch:cudaq.qview):
            valid(pair,scratch)
            z.ctrl(scratch[1],scratch[0])
            valid(pair,scratch)
        if not adjoint:
            @cudaq.kernel
            def domain_prepare(pair:cudaq.qview,scratch:cudaq.qview):
                base(pair,scratch)
                for iteration in range(rounds):
                    success_reflection(pair,scratch)
                    inverse_base(pair,scratch)
                    zero_reflection(pair,scratch)
                    base(pair,scratch)
                if rounds%2==1:_minus_one(scratch[0])
                x(scratch[0])
            result=domain_prepare
        else:
            @cudaq.kernel
            def domain_unprepare(pair:cudaq.qview,scratch:cudaq.qview):
                x(scratch[0])
                if rounds%2==1:_minus_one(scratch[0])
                for iteration in range(rounds):
                    inverse_base(pair,scratch)
                    zero_reflection(pair,scratch)
                    base(pair,scratch)
                    success_reflection(pair,scratch)
                inverse_base(pair,scratch)
            result=domain_unprepare
        _retain(base,inverse_base,zero_reflection,success_reflection,result)
        return result

    @cached_property
    def _prepare(self):return self._build(False)
    @cached_property
    def _inverse(self):return self._build(True)
    def kernel(self):return self._prepare
    def adjoint_kernel(self):return self._inverse


@dataclass(frozen=True)
class TriangularIndex:
    """XOR nu*(nu+1)//2+mu into an output, modulo its width.

    Generic schoolbook arithmetic takes O(k**2) gates and O(k) scratch.
    The one-bit specialization uses two CNOTs and one Toffoli.
    Pair=(mu[k],nu[k]); output width is at most 2*k.
    """
    pair_width:int
    output_width:int

    def __post_init__(self):
        if not 1<=self.output_width<=2*self.pair_width:raise ValueError('invalid index widths')

    @property
    def num_scratch(self):return 0 if self.pair_width==1 else 5*self.pair_width+4

    @cached_property
    def _kernel(self):
        k,m=self.pair_width,self.output_width
        if k==1:
            @cudaq.kernel
            def index(pair:cudaq.qview,output:cudaq.qview,scratch:cudaq.qview):
                x.ctrl(pair[0],output[0])
                x.ctrl(pair[1],output[0])
                if m>1:x.ctrl(pair[0],pair[1],output[1])
            _retain(index)
            return index
        width=2*k+1;po=k+1;mask=po+width;carry=mask+width
        one=[1]+[0]*k
        @cudaq.kernel
        def index(pair:cudaq.qview,output:cudaq.qview,scratch:cudaq.qview):
            for bit in range(k):x.ctrl(pair[k+bit],scratch[bit])
            add_constant(scratch.front(k+1),one,scratch[mask:mask+k+1],scratch[carry:carry+1])
            for i in range(k):
                for j in range(k+1):x.ctrl(pair[k+i],scratch[j],scratch[mask+i+j])
                add_register(scratch[mask:mask+width],scratch[po:po+width],scratch[carry:carry+1])
                for j in range(k+1):x.ctrl(pair[k+i],scratch[j],scratch[mask+i+j])
            for bit in range(k):x.ctrl(pair[bit],scratch[mask+bit])
            add_register(scratch[mask:mask+2*k],scratch[po+1:po+width],scratch[carry:carry+1])
            for bit in range(m):x.ctrl(scratch[po+1+bit],output[bit])
            subtract_register(scratch[mask:mask+2*k],scratch[po+1:po+width],scratch[carry:carry+1])
            for bit in range(k):x.ctrl(pair[bit],scratch[mask+bit])
            for step in range(k):
                i=k-1-step
                for j in range(k+1):x.ctrl(pair[k+i],scratch[j],scratch[mask+i+j])
                subtract_register(scratch[mask:mask+width],scratch[po:po+width],scratch[carry:carry+1])
                for j in range(k+1):x.ctrl(pair[k+i],scratch[j],scratch[mask+i+j])
            subtract_constant(scratch.front(k+1),one,scratch[mask:mask+k+1],scratch[carry:carry+1])
            for bit in range(k):x.ctrl(pair[k+bit],scratch[bit])
        _retain(index)
        return index

    def kernel(self):return self._kernel


class THCPairPrepare:
    """Alias PREPARE directly on the valid mu/nu pair registers.

    Inputs are signed spin-summed one-body row coefficients and symmetric
    Z. Diagonal pair weights are Z_mm/2, off-diagonal weights Z_mn.
    Signature: (pair[2*k], garbage[2*k+2*b+3], scratch[num_scratch]).
    Garbage: alternate pair, keep, sign, alternate sign, coin, comparator.
    Scratch is clean at the boundary. Angle precision of the analytic
    amplitude-amplification rotation is floating point; synthesis is a
    separately declared preparation cost, not a free FT rotation.
    """
    def __init__(self,one_body,z,keep_bits=2,*,qrom_variant='select',block_size=None):
        if np.iscomplexobj(one_body) or np.iscomplexobj(z):
            raise ValueError('THC coefficients must be real')
        one_body=np.asarray(one_body,float);z=np.asarray(z,float)
        if (one_body.ndim!=1 or not len(one_body) or z.ndim!=2 or z.shape[0]!=z.shape[1]
                or not len(z) or not np.isfinite(one_body).all() or not np.isfinite(z).all()
                or not np.allclose(z,z.T,rtol=0,atol=1e-12)):
            raise ValueError('finite one-body coefficients and symmetric Z required')
        if isinstance(keep_bits,bool) or int(keep_bits)!=keep_bits or keep_bits<1:
            raise ValueError('positive integer keep_bits required')
        self.domain=THCPairDomain(len(one_body),len(z));self.keep_bits=int(keep_bits)
        self.pairs=self.domain.pairs;self.pair_width=self.domain.pair_width
        weights=[z[mu,nu]*(.5 if mu==nu else 1.) if nu<len(z) else one_body[mu]
                 for mu,nu in self.pairs]
        self.weights=np.array(weights);self.alpha=float(abs(self.weights).sum())
        if self.alpha<=0:raise ValueError('positive LCU normalization required')
        self.probabilities=abs(self.weights)/self.alpha
        self.keep,self.alternate,rounded=_vose_tables(self.probabilities,self.keep_bits)
        self.table_probabilities=np.array(rounded)
        k,b=self.pair_width,self.keep_bits;d=len(weights)
        self.num_index=max(1,(d-1).bit_length())
        self.index=TriangularIndex(k,self.num_index)
        signs=(self.weights<0).astype(int)
        data=[]
        for i in range(d):
            mu,nu=self.pairs[self.alternate[i]]
            data.append(mu|(nu<<k)|(self.keep[i]<<(2*k))|
                        (int(signs[i])<<(2*k+b))|
                        (int(signs[self.alternate[i]])<<(2*k+b+1)))
        self.qrom=QROM(data,self.num_index,2*k+b+2,variant=qrom_variant,block_size=block_size)
        self.num_garbage=2*k+2*b+3
        self.num_scratch=max(self.domain.num_scratch,
            self.num_index+self.index.num_scratch+self.qrom.num_ladder,1)
        self.uniform_nonnegative=bool(np.all(self.weights>=0) and
            np.array_equal(self.table_probabilities,np.full(d,1/d)))

    @property
    def discretization_bound(self):return 1.5/(len(self.pairs)*(1<<self.keep_bits))

    def _build(self,adjoint):
        k,b,m=self.pair_width,self.keep_bits,self.num_index
        dw=self.domain.num_scratch;ti=self.index.num_scratch;lad=self.qrom.num_ladder
        pre=self.domain.kernel();undo=self.domain.adjoint_kernel()
        index=self.index.kernel();lookup=self.qrom.kernel
        output=2*k+b+2;coin=output;flag=coin+b
        sign=2*k+b
        @cudaq.kernel
        def table(pair:cudaq.qview,garbage:cudaq.qview,scratch:cudaq.qview):
            index(pair,scratch.front(m),scratch[m:m+ti])
            lookup(scratch.front(m),scratch[m+ti:m+ti+lad],garbage.front(output))
            index(pair,scratch.front(m),scratch[m:m+ti])
        @cudaq.kernel
        def compare(pair:cudaq.qview,garbage:cudaq.qview,scratch:cudaq.qview):
            cmp_ge_register(garbage[coin:coin+b],garbage[2*k:2*k+b],scratch.back(1),garbage[flag:flag+1])
        @cudaq.kernel
        def swap_pair(pair:cudaq.qview,garbage:cudaq.qview):
            for bit in range(2*k):swap.ctrl(garbage[flag],pair[bit],garbage[bit])
            swap.ctrl(garbage[flag],garbage[sign],garbage[sign+1])
        if not adjoint:
            @cudaq.kernel
            def pair_prepare(pair:cudaq.qview,garbage:cudaq.qview,scratch:cudaq.qview):
                pre(pair,scratch.front(dw))
                table(pair,garbage,scratch)
                for bit in range(b):h(garbage[coin+bit])
                compare(pair,garbage,scratch)
                swap_pair(pair,garbage)
            result=pair_prepare
        else:
            @cudaq.kernel
            def pair_unprepare(pair:cudaq.qview,garbage:cudaq.qview,scratch:cudaq.qview):
                swap_pair(pair,garbage)
                compare(pair,garbage,scratch)
                for bit in range(b):h(garbage[coin+bit])
                table(pair,garbage,scratch)
                undo(pair,scratch.front(dw))
            result=pair_unprepare
        _retain(table,compare,swap_pair,result)
        return result

    @cached_property
    def _prepare(self):return self._build(False)
    @cached_property
    def _inverse(self):return self._build(True)
    def kernel(self):return self._prepare
    def adjoint_kernel(self):return self._inverse
