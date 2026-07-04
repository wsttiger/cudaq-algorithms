# Draft upstream issues for NVIDIA/cuda-quantum

Found while prototyping a pure-Python Pauli LCU library layer
(`experiments/lcu_python/` on the `expt_lcu_python` branch and
`experiments/suzuki_trotter_python/` on `add_suzuki_trotter_python`).
Issues 1-2 are marshaling defects, 3 is a control-composition gap, 4 is a
silent control-flow miscompile. Not yet filed.

Environment for all:
- CUDA-Q built from source, commit `0be565550f4c23affdcbed9e4eaec38d2d0915e6`
- Python 3.11, target `qpp-cpu`, Linux x86_64 (AlmaLinux 8.10)

---

## Issue 1 — `@dataclass` kernel argument with a `list[int]` field fails with `std::bad_cast` when the list contains a negative value

**Title:** Python kernel: dataclass argument with `list[int]` field raises
`std::bad_cast` if the list contains a negative integer

### Description

A `@dataclass` used as a kernel argument works when its `list[int]` fields
contain only non-negative values, but launching the kernel raises
`RuntimeError: std::bad_cast` as soon as any list element is negative. The
same list passed as a bare `list[int]` argument (not wrapped in a dataclass)
works fine with negative values, so the defect is specific to the
struct-member marshaling path (it behaves as if the element type is cast
through an unsigned integer during argument packing).

### Minimal reproducer

```python
from dataclasses import dataclass
import cudaq

cudaq.set_target("qpp-cpu")

@dataclass(slots=True)
class Args:
    values: list[int]

@cudaq.kernel
def kernel(args: Args):
    q = cudaq.qvector(1)
    if args.values[0] < 0:
        x(q[0])

cudaq.sample(kernel, Args([1]))    # OK
cudaq.sample(kernel, Args([-1]))   # RuntimeError: std::bad_cast

# Control: the same data as a bare argument works.
@cudaq.kernel
def bare(values: list[int]):
    q = cudaq.qvector(1)
    if values[0] < 0:
        x(q[0])

cudaq.sample(bare, [-1])           # OK -> most_probable() == '1'
```

### Observed behavior matrix

| Case | Result |
|---|---|
| dataclass, scalar fields only | OK |
| dataclass, `list[int]`/`list[float]` fields, non-negative values | OK (tested 1–5 list fields, mixed lengths) |
| dataclass, any `list[int]` field containing a negative value | `std::bad_cast` |
| bare `list[int]` argument containing negative values | OK |
| captured dataclass (not argument) with negative list element | `std::bad_cast` (same failure) |

### Why it matters

Any struct that carries signs — e.g. the sign array of an LCU
decomposition, exponents, offsets — cannot be passed as an aggregated
kernel argument; libraries are forced to thread many parallel flat lists
through every kernel signature instead of one args object.

### Expected

Negative integers in a dataclass `list[int]` field marshal identically to
negative integers in a bare `list[int]` argument.

---

## Issue 2 — Captured empty list in a Python kernel fails with "Cannot infer runtime argument type"

**Title:** Python kernel: capturing an empty list raises
"Cannot infer runtime argument type" (annotated empty-list *arguments* work)

### Description

A kernel that captures a `list` from its enclosing scope fails to launch
when that list is empty, with `RuntimeError: Cannot infer runtime argument
type`. The element type is inferred from the values, and an empty list has
none — but the same empty list passed as an **annotated argument**
(`values: list[int]`) works, so the type information exists in the
annotated case and only the capture path lacks a fallback.

### Minimal reproducer

```python
import cudaq

cudaq.set_target("qpp-cpu")

# Annotated argument: empty list is fine.
@cudaq.kernel
def with_arg(values: list[int]):
    q = cudaq.qvector(1)
    if len(values) > 0:
        x(q[0])

cudaq.sample(with_arg, [])      # OK

# Captured variable: empty list fails.
captured = []

@cudaq.kernel
def with_capture():
    q = cudaq.qvector(1)
    if len(captured) > 0:
        x(q[0])

cudaq.sample(with_capture)      # RuntimeError: Cannot infer runtime argument type
```

### Why it matters

The natural "kernel factory" pattern — a host function that flattens data
and returns a `@cudaq.kernel` closure capturing the arrays — breaks exactly
at the boundary cases where an array is legitimately empty (e.g. a
zero-ancilla register's control-pattern array in an LCU library layer). The
factory must special-case empty data even though the kernel body handles the
empty case correctly.

### Expected / suggested

Either (a) allow an element-type fallback for captured empty lists (e.g.
from a type annotation on the captured variable, or defaulting when the list
is provably unused), or at minimum (b) a clearer diagnostic naming the
captured variable, since "Cannot infer runtime argument type" gives no hint
that a captured empty list is the cause.

---

## Issue 3 (limitation report / feature request) — controlled composite kernels

**Title:** Python kernel: no way to control an operation on a mixed
qubit + qview control set; `cudaq.control` fails on kernels that call
other kernels

### Description

Two related gaps make "one external control qubit + an ancilla register"
control sets — the standard shape for controlled SELECT / controlled
qubitization walks — awkward in Python kernels:

1. Gate-level: `x.ctrl(control_qubit, ancilla_view, target)` (any mix of a
   bare qubit and a `qview` in one control set) is rejected with
   `invalid argument type for control operand`. All-qubit and single-view
   control sets work.
2. Kernel-level: `cudaq.control(kernel, control, args...)` works for leaf
   kernels (including list arguments), but fails with
   `Could not successfully apply kernel specialization` when the controlled
   kernel itself calls another kernel — so composite operations (a walk
   step built from SELECT + reflections) cannot be controlled wholesale.

### Minimal reproducers

```python
import cudaq
cudaq.set_target("qpp-cpu")

# (1) mixed control set
@cudaq.kernel
def mixed():
    c = cudaq.qubit()
    q = cudaq.qvector(3)
    x.ctrl(c, q.front(2), q[2])   # error: invalid argument type for control operand

# (2) cudaq.control of a composite kernel
@cudaq.kernel
def leaf(q: cudaq.qview):
    for i in range(q.size()):
        x(q[i])

@cudaq.kernel
def composite(q: cudaq.qview):
    leaf(q)

@cudaq.kernel
def caller():
    c = cudaq.qubit()
    q = cudaq.qvector(2)
    x(c)
    cudaq.control(leaf, c, q)       # OK
    cudaq.control(composite, c, q)  # RuntimeError: Could not successfully
                                    # apply kernel specialization.
```

### Workaround / impact

Libraries can restructure so the control qubit shares one register with the
ancillas and slice views (`register.front(...)`/`register.back(...)`), but
that leaks into the public API: users must co-allocate their control qubit
with the library's ancilla register instead of passing any qubit they own.
Either fixing (1) or (2) would remove the constraint; (2) alone would allow
`cudaq.control(walk_step, ctrl, ...)` over composite operations.

## Issue 4 — `return` inside a Python kernel is silently ignored

**Title:** Python kernel: early `return` statements are silently ignored —
gates after a guard execute unconditionally

### Description

An `if condition: return` guard inside a `@cudaq.kernel` compiles without
warning but has no effect: the operations after it execute regardless of the
condition (and regardless of whether the condition is a kernel argument or a
constant). This is the most severe issue of this set because it produces
**silently wrong circuits** rather than an error.

It is also easy to miss in testing: guards whose "guarded" code consists of
loops over empty/zero-trip ranges appear to work (the loops do nothing for
exactly the inputs the guard was for), so a kernel can carry dead guards
through an entire test suite until one guarded path contains unconditional
gates. That is exactly how we found it — a C++-mirroring device kernel with
`if steps == 0: return` (masked by `for _ in range(steps)`) and
`if order not in (1, 2, 4): return` (NOT masked: an unsupported order
executed a wrong product formula instead of the documented no-op).

### Minimal reproducer

```python
import cudaq
cudaq.set_target("qpp-cpu")

@cudaq.kernel
def guarded(skip: int):
    q = cudaq.qvector(1)
    if skip == 1:
        return
    x(q[0])

print(cudaq.sample(guarded, 0))  # { 1:1000 }  (expected)
print(cudaq.sample(guarded, 1))  # { 1:1000 }  (expected { 0:1000 })
```

Also reproduces with multi-clause conditions
(`if a != 1 and a != 2: return`) and with multiple sequential guards.

### Expected / suggested

Either honor `return` control flow in void kernels, or — if early return is
intentionally unsupported — make the AST bridge reject it with a compile
error. Silent acceptance is the worst of both: the C++ device-kernel idiom
"invalid runtime inputs are no-ops via early return" ports over as a
silently wrong circuit.

### Workaround

Restructure kernel bodies as a single positively-guarded if-block:

```python
    valid = steps > 0 and order in_supported
    if valid:
        <entire body>
```

## Suggested labels

All: `python`, `kernel-builder`; issues 1, 2, and 4 `bug` (4 is the most severe — silent wrong circuits), issue 3 arguably enhancement. Issue 1 is the higher-impact one
(it blocks aggregated kernel-argument types outright); Issue 2 has an easy
workaround once diagnosed, so the diagnostic improvement alone would help.
