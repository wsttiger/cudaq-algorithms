# Draft upstream issues for NVIDIA/cuda-quantum

Found while prototyping a pure-Python Pauli LCU library layer
(`experiments/lcu_python/` on the `expt_lcu_python` branch). Both are
Python kernel argument/capture marshaling defects. Not yet filed.

Environment for both:
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

## Suggested labels

Both: `python`, `bug`, `kernel-builder`. Issue 1 is the higher-impact one
(it blocks aggregated kernel-argument types outright); Issue 2 has an easy
workaround once diagnosed, so the diagnostic improvement alone would help.
