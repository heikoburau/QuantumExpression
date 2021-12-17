QuantumExpression
=================

Evaluates arbitrary expressions of spin-1/2 (Pauli) operators and spinless fermions in a physicists-friendly way.
`QuantumExpression` is a C++ library with a python interface.
Implemented with strong focus on high performance of operator manipulations, e.g. multiplication, by utilizing optimized hash tables.

Features
========

- Natural syntax for composing arbitrary operators like Hamiltonians or observables.
- Convertable to dense or sparse matrices.
- Superior memory efficiency compared to sparse matrices for quasi-local operators.
- Specific functions like unitrary rotation, hermitian conjugation, partial trace, etc. included.


Installation
============

```bash
cd python-bindings
python setup.py install
```

Example
=======

```python
from QuantumExpression import (
    sigma_plus, sigma_minus, sigma_z,
    commutator, frobenius_norm
)
import numpy as np

L = 10

H_0 = -1 / 2 * sum(
    (sigma_plus(i) * sigma_minus(i + 1)) +
    (sigma_plus(i) * sigma_minus(i + 1)).dagger
    for i in range(L - 1)
) + sum(
    sigma_z(i) for i in range(L)
)

V = 0.1 * sum(
    sigma_z(i) * sigma_z(i + 1)
    for i in range(L - 1)
)

H = H_0 + V

print(frobenius_norm(commutator(H_0, H), L))

H_m = H.matrix(L)
print(np.trace(H_m @ H_m.T.conj()))
```
