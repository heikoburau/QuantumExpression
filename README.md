QuantumExpression
=================

Evaluating expressions of spin-1/2- and spinless-fermion operators.
`QuantumExpression` is a C++ library with a python interface.

Installation
============

```bash
cd python-bindings
python setup.py install
```

Example
=======

```python
from QuantumExpression import FermionExpression as fe
from QuantumExpression import commutator, frobenius_norm

L = 100

# coding: 1 -> creation, 2 -> annihilation, 3 -> number
H_0 = -1 / 2  * sum(
    (fe({i: 1, i + 1: 2}) + fe({i: 2, i + 1: 1})) for i in range(L - 1)
) + sum(
    fe({i: 3}) for i in range(L)
)

V = 0.1 * sum(
    fe({i: 3}) * fe({i + 1: 3})
    for i in range(L - 1)
)

H = H_0 + V

print(frobenius_norm(commutator(H_0, H)))

```
