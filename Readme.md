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
from QuantumExpression import cr_fermion, an_fermion, num_fermion
from QuantumExpression import commutator, frobenius_norm

L = 10

H_0 = -1 / 2 * sum(
    (cr_fermion(i) * an_fermion(i + 1)) + 
    (cr_fermion(i) * an_fermion(i + 1)).dagger
    for i in range(L - 1)
) + sum(
    num_fermion(i) for i in range(L)
)

V = 0.1 * sum(
    num_fermion(i) * num_fermion(i + 1)
    for i in range(L - 1)
)

H = H_0 + V

print(frobenius_norm(commutator(H_0, H), L))
```
