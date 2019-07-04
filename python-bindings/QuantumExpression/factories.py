from ._QuantumExpression import PauliExpression
from ._QuantumExpression import FermionExpression


def sigma_x(i):
    return PauliExpression(i, 1)


def sigma_y(i):
    return PauliExpression(i, 2)


def sigma_z(i):
    return PauliExpression(i, 3)


def sigma_plus(i):
    return 0.5 * (sigma_x(i) + 1j * sigma_y(i))


def sigma_minus(i):
    return 0.5 * (sigma_x(i) - 1j * sigma_y(i))


def cr_fermion(i):
    return FermionExpression(i, 1)


def an_fermion(i):
    return FermionExpression(i, 2)


def num_fermion(i):
    return FermionExpression(i, 3)
