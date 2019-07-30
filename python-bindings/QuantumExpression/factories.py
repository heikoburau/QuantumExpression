from ._QuantumExpression import PauliExpression
from ._QuantumExpression import FermionExpression

import numpy as np
from itertools import product
import math


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


def majorana_x(i):
    return cr_fermion(i) + an_fermion(i)


def majorana_y(i):
    return 1j * (cr_fermion(i) - an_fermion(i))


def from_fermion_matrix(matrix, threshold=None):
    N = int(math.log2(matrix.shape[0]))

    result = 0
    for op_codes in product(range(4), repeat=N):
        op = 1
        for i, code in enumerate(op_codes):
            if code == 0:
                op *= cr_fermion(i)
            elif code == 1:
                op *= an_fermion(i)
            elif code == 2:
                op *= num_fermion(i)
            elif code == 3:
                op *= 1 - num_fermion(i)
                
        term = complex(np.trace(matrix @ op.matrix(N).T.conj())) * op
        if threshold is None or abs(term) > threshold:
            result += term

    return result
