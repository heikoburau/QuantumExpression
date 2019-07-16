from ._QuantumExpression import PauliExpression
from ._QuantumExpression import FermionExpression
from .factories import (
    sigma_x, sigma_y, sigma_z, sigma_plus, sigma_minus,
    cr_fermion, an_fermion, num_fermion
)


def jordan_wigner(expr, N):
    if isinstance(expr, PauliExpression):
        return jordan_wigner_to_fermions(expr, N)
    if isinstance(expr, FermionExpression):
        return jordan_wigner_to_spins(expr, N)

    raise TypeError("argument has to be a `QuantumExpression`")


def jordan_wigner_to_fermions(pauli_expr, N):
    result = 0
    for pauli_term in pauli_expr:
        indices = pauli_term.pauli_string.indices
        types = pauli_term.pauli_string.types

        if types == [3]:
            result += pauli_term.coefficient * (
                2 * num_fermion(indices[0]) - 1
            )
            continue

        if types == [1, 1] and (
            indices[1] == indices[0] + 1 or
            indices == [0, N - 1]
        ):
            i, j = indices
            if indices == [0, N - 1]:
                i, j = j, i

            result += pauli_term.coefficient * (
                cr_fermion(i) * cr_fermion(j) +
                cr_fermion(i) * an_fermion(j) -
                an_fermion(i) * cr_fermion(j) -
                an_fermion(i) * an_fermion(j)
            )
            continue

        if pauli_term.is_numeric:
            result += pauli_term.coefficient
            continue

        raise ValueError(f"this pauli-string is not supported: {pauli_term.pauli_string}")

    return result


def jordan_wigner_to_spins(fermion_expr, N):
    result = 0

    for fermion_term in fermion_expr:
        indices = fermion_term.fermion_string.indices
        types = fermion_term.fermion_string.types

        if types == [3]:
            result += fermion_term.coefficient * 0.5 * (
                sigma_z(indices[0]) + 1
            )
            continue

        if len(types) == 2 and not any(t == 3 for t in types):
            i, j = indices
            z_string = 1
            if indices == [0, N - 1]:
                left_sigma = -sigma_plus(j) if types[1] == 1 else sigma_minus(j)
                right_sigma = sigma_plus(i) if types[0] == 1 else sigma_minus(i)
            else:
                for k in range(i + 1, j):
                    z_string *= -sigma_z(k)
                left_sigma = sigma_plus(i) if types[0] == 1 else -sigma_minus(i)
                right_sigma = sigma_plus(j) if types[1] == 1 else sigma_minus(j)

            result += fermion_term.coefficient * left_sigma * z_string * right_sigma
            continue

        if fermion_term.is_numeric:
            result += fermion_term.coefficient
            continue

        raise ValueError(f"this fermion-string is not supported: {fermion_term.fermion_string}")

    return result


def exchange_expr(expr, op_map):
    result = 0

    for term in expr:
        string = 1
        for i, t in term.quantum_string:
            string *= op_map[t](i)

        result += term.coefficient * string

    return result


def exchange_xz_forth(pauli_expr):
    op_map = {1: lambda i: -sigma_z(i), 2: sigma_y, 3: sigma_x}
    return exchange_expr(pauli_expr, op_map)


def exchange_xz_back(pauli_expr):
    op_map = {1: sigma_z, 2: sigma_y, 3: lambda i: -sigma_x(i)}
    return exchange_expr(pauli_expr, op_map)
