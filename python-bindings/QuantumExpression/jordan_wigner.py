from ._QuantumExpression import PauliExpression
from ._QuantumExpression import FermionExpression
from .factories import (
    sigma_x, sigma_y, sigma_z, sigma_plus, sigma_minus,
    cr_fermion, an_fermion, num_fermion, majorana_x, majorana_y
)


def jordan_wigner(expr, N, custom_term_handler=None):
    if isinstance(expr, PauliExpression):
        return jordan_wigner_to_fermions(expr, N, custom_term_handler)
    if isinstance(expr, FermionExpression):
        return jordan_wigner_to_spins(expr, N, custom_term_handler)

    raise TypeError("argument has to be a `QuantumExpression`")


def jordan_wigner_to_fermions(pauli_expr, N, custom_term_handler=None):
    result = 0
    for pauli_term in pauli_expr:
        indices = pauli_term.pauli_string.indices
        types = pauli_term.pauli_string.types

        if types == [3]:
            result += pauli_term.coefficient * (
                2 * num_fermion(indices[0]) - 1
            )
            continue

        if types == [1, 1] and (indices[1] == indices[0] + 1):
            i, j = indices

            result += pauli_term.coefficient * -1j * majorana_y(i) * majorana_x(i + 1)
            continue

        if pauli_term.is_numeric:
            result += pauli_term.coefficient
            continue

        if custom_term_handler is not None:
            result += custom_term_handler(pauli_term)
            continue
        else:
            raise ValueError(f"this pauli-string is not supported: {pauli_term.pauli_string}")

    return result


def jordan_wigner_to_spins(fermion_expr, N, custom_term_handler):
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
            for k in range(i + 1, j):
                z_string *= -sigma_z(k)
            left_sigma = sigma_plus(i) if types[0] == 1 else -sigma_minus(i)
            right_sigma = sigma_plus(j) if types[1] == 1 else sigma_minus(j)

            result += fermion_term.coefficient * left_sigma * z_string * right_sigma
            continue

        if fermion_term.is_numeric:
            result += fermion_term.coefficient
            continue

        if custom_term_handler is not None:
            result += custom_term_handler(fermion_term)
            continue
        else:
            raise ValueError(f"this fermion-string is not supported: {fermion_term.fermion_string}")

    return result
