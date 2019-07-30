import numpy as np
from .factories import (
    cr_fermion, an_fermion, num_fermion
)


# def hopping_matrix(fermion_expr, N, with_pairings=False):
#     """
#     H = (c_1_dagger, ... , c_N_dagger, c_N, ..., c_1) * ( )_nm * (c_1, ... , c_N, c_N_dagger, ..., c_1_dagger)^T
#     """

#     A = np.zeros((N, N), dtype=complex)
#     if with_pairings:
#         B = np.zeros((N, N), dtype=complex)
#         C = np.zeros((N, N), dtype=complex)

#     for term in fermion_expr:
#         indices = term.fermion_string.indices
#         types = term.fermion_string.types

#         if types == [3]:
#             i = indices[0]
#             A[i, i] = term.coefficient
#             continue

#         if len(types) == 2 and not any(t == 3 for t in types):
#             i, j = indices
#             coefficient = term.coefficient

#             if types[0] != types[1]:
#                 if types[0] == 2 and types[1] == 1:
#                     i, j = j, i
#                     coefficient = -coefficient

#                 A[i, j] = coefficient
#             elif types[0] == 1 and types[1] == 1:
#                 B[i, j] += coefficient
#                 B[j, i] += -coefficient
#             elif types[0] == 2 and types[1] == 2:
#                 C[i, j] += coefficient
#                 C[j, i] += -coefficient
#             continue

#         if term.is_numeric:
#             continue

#         raise ValueError(f"this term is not supported: {term}")

#     if with_pairings:
#         return 1 / 2 * np.block([
#             [A, B[:, ::-1]],
#             [C.conj()[::-1, :], -A.conj()[::-1, ::-1]]
#         ])
#     else:
#         return A


def hopping_matrix(fermion_expr, N):
    bar = lambda i: 2 * N - 1 - i

    result = np.zeros((2 * N, 2 * N), dtype=complex)
    for term in fermion_expr:
        indices = term.fermion_string.indices
        types = term.fermion_string.types

        if types == [3]:
            i = indices[0]
            result[i, i] += term.coefficient / 2
            result[bar(i), bar(i)] += -term.coefficient / 2
            continue

        if len(types) == 2 and not any(t == 3 for t in types):
            i, j = indices

            i = bar(i) if types[0] == 2 else i
            j = bar(j) if types[1] == 1 else j

            result[i, j] += term.coefficient / 2
            result[bar(j), bar(i)] += -term.coefficient / 2
            continue

        if term.is_numeric:
            continue

        raise ValueError(f"this term is not supported: {term}")

    return result


def expr_from_hopping_matrix(hopping_matrix, with_pairings=False):
    assert hopping_matrix.shape[0] == hopping_matrix.shape[1], "hopping matrix must be squared"

    N = hopping_matrix.shape[0] // 2 if with_pairings else hopping_matrix.shape[0]

    result = 0
    for i, j in np.ndindex(*hopping_matrix.shape):
        if i == j:
            op = num_fermion(i) if i < N else 1 - num_fermion(2 * N - 1 - i)
            result += complex(hopping_matrix[i, i]) * op
            continue

        left_op = cr_fermion(i) if i < N else an_fermion(2 * N - 1 - i)
        right_op = an_fermion(j) if j < N else cr_fermion(2 * N - 1 - j)

        result += complex(hopping_matrix[i, j]) * left_op * right_op

    return result
