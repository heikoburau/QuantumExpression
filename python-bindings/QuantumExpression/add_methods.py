from ._QuantumExpression import PauliExpression
from ._QuantumExpression import FermionExpression
from scipy import sparse
import numpy as np
from collections import defaultdict


def __repr__(self):
    class_name = "PauliExpression" if isinstance(self, PauliExpression) else "FermionExpression"
    terms_repr = [f"{term.coefficient} * {class_name}({repr(dict(term.quantum_string))})" for term in self]

    return "(\n  " + " +\n  ".join(terms_repr) + "\n)"


setattr(PauliExpression, "__repr__", __repr__)
setattr(FermionExpression, "__repr__", __repr__)


def __pow__(self, x):
    result = 1
    for n in range(x):
        result *= self

    return result


setattr(PauliExpression, "__pow__", __pow__)
setattr(FermionExpression, "__pow__", __pow__)


def to_json(self):
    return dict(
        type="PauliExpression" if isinstance(self, PauliExpression) else "FermionExpression",
        terms=[
            dict(
                coefficient=[
                    term.coefficient.real,
                    term.coefficient.imag
                ],
                quantum_string=dict(term.quantum_string)
            ) for term in self
        ]
    )


setattr(PauliExpression, "to_json", to_json)
setattr(FermionExpression, "to_json", to_json)


@staticmethod
def from_json(obj):
    assert obj["type"] in ("PauliExpression", "FermionExpression")
    qe = PauliExpression if obj["type"] == "PauliExpression" else FermionExpression

    return sum(
        (term["coefficient"][0] + 1j * term["coefficient"][1]) * qe(
            {int(index): op for index, op in term["quantum_string"].items()}
        )
        for term in obj["terms"]
    )


setattr(PauliExpression, "from_json", from_json)
setattr(FermionExpression, "from_json", from_json)


sparse_pauli_matrices = {
    op: sparse.identity(2) if op == 0 else sparse.csr_matrix(PauliExpression({0: op}).matrix(1))
    for op in range(4)
}

sparse_fermion_matrices = {
    op: sparse.identity(2) if op == 0 else sparse.csr_matrix(FermionExpression({0: op}).matrix(1))
    for op in range(4)
}

sigma_z_matrix = sparse.csr_matrix(np.array([
    [1, 0],
    [0, -1]
]))
sigma_z_products = {}


def sigma_z_product(n):
    if n not in sigma_z_products:
        result = sigma_z_matrix
        for i in range(1, n):
            result = sparse.kron(result, sigma_z_matrix)
        sigma_z_products[n] = result

    return sigma_z_products[n]


Sx = PauliExpression(0, 1).matrix(1, "paulis")
Sy = PauliExpression(0, 2).matrix(1, "paulis")
Sz = PauliExpression(0, 3).matrix(1, "paulis")


def sparse_matrix(self, N, basis="spins", U_list=None):
    assert basis in ("spins", "paulis")

    result = 0

    if basis == "paulis":
        if U_list is None:
            U_list = [np.eye(4)] * N
        if not isinstance(U_list, (list, tuple)):
            U_list = [U_list] * N

        S_map = [
            {
                1: sparse.csr_matrix(U @ Sx @ U.T.conj()),
                2: sparse.csr_matrix(U @ Sy @ U.T.conj()),
                3: sparse.csr_matrix(U @ Sz @ U.T.conj())
            }
            for U in U_list
        ]
        for term in self:
            sparse_string = sparse.identity(4**N)
            for i, op in term.quantum_string:
                right = sparse.identity(4**i) if i > 0 else 1
                left = sparse.identity(4**(N - i - 1)) if i < N - 1 else 1
                sparse_op = sparse.kron(
                    left,
                    sparse.kron(S_map[i][op], right)
                )
                sparse_string *= sparse_op

            result += term.coefficient * sparse_string

        return result

    is_fermionic = isinstance(self, FermionExpression)
    sparse_matrices = sparse_fermion_matrices if is_fermionic else sparse_pauli_matrices

    for term in self:
        sparse_string = sparse.identity(2**N)
        for i, op in term.quantum_string:
            if is_fermionic and op in [1, 2]:
                right = sigma_z_product(i) if i > 0 else 1
            else:
                right = sparse.identity(2**i) if i > 0 else 1
            left = sparse.identity(2**(N - i - 1)) if i < N - 1 else 1
            sparse_op = sparse.kron(
                left,
                sparse.kron(sparse_matrices[op], right)
            )
            sparse_string *= sparse_op

        result += term.coefficient * sparse_string

    return result


def effective_matrix(self, basis):
    basis = {s: i for i, s in enumerate(basis)}

    data, row_ind, col_ind = [], [], []

    for row, row_i in basis.items():
        for A_i in self:
            col_term = PauliExpression(row, 1.0) * A_i

            try:
                col_ind.append(basis[col_term.pauli_string])
                row_ind.append(row_i)
                data.append(col_term.coefficient)
            except KeyError:
                pass

    return sparse.csr_matrix((data, (row_ind, col_ind)), shape=(len(basis), len(basis)))


setattr(PauliExpression, "sparse_matrix", sparse_matrix)
setattr(FermionExpression, "sparse_matrix", sparse_matrix)
setattr(PauliExpression, "effective_matrix", effective_matrix)
