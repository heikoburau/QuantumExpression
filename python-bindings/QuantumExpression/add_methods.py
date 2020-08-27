from ._QuantumExpression import PauliExpression
from ._QuantumExpression import FermionExpression
from scipy import sparse
import numpy as np


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


def sparse_matrix(self, N):
    result = 0

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
