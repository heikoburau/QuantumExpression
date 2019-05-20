from ._QuantumExpression import PauliExpression
from ._QuantumExpression import FermionExpression
from scipy import sparse


def __repr__(self):
    class_name = "PauliExpression" if isinstance(self, PauliExpression) else "FermionExpression"
    terms_repr = [f"{term.coefficient} * {class_name}({repr(dict(term.quantum_string))})" for term in self]

    return "(\n  " + " +\n  ".join(terms_repr) + "\n)"


setattr(PauliExpression, "__repr__", __repr__)
setattr(FermionExpression, "__repr__", __repr__)


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


def sparse_matrix(self, N):
    result = 0

    sparse_matrices = sparse_pauli_matrices if isinstance(self, PauliExpression) else sparse_fermion_matrices

    for term in self:
        sparse_string = sparse.identity(2**N)
        for i, op in term.quantum_string:
            left = sparse.identity(2**i) if i > 0 else 1
            right = sparse.identity(2**(N - i - 1)) if i < N - 1 else 1
            sparse_op = sparse.kron(
                sparse.kron(right, sparse_matrices[op]),
                left
            )
            sparse_string *= sparse_op

        result += term.coefficient * sparse_string

    return result


setattr(PauliExpression, "sparse_matrix", sparse_matrix)
setattr(FermionExpression, "sparse_matrix", sparse_matrix)
