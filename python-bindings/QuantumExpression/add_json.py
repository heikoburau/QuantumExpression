from ._QuantumExpression import PauliExpression
from ._QuantumExpression import FermionExpression


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
