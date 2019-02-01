#pragma once

#include "QuantumExpression.hpp"
#include <math.h>


namespace quantum_expression {


inline complex<double> trace(const FermionExpression& expr, const unsigned int N) {
    complex<double> result = 0.0+0i;

    for(const auto& term : expr.diagonal_terms()) {
        result += term.second * pow(2.0, N - term.first.size());
    }

    return result;
}


double frobenius_norm(const FermionExpression& expr, const unsigned int N) {
    const auto expr_dagger = expr.dagger();

    complex<double> result = 0.0+0i;

    for(const auto& term : expr) {
        const auto square_term = expr_dagger * FermionExpression(term);
        result += trace(square_term, N);
    }

    return sqrt(result.real());
}


complex<double> mul_trace(const FermionExpression& A, const FermionExpression& B, const unsigned int N) {
    complex<double> result = 0.0+0i;

    for(const auto& a_term : A) {
        for(const auto& b_term : B) {
            const auto& ab = FermionExpression(a_term) * FermionExpression(b_term);
            result += trace(ab, N);
        }
    }

    return result;
}

} // quantum_expression
