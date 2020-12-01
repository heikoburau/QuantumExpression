#pragma once

#include "QuantumExpression.hpp"
#include <Eigen/Sparse>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>
#include <vector>
#include <algorithm>
#include <array>
#include <math.h>
#include <stdint.h>


namespace quantum_expression {

using namespace std;


inline QuantumExpression<PauliString> commutator(
    const QuantumExpression<PauliString>& a,
    const QuantumExpression<PauliString>& b
) {
    QuantumExpression<PauliString> result;
    result.terms.reserve(a.size() * b.size());

    for(const auto& a_term : a) {
        for(const auto& b_term : b) {
            if(!a_term.first.commutes_with(b_term.first)) {
                const QuantumExpression<PauliString> a_expression(a_term);
                const QuantumExpression<PauliString> b_expression(b_term);

                result += a_expression * b_expression - b_expression * a_expression;
            }
        }
    }

    return result;
}


inline QuantumExpression<FastPauliString> commutator(
    const QuantumExpression<FastPauliString>& a,
    const QuantumExpression<FastPauliString>& b
) {
    QuantumExpression<FastPauliString> result;
    result.reserve(a.size() * b.size());

    for(const auto& a_term : a) {
        for(const auto& b_term : b) {

            const auto factor_and_string = commutator(a_term.first, b_term.first);

            if(factor_and_string.first != 0.0) {
                result.add(
                    factor_and_string.first * a_term.second * b_term.second,
                    factor_and_string.second
                );
            }
        }
    }

    return result;
}


inline FermionExpression commutator(
    const FermionExpression& a,
    const FermionExpression& b
) {
    FermionExpression result;
    result.terms.reserve(max(a.size(), b.size()));

    for(const auto& a_term : a) {
        for(const auto& b_term : b) {
            const FermionExpression a_expression(a_term);
            const FermionExpression b_expression(b_term);

            result += a_expression * b_expression - b_expression * a_expression;
        }
    }

    return result;
}


inline FermionExpression anti_commutator(
    const FermionExpression& a,
    const FermionExpression& b
) {
    FermionExpression result;
    result.terms.reserve(max(a.size(), b.size()));

    for(const auto& a_term : a) {
        for(const auto& b_term : b) {
            const FermionExpression a_expression(a_term);
            const FermionExpression b_expression(b_term);

            result += a_expression * b_expression + b_expression * a_expression;
        }
    }

    return result;
}


inline complex<double> trace(const PauliExpression& expr, const unsigned int N) {
    return std::pow(2, N) * expr[typename PauliExpression::QuantumString()];
}


inline complex<double> trace(const FermionExpression& expr, const unsigned int N) {
    complex<double> result = 0.0+0i;

    for(const auto& term : expr.diagonal_terms()) {
        result += term.second * std::pow(2.0, N - term.first.size());
    }

    return result;
}


inline decltype(auto) partial_trace(const FermionExpression& expr, const unsigned int N, const unsigned int interface) {
    const auto dim_A = 1 << interface;

    xt::pytensor<complex<double>, 2> result(
        array<long int, 2>({static_cast<long int>(dim_A), static_cast<long int>(dim_A)})
    );
    result = xt::zeros<complex<double>>({dim_A, dim_A});

    uint64_t state, state_prime;
    const uint64_t A_partition = (1 << interface) - 1;
    const uint64_t B_partition = ((1 << (N - interface)) - 1) << interface;

    for(const auto& term : expr) {
        state = 0;
        for(const auto& symbol : term.first) {
            state |= 1u << symbol.index;
        }

        for(const auto& prime_term : expr) {
            state_prime = 0;
            for(const auto& symbol : prime_term.first) {
                state_prime |= 1u << symbol.index;
            }

            if((state & B_partition) == (state_prime & B_partition)) {
                const auto idx = state & A_partition;
                const auto prime_idx = state_prime & A_partition;

                result(idx, prime_idx) += term.second * conj(prime_term.second);
            }
        }
    }

    return result;
}


inline decltype(auto) partial_trace(const xt::pytensor<complex<double>, 1>& state, const unsigned int N, const unsigned int interface) {
    const auto dim_A = 1u << interface;
    const auto dim_B = 1u << (N - interface);

    xt::pytensor<complex<double>, 2> result(
        array<long int, 2>({static_cast<long int>(dim_A), static_cast<long int>(dim_A)})
    );
    result = xt::zeros<complex<double>>({dim_A, dim_A});

    for(auto idx_A = 0u; idx_A < dim_A; idx_A++) {
        for(auto idx_A_prime = 0u; idx_A_prime < dim_A; idx_A_prime++) {
            for(auto idx_B = 0u; idx_B < dim_B; idx_B++) {
                result(idx_A, idx_A_prime) += (
                    state[idx_A | (idx_B << interface)] * conj(state[idx_A_prime | (idx_B << interface)])
                );
            }
        }
    }

    return result;
}


template<typename QuantumExpression_t>
inline double frobenius_norm(const QuantumExpression_t& expr, const unsigned int N) {
    const auto expr_dagger = expr.dagger();

    complex<double> result = 0.0+0i;

    for(const auto& term : expr) {
        const auto square_term = expr_dagger * QuantumExpression_t(term);
        result += trace(square_term, N);
    }

    return sqrt(result.real());
}


inline complex<double> mul_trace(const FermionExpression& A, const FermionExpression& B, const unsigned int N) {
    complex<double> result = 0.0+0i;

    for(const auto& a_term : A) {
        for(const auto& b_term : B) {
            const auto& ab = FermionExpression(a_term) * FermionExpression(b_term);
            result += trace(ab, N);
        }
    }

    return result;
}


inline FermionExpression exp_and_apply(const FermionExpression& H, const complex<double>& factor, const FermionExpression& state) {
    using complex_t = complex<double>;
    FermionExpression result;
    result.terms.reserve(state.size());

    vector<pair<uint64_t, complex_t>> exp_H;
    exp_H.reserve(H.size());
    for(const auto& H_term : H) {
        uint64_t op = 0;
        for(const auto symbol : H_term.first) {
            op |= 1 << symbol.index;
        }
        exp_H.push_back({op, exp(factor * H_term.second)});
    }

    for(const auto& state_term : state) {
        uint64_t configuration = 0;
        for(const auto& symbol : state_term.first) {
            configuration |= 1 << symbol.index;
        }

        auto prefactor = state_term.second;
        for(const auto& op_and_factor : exp_H) {
            const auto& op = op_and_factor.first;
            const auto& op_factor = op_and_factor.second;

            // check if there is a fermion for each site the operator lives on
            if((configuration & op) == op) {
                prefactor *= op_factor;
            }
        }

        result.insert({state_term.first, prefactor});
    }

    return result;
}


inline FermionExpression change_basis(const FermionExpression& state, const vector<FermionExpression>& new_basis, const double threshold) {
    FermionExpression result;

    for(const auto& term : state) {
        FermionExpression term_result(term.second);

        for(const auto symbol : term.first) {
            term_result = mul(term_result, new_basis[symbol.index], 0.5 * threshold);
        }

        result += term_result;
    }

    return result.apply_threshold(threshold);
}


inline FermionExpression substitute(
    const FermionExpression& expr,
    const vector<FermionExpression>& c_dagger_vec,
    const vector<FermionExpression>& c_vec,
    const vector<FermionExpression>& n_vec,
    const double threshold
) {
    FermionExpression result;

    array<const vector<FermionExpression>*, 3> new_operators = {&c_dagger_vec, &c_vec, &n_vec};

    for(const auto& term : expr) {
        FermionExpression term_result(term.second);

        for(const auto symbol : term.first) {
            term_result = mul(term_result, (*new_operators[symbol.op.type - 1])[symbol.index], 0.5 * threshold);
        }

        result += term_result;
    }

    return result;
}


Eigen::SparseMatrix<complex<double>> effective_matrix(
    const PauliExpression& op, const PauliExpression& basis, const unsigned int trans_inv_length=0u
) {
    using SparseMatrix = Eigen::SparseMatrix<complex<double>>;
    using Triplet = Eigen::Triplet<complex<double>>;

    unordered_map<FastPauliString, unsigned int> indices_map;
    indices_map.reserve(basis.size());

    auto i = 0u;
    for(const auto& term : basis) {
        indices_map.insert({term.first, i++});
    }

    vector<Triplet> triplet_list;
    triplet_list.reserve(op.size() * basis.size());

    auto col = 0u;
    for(const auto& b : basis) {
        for(const auto& h : op) {
            auto factor_and_string = b.first * h.first;
            if(trans_inv_length > 0u) {
                factor_and_string.second = factor_and_string.second.rotate_to_smallest(trans_inv_length);
            }
            const auto row_it = indices_map.find(factor_and_string.second);
            if(row_it != indices_map.end()) {
                triplet_list.push_back(Triplet(
                    row_it->second, col, factor_and_string.first * h.second
                ));
            }
        }

        col++;
    }

    SparseMatrix result(basis.size(), basis.size());
    result.setFromTriplets(triplet_list.begin(), triplet_list.end());

    return result;
}


inline decltype(auto) su2_su2_matrix(const PauliExpression& expr_a, const PauliExpression& expr_b, const unsigned int N) {
    const auto dim = 1u << (2u * N);

    xt::pytensor<complex<double>, 2> result(
        array<long int, 2>({static_cast<long int>(dim), static_cast<long int>(dim)})
    );

    for(auto conf_idx = 0u; conf_idx < dim; conf_idx++) {
        const auto conf = FastPauliString::enumerate(conf_idx);

        for(const auto& first_term : expr_a * conf) {
            if(first_term.second != 0.0) {
                for(const auto& second_term : first_term.first * expr_b) {
                    result(second_term.first.enumeration_index(), conf_idx) += first_term.second * second_term.second;
                }
            }
        }
    }

    return result;
}


} // quantum_expression
