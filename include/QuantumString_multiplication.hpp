#pragma once


namespace quantum_expression {


inline decltype(auto) operator*(const FermionString& a, const FermionString& b) {
    using Term = pair<FermionString, complex<double>>;
    using Terms = forward_list<Term>;

    auto overall_prefactor = 1.0;
    for(const auto& b_symbol : b) {
        if(b_symbol.op.type == 3 || b_symbol.op.type == 0) {
            continue;
        }

        auto num_exchanges = 0u;
        for(const auto& a_symbol : a) {
            if(a_symbol.index > b_symbol.index) {
                if(a_symbol.op.type == 1 || a_symbol.op.type == 2) {
                    num_exchanges++;
                }
            }
        }
        if(num_exchanges % 2u == 1u) {
            overall_prefactor *= -1.0;
        }
    }

    Terms result = {{FermionString(), overall_prefactor}};

    a.loop_in_common(
        b,
        [&](const auto& a_symbol) {
            for(auto& term : result) {
                term.first.add_symbol(a_symbol);
            }
        },
        [&](const auto& b_symbol) {
            for(auto& term : result) {
                term.first.add_symbol(b_symbol);
            }
        },
        [&](const auto& a_symbol, const auto& b_symbol) -> bool {
            const auto op_and_prefactor_and_constant = a_symbol.op * b_symbol.op;
            const auto& op = get<0>(op_and_prefactor_and_constant);
            const auto& prefactor = get<1>(op_and_prefactor_and_constant);
            const auto& constant = get<2>(op_and_prefactor_and_constant);

            if(prefactor == 0.0) {
                result = Terms();
                return false;
            }

            Terms terms_of_constant;
            if(constant != 0.0) {
                for(const auto& term : result) {
                    terms_of_constant.push_front({term.first, constant * term.second});
                }
            }
            for(auto& term : result) {
                if(!op.is_identity()) {
                    term.first.add_symbol({a_symbol.index, op});
                }
                term.second *= prefactor;
            }
            for(const auto& term : terms_of_constant) {
                result.push_front(term);
            }

            return true;
        }
    );

    return result;
}


inline decltype(auto) operator*(const PauliString& a, const PauliString& b) {
    using Term = pair<PauliString, complex<double>>;
    using Terms = forward_list<Term>;

    PauliString result;
    complex<double> prefactor = 1;

    a.loop_in_common(
        b,
        [&](const auto& a_symbol) {
            result.add_symbol(a_symbol);
        },
        [&](const auto& b_symbol) {
            result.add_symbol(b_symbol);
        },
        [&](const auto& a_symbol, const auto& b_symbol) -> bool {
            const auto op_and_prefactor_and_constant = a_symbol.op * b_symbol.op;
            const auto& op = get<0>(op_and_prefactor_and_constant);
            const auto& prefactor_i = get<1>(op_and_prefactor_and_constant);

            if(!op.is_identity()) {
                result.add_symbol({a_symbol.index, op});
            }
            prefactor *= prefactor_i;

            return true;
        }
    );

    return Terms({Term(result, prefactor)});
}

} // namespace quantum_expression
