#pragma once

#include "PauliOperator.hpp"
#include "FermionOperator.hpp"
#include "Spins.hpp"

#include <algorithm>
#include <iterator>
#include <forward_list>
#include <set>
#include <map>
#include <utility>
#include <initializer_list>
#include <iostream>
#include <complex>
#include <iterator>
#include <tuple>
#include <vector>
#include <assert.h>


namespace quantum_expression {

using namespace std;
using namespace complex_literals;


template<typename QuantumOperator_t>
class QuantumString {
public:
    using QuantumOperator = QuantumOperator_t;
    using This = QuantumString<QuantumOperator>;
    using Configuration = typename QuantumOperator::Configuration;

    struct Symbol {
        int             index;
        QuantumOperator op;

        inline bool operator==(const Symbol& other) const {
            return (this->index == other.index) && (this->op == other.op);
        }

        inline bool operator!=(const Symbol& other) const {
            return (this->index != other.index) || (this->op != other.op);
        }
    };

    using Symbols = vector<Symbol>;
    Symbols symbols;

    struct symbol_iterator
    : public
    iterator<typename Symbols::const_iterator::iterator_category, pair<int, int>>,
    Symbols::const_iterator
    {
        symbol_iterator(
            const typename Symbols::const_iterator& base
        ) : Symbols::const_iterator(base) {}

        inline pair<int, int> operator*() const {
            const auto symbol = Symbols::const_iterator::operator*();
            return {symbol.index, symbol.op.type};
        }
    };

public:
    QuantumString() = default;

    inline QuantumString(initializer_list<pair<int, QuantumOperator>> symbols) {
        this->symbols.reserve(symbols.size());
        for(const auto& x : symbols) {
            this->symbols.push_back({x.first, x.second});
        }
        this->sort_symbols();
    }

    inline QuantumString(
        const vector<Symbol>& symbols
    ) : symbols(symbols) {}

    inline QuantumString(const map<int, int>& symbols) {
        this->symbols.reserve(symbols.size());
        for(const auto& symbol : symbols) {
            this->symbols.push_back({symbol.first, QuantumOperator(symbol.second)});
        }
        this->sort_symbols();
    }

    inline QuantumString(const map<int, char>& symbols) {
        this->symbols.reserve(symbols.size());

        for(const auto& symbol : symbols) {
            this->symbols.push_back({symbol.first, QuantumOperator(symbol.second)});
        }
        this->sort_symbols();
    }

    inline void sort_symbols() {
        sort(
            this->symbols.begin(),
            this->symbols.end(),
            [](const Symbol& a, const Symbol& b) {
                return a.index < b.index;
            }
        );
    }

    inline void add_symbol(const Symbol& symbol) {
        this->symbols.push_back(symbol);
    }

    inline vector<int> get_indices() const {
        vector<int> result;
        result.reserve(this->symbols.size());
        for(const auto& symbol : *this) {
            result.push_back(symbol.index);
        }

        return result;
    }

    inline vector<int> get_types() const {
        vector<int> result;
        result.reserve(this->symbols.size());
        for(const auto& symbol : *this) {
            result.push_back(symbol.op.type);
        }

        return result;
    }

    inline pair<QuantumString, double> dagger() const {
        QuantumString result(*this);

        for(auto& symbol : result.symbols) {
            symbol.op = symbol.op.dagger();
        }

        auto prefactor = 1.0;
        if(!QuantumOperator::is_pauli_operator()) {

            auto n = 0u;
            for(const auto& symbol : result.symbols) {
                if(symbol.op.type == 1u || symbol.op.type == 2u) {
                    n++;
                }
            }

            if(n > 1) {
                const auto num_permutations = ((n - 1) * n) / 2;

                if(num_permutations % 2u == 1u) {
                    prefactor = -1.0;
                }
            }
        }

        return {result, prefactor};
    }

    inline bool operator==(const QuantumString& other) const {
        return this->symbols == other.symbols;
    }

    inline bool operator!=(const QuantumString& other) const {
        return this->symbols != other.symbols;
    }

    inline operator bool() const {
        return !this->symbols.empty();
    }

    inline bool cast_to_bool() const {
        return *this;
    }

    inline size_t size() const {
        return this->symbols.size();
    }

    inline bool contains(const int index) const {
        return count_if(this->symbols.begin(), this->symbols.end(), [=](const auto& s) {return s.index == index;}) > 0u;
    }

    string str() const {
        stringstream result;

        if(!(*this)) {
            result << "QuantumString \"\"\n";
            return result.str();
        }

        const int min = this->min_index();
        const int max = this->max_index();

        result << "QuantumString " << min << " -> \"";
        for(int i = min; i <= max; i++) {
            if(this->contains(i)) {
                result << (*this)[i].str();
            }
            else {
                result << " ";
            }
        }
        result << "\"";

        return result.str();
    }

    inline forward_list<pair<Symbol, Symbol>> overlap(const QuantumString& other) const {
        forward_list<pair<Symbol, Symbol>> overlap;

        if(!(*this) || !other) {
            return overlap;
        }

        if(
            this->symbols.front().index > other.symbols.back().index ||
            this->symbols.back().index < other.symbols.front().index
        ) {
            return overlap;
        }

        auto a_it = this->symbols.begin();
        auto b_it = other.symbols.begin();

        while(a_it != this->symbols.end() && b_it != other.symbols.end()) {
            const auto a = *a_it;
            const auto b = *b_it;

            if(a.index == b.index) {
                overlap.push_front({a, b});
                a_it++;
                b_it++;
            }
            else if(a.index < b.index) {
                a_it++;
            }
            else {
                b_it++;
            }
        }

        return overlap;
    }

    template<typename FunctionA, typename FunctionB, typename FunctionAB>
    inline void loop_in_common(const QuantumString& other, FunctionA function_a, FunctionB function_b, FunctionAB function_ab) const {
        auto a_it = this->symbols.begin();
        auto b_it = other.symbols.begin();

        while(a_it != this->symbols.end() || b_it != other.symbols.end()) {
            const auto a = a_it != this->symbols.end() ? *a_it : Symbol({100000, QuantumOperator(-1)});
            const auto b = b_it != other.symbols.end() ? *b_it : Symbol({100000, QuantumOperator(-1)});

            if(a.index == b.index) {
                if(!function_ab(a, b)) {
                    return;
                }
                a_it++;
                b_it++;
            }
            else if(a.index < b.index) {
                function_a(a);
                a_it++;
            }
            else {
                function_b(b);
                b_it++;
            }
        }
    }

    inline bool commutes_with(const QuantumString& other) const {
        if(QuantumOperator::is_pauli_operator()) {
            auto num_different_operators = 0u;
            for(const auto a_and_b : this->overlap(other)) {
                if(a_and_b.first.op != a_and_b.second.op) {
                    num_different_operators++;
                }
            }

            return (num_different_operators % 2u) == 0u;
        }

        assert(0);
        return false;
    }

    inline pair<Configuration, complex<double>> apply(Configuration conf) const {
        complex<double> factor = 1.0;

        for(const auto& symbol : this->symbols) {
            const auto conf_and_factor = symbol.op.apply(conf, symbol.index);

            conf = conf_and_factor.first;
            factor *= conf_and_factor.second;

            if(factor == 0.0) {
                return {conf, 0.0};
            }
        }

        return {conf, factor};
    }

    inline void apply(complex<double>* out_state, complex<double>* in_state, const unsigned int dim_N) const {
        for(const auto& symbol : this->symbols) {
            const auto index_bit = (1u << symbol.index);

            if(symbol.op.type == 1) {
                for(auto conf = 0u; conf < dim_N; conf++) {
                    out_state[conf ^ index_bit] = in_state[conf];
                }
            }
            else if (symbol.op.type == 2) {
                for(auto conf = 0u; conf < dim_N; conf++) {
                    out_state[conf ^ index_bit] = 1i * (2.0 * double(bool(conf & index_bit)) - 1.0) * in_state[conf];
                }
            }
            else if (symbol.op.type == 3) {
                for(auto conf = 0u; conf < dim_N; conf++) {
                    out_state[conf] = (2.0 * double(bool(conf & index_bit)) - 1.0) * in_state[conf];
                }
            }

            complex<double>* tmp = in_state;
            in_state = out_state;
            out_state = tmp;
        }
        if(this->size() % 2u == 0u) {
            for(auto conf = 0u; conf < dim_N; conf++) {
                out_state[conf] = in_state[conf];
            }
        }
    }

    inline bool is_diagonal() const {
        for(const auto& symbol : this->symbols) {
            if(symbol.op == 1 || symbol.op == 2) {
                return false;
            }
        }

        return true;
    }

    inline QuantumOperator& operator[](const int index) {
        for(const auto& symbol : this->symbols) {
            if(index == symbol.index) {
                return symbol.op;
            }
        }
        cerr << "[QuantumString] could not find index " << index << endl;
        return this->symbols[0].op;
    }

    inline const QuantumOperator& operator[](const int index) const {
        for(const auto& symbol : this->symbols) {
            if(index == symbol.index) {
                return symbol.op;
            }
        }
        cerr << "[QuantumString] could not find index " << index << endl;
        return this->symbols.front().op;
    }

    int min_index() const {
        return this->symbols.front().index;
    }

    int max_index() const {
        return this->symbols.back().index;
    }

    decltype(auto) begin() {
        return this->symbols.begin();
    }

    decltype(auto) begin() const {
        return this->symbols.begin();
    }

    decltype(auto) end() {
        return this->symbols.end();
    }

    decltype(auto) end() const {
        return this->symbols.end();
    }

    inline symbol_iterator begin_symbols() const {
        return symbol_iterator(this->symbols.begin());
    }

    inline symbol_iterator end_symbols() const {
        return symbol_iterator(this->symbols.end());
    }
};

using PauliString = QuantumString<PauliOperator>;
using FermionString = QuantumString<FermionOperator>;

template<typename QuantumOperator>
ostream& operator<<(ostream& os, const QuantumString<QuantumOperator>& quantum_string) {
    os << quantum_string.str();
    return os;
}


} // namespace quantum_expression

#include "QuantumString_multiplication.hpp"


namespace std {

template<typename QuantumOperator>
struct hash<quantum_expression::QuantumString<QuantumOperator>> {
    inline size_t operator()(const quantum_expression::QuantumString<QuantumOperator>& quantum_string) const {
        size_t result = 0;
        for(const auto& symbol : quantum_string.symbols) {
            const auto index = symbol.index;
            result += index + index * index;
        }
        return result;
    }
};

} // std
