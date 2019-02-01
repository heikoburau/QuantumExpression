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
#include <assert.h>


namespace quantum_expression {

using namespace std;
using namespace complex_literals;


template<typename QuantumOperator>
class QuantumString {
public:
    using This = QuantumString<QuantumOperator>;

    // TODO: check if multi-set/map is faster
    set<int>                    indices;
    // TODO: check if unordered_map is faster
    map<int, QuantumOperator>   operators;

    struct operator_iterator
    : public
    iterator<typename map<int, QuantumOperator>::const_iterator::iterator_category, pair<int, int>>,
    map<int, QuantumOperator>::const_iterator
    {
        operator_iterator(
            const typename map<int, QuantumOperator>::const_iterator& base
        ) : map<int, QuantumOperator>::const_iterator(base) {}

        inline pair<int, int> operator*() const {
            const auto index_and_op = map<int, QuantumOperator>::const_iterator::operator*();
            return {index_and_op.first, index_and_op.second.type};
        }
    };

public:
    QuantumString() = default;

    inline QuantumString(initializer_list<pair<int, QuantumOperator>> operators) {
        for(const auto& x : operators) {
            this->indices.insert(x.first);
            this->operators.insert(x);
        }
    }

    inline QuantumString(
        const map<int, QuantumOperator>& operators
    ) : operators(operators) {
        for(const auto& index_and_op : operators) {
            this->indices.insert(index_and_op.first);
        }
    }

    inline QuantumString(const map<int, int>& operators) {
        for(const auto& index_and_type : operators) {
            this->indices.insert(index_and_type.first);
            this->operators.insert({index_and_type.first, QuantumOperator(index_and_type.second)});
        }
    }

    inline QuantumString(const map<int, char>& operators) {
        for(const auto& index_and_name : operators) {
            this->indices.insert(index_and_name.first);
            this->operators.insert({index_and_name.first, QuantumOperator(index_and_name.second)});
        }
    }

    inline QuantumString dagger() const {
        QuantumString result(*this);

        for(auto& index_and_op : result.operators) {
            index_and_op.second = index_and_op.second.dagger();
        }

        return result;
    }

    inline bool operator==(const QuantumString& other) const {
        return this->operators == other.operators;
    }

    inline bool operator!=(const QuantumString& other) const {
        return this->operators != other.operators;
    }

    inline operator bool() const {
        return !this->indices.empty();
    }

    inline bool cast_to_bool() const {
        return *this;
    }

    inline size_t size() const {
        return this->indices.size();
    }

    string str() const {
        stringstream result;

        if(this->indices.empty()) {
            result << "QuantumString \"\"\n";
            return result.str();
        }

        const int min = this->min_index();
        const int max = this->max_index();

        result << "QuantumString " << min << " -> \"";
        for(int i = min; i <= max; i++) {
            const auto search = this->operators.find(i);
            if(search == this->operators.end()) {
                result << " ";
            }
            else {
                result << search->second.str();
            }
        }
        result << "\"";

        return result.str();
    }

    inline forward_list<int> overlap(const QuantumString& other) const {
        forward_list<int> overlap;

        set_intersection(
            this->indices.begin(), this->indices.end(),
            other.indices.begin(), other.indices.end(),
            front_inserter(overlap)
        );

        return overlap;
    }

    inline forward_list<int> difference(const QuantumString& other) const {
        forward_list<int> difference;

        set_symmetric_difference(
            this->indices.begin(), this->indices.end(),
            other.indices.begin(), other.indices.end(),
            front_inserter(difference)
        );

        return difference;
    }

    inline bool commutes_with(const QuantumString& other) const {
        const auto overlap = this->overlap(other);

        if(overlap.empty())
            return true;

        if(QuantumOperator::is_pauli_operator()) {
            auto num_different_operators = 0u;
            for(const auto i : overlap) {
                if((*this)[i] != other[i]) {
                    num_different_operators++;
                }
            }

            return (num_different_operators % 2u) == 0u;
        }

        assert(0);
        return false;
    }

    inline pair<Spins, complex<double>> apply(Spins spins) const {
        complex<double> factor = 1.0;

        for(const auto& index_and_op : this->operators) {
            const auto spins_and_factor = index_and_op.second.apply(spins, index_and_op.first);

            spins = spins_and_factor.first;
            factor *= spins_and_factor.second;
        }

        return {spins, factor};
    }

    inline bool is_diagonal() const {
        for(const auto& index_and_op : this->operators) {
            const auto& op = index_and_op.second;
            if(op.type == 1 || op.type == 2) {
                return false;
            }
        }

        return true;
    }

    inline QuantumOperator& operator[](const int index) {
        return this->operators.find(index)->second;
    }

    inline const QuantumOperator& operator[](const int index) const {
        return this->operators.find(index)->second;
    }

    int min_index() const {
        return *min_element(this->indices.begin(), this->indices.end());
    }

    int max_index() const {
        return *max_element(this->indices.begin(), this->indices.end());
    }

    inline operator_iterator begin() const {
        return this->operators.begin();
    }

    inline operator_iterator end() const {
        return this->operators.end();
    }

};


template<typename QuantumString_t>
inline decltype(auto) operator*(const QuantumString_t& a, const QuantumString_t& b) {
    using Term = pair<QuantumString_t, complex<double>>;
    using Terms = forward_list<Term>;
    Terms result = {{QuantumString_t(), 1.0}};

    for(const auto i : a.overlap(b)) {
        const auto op_and_prefactor_and_constant = a[i] * b[i];
        const auto& op = get<0>(op_and_prefactor_and_constant);
        const auto& prefactor = get<1>(op_and_prefactor_and_constant);
        const auto& constant = get<2>(op_and_prefactor_and_constant);

        if(prefactor == 0.0) {
            return Terms();
        }

        Terms terms_of_constant;
        if(constant != 0.0) {
            for(const auto& term : result) {
                terms_of_constant.push_front({term.first, constant * term.second});
            }
        }
        for(auto& term : result) {
            if(!op.is_identity()) {
                term.first.indices.insert(i);
                term.first.operators.insert({i, op});
            }
            term.second *= prefactor;
        }
        for(const auto& term : terms_of_constant) {
            result.push_front(term);
        }
    }

    // TODO: iterate a and b separately
    for(const auto i : a.difference(b)) {
        for(auto& term : result) {
            // TODO: try `emplace`
            term.first.indices.insert(i);

            const auto in_a = a.operators.find(i);
            if(in_a != a.operators.end()) {
                term.first.operators.insert(*in_a);
            }
            else {
                term.first.operators.insert({i, b[i]});
            }
        }
    }

    return result;
}


template<typename QuantumOperator>
ostream& operator<<(ostream& os, const QuantumString<QuantumOperator>& quantum_string) {
    os << quantum_string.str();
    return os;
}

using PauliString = QuantumString<PauliOperator>;
using FermionString = QuantumString<FermionOperator>;

} // namespace quantum_expression


namespace std {

using namespace quantum_expression;

template<typename QuantumOperator>
struct hash<QuantumString<QuantumOperator>> {
    inline size_t operator()(const QuantumString<QuantumOperator>& quantum_string) const {
        size_t result = 0;
        for(const auto index : quantum_string.indices) {
            result += index + index * index;
        }
        return result;
    }
};

} // std
