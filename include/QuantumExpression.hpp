#pragma once

#include "QuantumString.hpp"
#include "Spins.hpp"

#ifndef NO_PYTHON

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"

#endif

#include <unordered_map>
#include <complex>
#include <forward_list>
#include <vector>
#include <map>
#include <algorithm>
#include <sstream>
#include <string>
#include <iomanip>
#include <math.h>
#include <iterator>


namespace quantum_expression {

using namespace std;


template<typename QuantumString_t, typename Coefficient_t=complex<double>>
class QuantumExpression {
public:
    using QuantumString = QuantumString_t;
    using Coefficient = Coefficient_t;
    using This = QuantumExpression<QuantumString, Coefficient>;
    using Configuration = typename QuantumString::Configuration;

    typedef pair<QuantumString, Coefficient> Term;
    typedef unordered_multimap<
        QuantumString,
        Coefficient
    > Terms;

    Terms terms;

    struct term_iterator
    : public
    iterator<typename Terms::const_iterator::iterator_category, QuantumExpression>,
    Terms::const_iterator
    {
        term_iterator(const typename Terms::const_iterator& base) : Terms::const_iterator(base) {}

        inline This operator*() const {
            return This(Terms::const_iterator::operator*());
        }
    };

public:

    QuantumExpression() = default;
    QuantumExpression(const Coefficient& coefficient) {
        this->insert({QuantumString(), coefficient});
    }
    inline explicit QuantumExpression(const QuantumString& quantum_string, const Coefficient& coefficient) {
        if(coefficient != 0.0) {
            this->insert({quantum_string, coefficient});
        }
    }
    inline explicit QuantumExpression(const map<int, int>& quantum_string_map) {
        map<int, int> cleaned_quantum_string_map;
        for(const auto& index_and_type : quantum_string_map) {
            if(index_and_type.second >= 1 && index_and_type.second <= 3) {
                cleaned_quantum_string_map.insert(index_and_type);
            }
        }

        this->insert({cleaned_quantum_string_map, 1.0});
    }
    inline QuantumExpression(const Term& term) {
        this->insert(term);
    }
    inline QuantumExpression(initializer_list<Term> terms) {
        for(const auto& term : terms) {
            this->insert(term);
        }
    }

    inline This dagger() const {
        This result;
        result.terms.reserve(this->size());

        for(const auto& term : this->terms) {
            const auto string_and_prefactor = term.first.dagger();
            const auto& str = string_and_prefactor.first;
            const auto& prefactor = string_and_prefactor.second;

            result.insert({str, prefactor * conj(term.second)});
        }

        return result;
    }

    inline bool operator==(const This& other) const {
        return this->terms == other.terms;
    }

    inline bool operator==(const Coefficient& number) const {
        if(this->terms.empty()) {
            return number == 0.0;
        }

        return *this == This(number);
    }

    inline bool operator!=(const This& other) const {
        return this->terms != other.terms;
    }

    inline bool operator!=(const Coefficient& number) const {
        return !(*this == number);
    }

    inline Coefficient operator[](const QuantumString& quantum_string) const {
        const auto search = this->terms.find(quantum_string);

        if(search != this->end()) {
            return search->second;
        }
        return 0.0;
    }

    inline Coefficient __getitem__(const This& other) const {
        assert(other.size() == 1u);

        return (*this)[other.begin()->first];
    }

    inline size_t size() const {
        return this->terms.size();
    }

    inline void insert(const Term& term) {
        if(term.second != 0.0) {
            this->terms.insert(term);
        }
    }

    string str() const {
        stringstream result;

        forward_list<int> all_indices;
        for(const auto& term : *this) {
            const auto indices = term.first.get_indices();
            copy(
                indices.begin(),
                indices.end(),
                front_inserter(all_indices)
            );
        }
        if(all_indices.empty()) {
            result << "QuantumExpression " << this->get_coefficient() << endl;
            return result.str();
        }

        const int min = *min_element(all_indices.begin(), all_indices.end());
        const int max = *max_element(all_indices.begin(), all_indices.end());

        vector<Term> sorted_terms(this->begin(), this->end());
        sort(
            sorted_terms.begin(),
            sorted_terms.end(),
            [](const Term& a, const Term& b) {
                return abs(a.second) > abs(b.second);
            }
        );

        result << "QuantumExpression " << min << ":\n";
        for(const auto& term : sorted_terms) {
            result << right << setw(20) << term.second << " -> ";
            for(int i = min; i <= max; i++) {
                if(term.first.contains(i)) {
                    result << term.first[i].str();
                }
                else {
                    result << " ";
                }
            }
            result << endl;
        }

        return result.str();
    }

    inline This& operator+=(const This& other) {
        for(const auto& other_term : other) {
            auto search = this->terms.find(other_term.first);

            if(search == this->terms.end()) {
                // TODO: try using `emplace`
                if(other_term.second != 0.0) {
                    this->insert(other_term);
                }
            }
            else {
                search->second += other_term.second;
                if(search->second == 0.0) {
                    this->terms.erase(search);
                }
            }
        }

        return *this;
    }

    inline This& operator-=(const This& other) {
        for(const auto& other_term : other) {
            auto search = this->terms.find(other_term.first);

            if(search == this->terms.end()) {
                // TODO: try using `emplace`
                if(other_term.second != 0.0) {
                    this->insert({other_term.first, -other_term.second});
                }
            }
            else {
                search->second -= other_term.second;
                if(search->second == 0.0) {
                    this->terms.erase(search);
                }
            }
        }

        return *this;
    }

    inline This operator*=(const Coefficient& number) {
        for(auto& term : *this) {
            term.second *= number;
        }

        return *this;
    }

    inline This operator-() const {
        This result = *this;

        for(auto& term : result) {
            term.second *= -1.0;
        }

        return result;
    }

    inline This operator+() const {
        return This(*this);
    }

    inline This apply_threshold(const double threshold) const {
        This result;
        result.terms.reserve(this->size());

        for(const auto& term : *this) {
            if(abs(term.second) >= threshold) {
                result.terms.insert(term);
            }
        }

        return result;
    }

    inline This exp(const double threshold=0.0) const {
        // WARNING: only imaginary terms are considered

        This result(1.0);

        for(const auto& term : *this) {
            const double angle = term.second.imag();

            const This x = {
                Term({QuantumString(), cos(angle)}),
                Term({term.first, 1i * sin(angle)})
            };

            result = result * x;

            if(threshold > 0.0) {
                result = result.apply_threshold(threshold);
            }
        }

        return result;
    }

    inline This rotate_by(const This& generator, const double threshold=0.0) const {
        if(this->is_numeric()) {
            return This(this->get_coefficient());
        }

        This result;

        for(const auto& term : *this) {
            This relevant_generator;

            for(const auto& generator_term : generator) {
                if(!term.first.commutes_with(generator_term.first)) {
                    relevant_generator.insert({generator_term.first, 2.0 * generator_term.second});
                }
            }

            result += relevant_generator.exp(threshold) * This(term);
        }

        return result;
    }

    inline This diagonal_terms() const {
        This result;
        for(const auto& term : *this) {
            if(term.first.is_diagonal()) {
                result.insert(term);
            }
        }

        return result;
    }

    inline This max_term() const {
        return This(*max_element(
            this->begin(),
            this->end(),
            [](const Term& a, const Term& b) {
                return abs(a.second) < abs(b.second);
            }
        ));
    }

    inline double max_norm() const {
        if(this->terms.empty()) {
            return 0.0;
        }
        return abs(this->max_term().begin()->second);
    }

    inline double absolute() const {
        if(this->terms.empty()) {
            return 0.0;
        }
        return abs(this->begin()->second);
    }

    inline decltype(auto) apply(const Configuration& conf) const {
        forward_list<pair<Configuration, Coefficient>> result;

        for(const auto& term : *this) {
            const auto conf_and_factor = term.first.apply(conf);

            result.push_front({conf_and_factor.first, term.second * conf_and_factor.second});
        }

        return result;
    }

#ifndef NO_PYTHON

    decltype(auto) matrix(const unsigned int N) const {
        const auto dim_N = 1u << N;

        xt::pytensor<Coefficient, 2u> result({
            (long int)dim_N, (long int)dim_N
        });
        result = xt::zeros<Coefficient>(result.shape());

        for(auto configuration = 0u; configuration < dim_N; configuration++) {
            Configuration conf(configuration);

            for(const auto& conf_and_value : this->apply(conf)) {
                if(conf_and_value.second != 0.0) {
                    result(static_cast<unsigned int>(conf_and_value.first), configuration) += conf_and_value.second;
                }
            }
        }

        return result;
    }

#endif

    inline Coefficient trace(const unsigned int N) const {
        return pow(2, N) * (*this)[QuantumString()];
    }

    inline bool commutes_with(const This& other) const {
        if(QuantumString::QuantumOperator::is_pauli_operator()) {
            for(const auto& my_term : *this) {
                for(const auto& other_term : other) {
                    if(!my_term.first.commutes_with(other_term.first)) {
                        return false;
                    }
                }
            }
        }
        else {
            for(const auto& my_term : *this) {
                for(const auto& other_term : other) {
                    const auto a = This(my_term);
                    const auto b = This(other_term);
                    if(a * b - b * a != 0.0) {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    inline This extract_noncommuting_with(const This& other) const {
        This result;

        if(QuantumString::QuantumOperator::is_pauli_operator()) {
            for(const auto& my_term : *this) {
                for(const auto& other_term : other) {
                    if(!my_term.first.commutes_with(other_term.first)) {
                        if(result.terms.find(my_term.first) == result.terms.end()) {
                            result.insert(my_term);
                        }
                    }
                }
            }
        }
        else {
            for(const auto& my_term : *this) {
                for(const auto& other_term : other) {
                    const auto a = This(my_term);
                    const auto b = This(other_term);
                    if(a * b - b * a != 0.0) {
                        result.insert(my_term);
                    }
                }
            }
        }

        return result;
    }

    inline Coefficient expectation_value_of_plus_x_state() const {
        Coefficient result = 0.0;

        for(const auto& term : *this) {
            bool has_no_sigma_yz = true;
            for(const auto& symbol : term.first) {
                if(symbol.op.type != 1u) {
                    has_no_sigma_yz = false;
                    break;
                }
            }

            if(has_no_sigma_yz) {
                result += term.second;
            }
        }

        return result;
    }

    inline Coefficient get_coefficient() const {
        assert(this->size() <= 2u);

        if(this->terms.empty()) {
            return 0.0;
        }

        return this->begin()->second;
    }

    inline void set_coefficient(const Coefficient& value) {
        assert(this->size() <= 2u);

        if(this->terms.empty()) {
            this->insert({QuantumString(), value});
        }
        else {
            this->begin()->second = value;
        }
    }

    inline QuantumString get_quantum_string() const {
        assert(this->size() <= 2u);

        if(this->terms.empty()) {
            return QuantumString();
        }

        return this->begin()->first;
    }

    inline This real() const {
        This result;
        result.terms.reserve(this->size());

        for(const auto& term : *this) {
            const auto coefficient = term.second.real();
            if(coefficient != 0.0) {
                result.insert({term.first, coefficient});
            }
        }

        return result;
    }

    inline This imag() const {
        This result;
        result.terms.reserve(this->size());

        for(const auto& term : *this) {
            const auto coefficient = term.second.imag();
            if(coefficient != 0.0) {
                result.insert({term.first, coefficient});
            }
        }

        return result;
    }

    inline bool is_numeric() const {
        if(this->size() > 1u)
            return false;

        if(this->terms.empty())
            return true; // equals zero

        return this->begin()->first == QuantumString();
    }

    inline typename Terms::iterator begin() {
        return this->terms.begin();
    }

    inline typename Terms::iterator end() {
        return this->terms.end();
    }

    inline typename Terms::const_iterator begin() const {
        return this->terms.begin();
    }

    inline typename Terms::const_iterator end() const {
        return this->terms.end();
    }

    inline term_iterator begin_terms() const {
        return term_iterator(this->terms.begin());
    }

    inline term_iterator end_terms() const {
        return term_iterator(this->terms.end());
    }

};


template<typename QuantumString, typename Coefficient>
ostream& operator<<(ostream& os, const QuantumExpression<QuantumString, Coefficient>& x) {
    os << x.str();
    return os;
}


template<typename QuantumString, typename Coefficient>
inline QuantumExpression<QuantumString, Coefficient> operator+(
    const QuantumExpression<QuantumString, Coefficient>& a,
    const QuantumExpression<QuantumString, Coefficient>& b
) {
    QuantumExpression<QuantumString, Coefficient> result = a;
    result += b;
    return result;
}

template<typename QuantumString, typename Coefficient>
inline QuantumExpression<QuantumString, Coefficient> operator+(
    const QuantumExpression<QuantumString, Coefficient>& x,
    const typename QuantumExpression<QuantumString, Coefficient>::Coefficient& number
) {
    return x + QuantumExpression<QuantumString, Coefficient>(number);
}

template<typename QuantumString, typename Coefficient>
inline QuantumExpression<QuantumString, Coefficient> operator+(
    const typename QuantumExpression<QuantumString, Coefficient>::Coefficient& number,
    const QuantumExpression<QuantumString, Coefficient>& x
) {
    return QuantumExpression<QuantumString, Coefficient>(number) + x;
}

template<typename QuantumString, typename Coefficient>
inline QuantumExpression<QuantumString, Coefficient> operator-(
    const QuantumExpression<QuantumString, Coefficient>& a,
    const QuantumExpression<QuantumString, Coefficient>& b
) {
    QuantumExpression<QuantumString, Coefficient> result = a;
    result -= b;
    return result;
}

template<typename QuantumString, typename Coefficient>
inline QuantumExpression<QuantumString, Coefficient> operator-(
    const QuantumExpression<QuantumString, Coefficient>& x,
    const typename QuantumExpression<QuantumString, Coefficient>::Coefficient& number
) {
    return x - QuantumExpression<QuantumString, Coefficient>(number);
}

template<typename QuantumString, typename Coefficient>
inline QuantumExpression<QuantumString, Coefficient> operator-(
    const typename QuantumExpression<QuantumString, Coefficient>::Coefficient& number,
    const QuantumExpression<QuantumString, Coefficient>& x
) {
    return QuantumExpression<QuantumString, Coefficient>(number) - x;
}

template<typename QuantumString, typename Coefficient>
inline QuantumExpression<QuantumString, Coefficient> mul(
    const QuantumExpression<QuantumString, Coefficient>& a,
    const QuantumExpression<QuantumString, Coefficient>& b,
    const double threshold = 0.0
) {
    QuantumExpression<QuantumString, Coefficient> result;
    result.terms.reserve(2 * a.terms.size() * b.terms.size());

    for(const auto& b_term : b) {
        QuantumExpression<QuantumString, Coefficient> a_times_b_term;

        for(const auto& a_term : a) {
            if(abs(a_term.second * b_term.second) < threshold) {
                continue;
            }

            for(const auto& new_term : a_term.first * b_term.first) {
                a_times_b_term += QuantumExpression<QuantumString, Coefficient>(
                    new_term.first,
                    new_term.second * a_term.second * b_term.second
                );
            }
        }

        result += a_times_b_term;
    }

    return result;
}

template<typename QuantumString, typename Coefficient>
inline QuantumExpression<QuantumString, Coefficient> operator*(
    const QuantumExpression<QuantumString, Coefficient>& a,
    const QuantumExpression<QuantumString, Coefficient>& b
) {
    return mul(a, b);
}

template<typename QuantumString, typename Coefficient>
inline QuantumExpression<QuantumString, Coefficient> operator*(
    const QuantumExpression<QuantumString, Coefficient>& x,
    const typename QuantumExpression<QuantumString, Coefficient>::Coefficient& number
) {
    QuantumExpression<QuantumString, Coefficient> result = x;
    result *= number;
    return result;
}

template<typename QuantumString, typename Coefficient>
inline QuantumExpression<QuantumString, Coefficient> operator*(
    const typename QuantumExpression<QuantumString, Coefficient>::Coefficient& number,
    const QuantumExpression<QuantumString, Coefficient>& x
) {
    return x * number;
}


using PauliExpression = QuantumExpression<PauliString>;
using FermionExpression = QuantumExpression<FermionString>;

} // quantum_expression
