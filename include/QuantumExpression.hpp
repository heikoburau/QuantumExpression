#pragma once

#include "FastPauliString.hpp"
#include "QuantumString.hpp"
#include "Spins.hpp"

#ifndef NO_PYTHON

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xeval.hpp"

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
    typedef unordered_map<
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
    inline explicit QuantumExpression(const int index, const int type) {
        map<int, int> quantum_string = {{index, type}};
        this->insert({quantum_string, 1.0});
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


#ifndef NO_PYTHON
    static inline QuantumExpression<FastPauliString, Coefficient> from_pauli_vector(const xt::pytensor<Coefficient, 1u>& vector) {
        QuantumExpression<FastPauliString, Coefficient> result;
        result.reserve(vector.size());

        for(auto conf_idx = 0u; conf_idx < vector.size(); conf_idx++) {
            result.add(vector(conf_idx), FastPauliString::enumerate(conf_idx));
        }

        return result;
    }

#endif // NO_PYTHON

    inline void assign(const Coefficient& x) {
        for(auto& term : *this) {
            term.second = x;
        }
    }

    inline This dagger() const {
        This result;
        result.reserve(this->size());

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
        if(this->empty()) {
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

    inline Coefficient __getitem__(const QuantumString& quantum_string) const {
        return (*this)[quantum_string];
    }

    inline void __setitem__(const QuantumString& quantum_string, const Coefficient& value) {
        const auto search = this->terms.find(quantum_string);

        if(search != this->end()) {
            search->second = value;
        }
    }

    inline size_t size() const {
        return this->terms.size();
    }

    inline void insert(const Term& term) {
        if(term.second != 0.0) {
            this->terms.insert(term);
        }
    }

    inline bool empty() const {
        return this->terms.empty();
    }

    inline void reserve(unsigned int count) {
        if(count > 10000u) {
            this->terms.max_load_factor(2);
        }
        else {
            this->terms.max_load_factor(1);
        }
        this->terms.reserve(count);
    }

    inline string str() const {
        stringstream result;

        if(this->empty()) {
            result << "0 * \"\"\n";
            return result.str();
        }

        vector<Term> sorted_terms(this->begin(), this->end());
        sort(
            sorted_terms.begin(),
            sorted_terms.end(),
            [](const Term& a, const Term& b) {
                return abs(a.second) > abs(b.second);
            }
        );

        auto indent = 0u;
        for(const auto& term : sorted_terms) {
            stringstream coeff_str;
            coeff_str.precision(5);
            coeff_str << term.second;
            const auto length = coeff_str.str().length();
            if(length > indent) {
                indent = length;
            }
        }

        result.precision(5);
        for(const auto& term : sorted_terms) {
            result << right << setw(indent + 1) << term.second << " * " << term.first.str() << endl;
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

    inline void add(const Coefficient& coefficient, const QuantumString& quantum_string) {
        auto search = this->terms.find(quantum_string);

        if(search == this->terms.end()) {
            // TODO: try using `emplace`
            this->terms.insert({quantum_string, coefficient});
        }
        else {
            search->second += coefficient;
        }
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

    inline This operator/=(const Coefficient& number) {
        for(auto& term : *this) {
            term.second /= number;
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

    inline This translationally_invariant(const unsigned int length) const {
        This result;
        result.reserve(this->size());

        for(const auto& term : *this) {
            result.add(
                term.second,
                term.first.rotate_to_smallest(length)
            );
        }

        return result;
    }

    inline This apply_threshold(const double threshold) const {
        This result;
        result.reserve(this->size());

        for(const auto& term : *this) {
            if(abs(term.second) >= threshold) {
                result.terms.insert(term);
            }
        }

        return result;
    }

    inline This roll(unsigned int shift, unsigned int length) const {
        This result;
        result.reserve(this->size());

        for(const auto& term : *this) {
            result.add(
                term.second,
                term.first.roll(shift, length)
            );
        }

        return result;
    }

    inline This crop(const double threshold) const {
        return this->apply_threshold(threshold);
    }

    inline This crop_rotation(const double threshold) const {
        const auto cropped = this->crop(threshold);
        const auto norm = cropped.l2_norm();
        if(norm == 0.0) {
            return QuantumExpression(1.0);
        }

        return cropped / norm;
    }


    inline This crop_rotation_generator(const double threshold) const {
        const auto beta = this->abs2();
        const auto cropped = this->crop(threshold);
        const auto beta_c = cropped.abs2();
        if(beta_c == 0.0) {
            return QuantumExpression(0.0);
        }

        return cropped * sqrt(beta / beta_c);
    }

    inline This exp(const double threshold=0.0) const {
        This result(1.0);

        for(const auto& term : *this) {
            const This x = {
                Term({QuantumString(), cosh(term.second)}),
                Term({term.first, sinh(term.second)})
            };

            result = result * x;

            if(threshold > 0.0) {
                result = result.crop_rotation(threshold);
            }
        }

        return result;
    }

    inline This transform(const This& generator, const double exp_threshold=0.0, const double threshold=0.0) const {
        // CAUTION: The `exp_threshold` parameter belongs to the cutoff of the exponentiated `generator` and
        // not to the final result.

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

            auto rotated_term = relevant_generator.exp(exp_threshold) * This(term);
            if(threshold > 0.0) {
                rotated_term = rotated_term.apply_threshold(threshold);
            }

            result += rotated_term;
        }

        return result;
    }

    inline This rotate_by(const This& generator, const double exp_threshold=0.0, const double threshold=0.0) const {
        return this->transform(generator, exp_threshold, threshold);
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

    inline double abs2() const {
        auto result = 0.0;
        for(const auto& term : *this) {
            result += norm(term.second);
        }
        return result;
    }

    inline double l2_norm() const {
        return sqrt(this->abs2());
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

    decltype(auto) matrix(const unsigned int N, const string basis) const {
        // const auto dim_N = 1u << N;
        auto dim_N = 0u;
        if(basis == string("spins")) {
            dim_N = 1u << N;
        }
        else if(basis == string("paulis")) {
            dim_N = 1u << (2u * N);
        }

        xt::pytensor<Coefficient, 2u> result({
            (long int)dim_N, (long int)dim_N
        });
        result = xt::zeros<Coefficient>(result.shape());

        if(basis == string("spins")) {
            for(auto configuration = 0u; configuration < dim_N; configuration++) {
                Configuration conf(configuration);

                for(const auto& conf_and_value : this->apply(conf)) {
                    if(conf_and_value.second != 0.0) {
                        result(static_cast<unsigned int>(conf_and_value.first), configuration) += conf_and_value.second;
                    }
                }
            }
        }
        else if(basis == string("paulis")) {
            for(auto conf_idx = 0u; conf_idx < dim_N; conf_idx++) {
                const auto conf = QuantumString::enumerate(conf_idx);

                for(const auto& term : (*this) * conf) {
                    if(term.second != 0.0) {
                        result(term.first.enumeration_index(), conf_idx) += term.second;
                    }
                }
            }
        }

        return result;
    }

    decltype(auto) apply_on_state(const xt::pytensor<Coefficient, 1u>& state) const {
        const auto dim_N = state.size();
        const auto shape = array<size_t, 1u>{(size_t)dim_N};

        xt::xtensor<Coefficient, 1u> result(shape);
        result = xt::zeros<Coefficient>(shape);
        xt::xtensor<Coefficient, 1u> in(shape);
        xt::xtensor<Coefficient, 1u> out(shape);

        for(const auto& term : *this) {
            in = state;

            term.first.apply(out.data(), in.data(), dim_N);
            result += xt::eval(term.second * out);
        }

        return xt::pytensor<Coefficient, 1u>(result);
    }

    decltype(auto) get_vector() const {
        using shape_type = typename xt::pytensor<Coefficient, 1u>::shape_type;

        xt::pytensor<Coefficient, 1u> result(shape_type{(long int)this->size()});

        auto i = 0u;
        for(const auto& term : *this) {
            result[i] = term.second;

            i++;
        }

        return result;
    }

    void set_vector(const xt::pytensor<Coefficient, 1u>& vec) {
        auto i = 0u;
        for(auto& term : *this) {
            term.second = vec[i];

            i++;
        }
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
            if(term.first.has_no_sigma_yz()) {
                result += term.second;
            }

        }

        return result;
    }

    inline Coefficient vacuum_expectation_value() const {
        return (*this)[QuantumString()];
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
        result.reserve(this->size());

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
        result.reserve(this->size());

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
inline ostream& operator<<(ostream& os, const QuantumExpression<QuantumString, Coefficient>& x) {
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
    result.reserve(a.size() * b.size());

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

template<typename Coefficient>
inline QuantumExpression<FastPauliString, Coefficient> mul(
    const QuantumExpression<FastPauliString, Coefficient>& a,
    const QuantumExpression<FastPauliString, Coefficient>& b,
    const double threshold = 0.0
) {
    QuantumExpression<FastPauliString, Coefficient> result;
    result.reserve(a.size() * b.size());

    for(const auto& b_term : b) {
        for(const auto& a_term : a) {
            if(abs(a_term.second * b_term.second) < threshold) {
                continue;
            }

            const auto factor_and_string = a_term.first * b_term.first;

            result += QuantumExpression<FastPauliString, Coefficient>(
                factor_and_string.second,
                factor_and_string.first * a_term.second * b_term.second
            );
        }
    }

    return result;
}

template<typename QuantumString, typename Coefficient>
inline QuantumExpression<QuantumString, Coefficient> pow(
    const QuantumExpression<QuantumString, Coefficient>& base,
    const unsigned int exponent
) {
        QuantumExpression<QuantumString, Coefficient> result(1.0);

        for(auto i = 0u; i < exponent; i++) {
            result = mul(base, result);
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

template<typename QuantumString, typename Coefficient>
inline QuantumExpression<QuantumString, Coefficient> operator/(
    const QuantumExpression<QuantumString, Coefficient>& x,
    const typename QuantumExpression<QuantumString, Coefficient>::Coefficient& number
) {
    QuantumExpression<QuantumString, Coefficient> result = x;
    result /= number;
    return result;
}

template<typename QuantumString, typename Coefficient>
inline QuantumExpression<QuantumString, Coefficient> operator*(
    const QuantumExpression<QuantumString, Coefficient>& a,
    const QuantumString& b
) {
    // not implemented;

    return QuantumExpression<QuantumString, Coefficient>();
}


template<typename Coefficient>
inline QuantumExpression<FastPauliString, Coefficient> operator*(
    const QuantumExpression<FastPauliString, Coefficient>& a,
    const FastPauliString& b
) {
    QuantumExpression<FastPauliString, Coefficient> result;
    result.reserve(a.size());

    for(const auto& term : a) {
        const auto factor_and_string = term.first * b;

        result.add(term.second * factor_and_string.first, factor_and_string.second);
    }

    return result;
}

template<typename Coefficient>
inline QuantumExpression<FastPauliString, Coefficient> operator*(
    const FastPauliString& a,
    const QuantumExpression<FastPauliString, Coefficient>& b
) {
    QuantumExpression<FastPauliString, Coefficient> result;
    result.reserve(b.size());

    for(const auto& term : b) {
        const auto factor_and_string = a * term.first;

        result.add(term.second * factor_and_string.first, factor_and_string.second);
    }

    return result;
}


using PauliExpression = QuantumExpression<FastPauliString>;
using FermionExpression = QuantumExpression<FermionString>;

} // quantum_expression
