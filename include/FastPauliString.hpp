#include "PauliOperator.hpp"

#include <cstdint>
#include <initializer_list>
#include <forward_list>
#include <utility>
#include <iostream>
#include <complex>
#include <iterator>
#include <vector>
#include <map>
#include <string>


namespace quantum_expression {


using namespace std;
using namespace complex_literals;


template<typename T>
inline decltype(auto) bit_count(const T& x) {
    return __builtin_popcountll(x);
}


struct FastPauliString {
    using dtype = uint64_t;

    dtype a, b;

    using Configuration = typename PauliOperator::Configuration;
    using QuantumOperator = PauliOperator;

    struct symbol_iterator : public iterator<forward_iterator_tag, pair<int, int>>
    {
        dtype a, b;
        unsigned int index;

        inline symbol_iterator(const dtype& a, const dtype& b, const unsigned int index)
        : a(a), b(b), index(index) {
            auto type = (*(*this)).second;
            while(this->index < 64u && type == 0u) {
                this->index++;
                type = (*(*this)).second;
            }
        }

        inline bool operator==(const symbol_iterator& other) const {
            return this->index == other.index;
        }

        inline bool operator!=(const symbol_iterator& other) const {
            return this->index != other.index;
        }

        inline symbol_iterator& operator++() {
            auto type = 0u;
            while(this->index < 64u && type == 0u) {
                this->index++;
                type = (*(*this)).second;
            }

            return *this;
        }

        inline pair<int, int> operator*() const {
            const auto type = (
                int(bool(this->a & (1lu << this->index))) |
                (int(bool(this->b & (1lu << this->index))) << 1lu)
            );

            return {this->index, type};
        }
    };

    inline FastPauliString() : a(0), b(0) {};
    inline FastPauliString(const dtype& a, const dtype& b) : a(a), b(b) {};

    inline FastPauliString(initializer_list<pair<int, PauliOperator>> symbols) {
        this->a = this->b = dtype(0u);

        for(const auto& x : symbols) {
            this->set_at(x.first, x.second.type);
        }
    }

    inline FastPauliString(const map<int, int>& symbols) {
        this->a = this->b = dtype(0u);

        for(const auto& symbol : symbols) {
            this->set_at(symbol.first, symbol.second);
        }
    }

    inline void set_at(const unsigned int idx, const unsigned int type) {
        if(idx > 63u) {
            cerr << "[FastPauliString] index " << idx << " exceeds capacity (64 sites)." << endl;
            return;
        }

        switch(type)
        {
        case 0u:
            this->a &= ~(1lu << idx);
            this->b &= ~(1lu << idx);
            break;
        case 1lu:
            this->a |= 1lu << idx;
            this->b &= ~(1lu << idx);
            break;
        case 2u:
            this->a &= ~(1lu << idx);
            this->b |= 1lu << idx;
            break;
        case 3u:
            this->a |= 1lu << idx;
            this->b |= 1lu << idx;
        }
    }

    inline unsigned int operator[](const unsigned int idx) const {
        return (
            int(bool(this->a & (1lu << idx))) |
            (int(bool(this->b & (1lu << idx))) << 1lu)
        );
    }

    inline pair<FastPauliString, double> dagger() const {
        return {FastPauliString(*this), 1.0};
    }

    inline bool contains(const int index) const {
        return (*this)[index];
    }

    inline size_t size() const {
        return bit_count(this->a | this->b);
    }

    inline operator bool() const {
        return this->size();
    }

    inline bool cast_to_bool() const {
        return *this;
    }

    inline bool operator==(const FastPauliString& other) const {
        return (this->a == other.a) && (this->b == other.b);
    }

    inline bool operator!=(const FastPauliString& other) const {
        return (this->a != other.a) || (this->b != other.b);
    }

    inline bool operator<(const FastPauliString& other) const {
        if(this->a == other.a) {
            return this->b < other.b;
        }
        else {
            return this->a < other.a;
        }
    }

    inline dtype is_non_trivial() const {
        return this->a | this->b;
    }

    inline dtype is_different(const FastPauliString& other) const {
        return (this->a ^ other.a) | (this->b ^ other.b);
    }

    inline dtype is_sigma_x() const {
        return (this->a) & (~this->b);
    }

    inline dtype is_sigma_y() const {
        return (~this->a) & (this->b);
    }

    inline dtype is_sigma_z() const {
        return (this->a) & (this->b);
    }

    inline dtype is_diagonal_bitwise() const {
        return ~(this->a ^ this->b);
    }

    inline bool is_diagonal() const {
        return !(this->a ^ this->b);
    }

    inline bool has_no_sigma_yz() const {
        return !(this->is_sigma_y() | this->is_sigma_z());
    }

    inline dtype epsilon_is_negative(const FastPauliString& other) const {
        return (
            (this->is_sigma_x() & other.is_sigma_z()) |
            (this->is_sigma_y() & other.is_sigma_x()) |
            (this->is_sigma_z() & other.is_sigma_y())
        );
    }

    inline bool commutes_with(const FastPauliString& other) const {
        return !(
            bit_count(
                this->is_non_trivial() & other.is_non_trivial() & this->is_different(other)
            ) & 1lu
        );
    }

    inline complex<double> complex_prefactor() const {
        complex<double> result = 1.0;

        const auto num_sigma_y = bit_count(this->is_sigma_y());

        // is there a factor i*i=-1 left?
        if((num_sigma_y & 3u) > 1lu) {
            result *= -1.0;
        }

        if(num_sigma_y & 1lu) {
            result *= 1.0i;
        }

        return result;
    }

    inline pair<Configuration, complex<double>> apply(Configuration conf) const {
        complex<double> factor = this->complex_prefactor();

        if(bit_count((~conf) & (this->is_sigma_z() | this->is_sigma_y())) & 1lu) {
            factor *= -1.0;
        }

        conf.configuration ^= (~this->is_diagonal_bitwise());

        return {conf, factor};
    }

    inline void apply(complex<double>* out_state, complex<double>* in_state, const unsigned int dim_N) const {
        const auto prefactor = this->complex_prefactor();
        const auto is_flipping = ~this->is_diagonal_bitwise();
        const auto is_sigma_yz = this->is_sigma_y() | this->is_sigma_z();

        for(auto conf = 0u; conf < dim_N; conf++) {
            const auto factor = bit_count((~conf) & is_sigma_yz) & 1lu ? -1.0 : 1.0;
            out_state[conf ^ is_flipping] = prefactor * factor * in_state[conf];
        }
    }

    inline FastPauliString rotate_to_smallest(unsigned int length) const {
        const auto mask = (1lu << length) - 1lu;
        auto result = *this;
        auto x = *this;

        for(auto i = 0u; i < length; i++) {
            const auto a_shifted = (x.a << 1u) | (x.a >> (length - 1u));
            const auto b_shifted = (x.b << 1u) | (x.b >> (length - 1u));

            x.a = a_shifted & mask;
            x.b = b_shifted & mask;

            if(x < result) {
                result = x;
            }
        }

        return result;
    }

    inline string str() const {
        stringstream result;

        if(!(*this)) {
            result << "\"\"";
            return result.str();
        }

        result << "\"";
        for(auto i = 0u; i < 64u; i++) {
            if(this->contains(i)) {
                result << PauliOperator(static_cast<int>((*this)[i])).str();
            }
            else {
                result << " ";
            }
        }
        result << "\"";

        return result.str();
    }

    inline symbol_iterator begin_symbols() const {
        return symbol_iterator(this->a, this->b, 0u);
    }

    inline symbol_iterator end_symbols() const {
        return symbol_iterator(this->a, this->b, 64u);
    }
};


inline ostream& operator<<(ostream& os, const FastPauliString& pauli_string) {
    os << pauli_string.str();
    return os;
}


inline decltype(auto) operator*(const FastPauliString& a, const FastPauliString& b) {
    complex<double> factor = 1.0;
    if(bit_count(a.epsilon_is_negative(b)) & 1lu) {
        factor *= -1.0;
    }

    const auto num_epsilon = bit_count(
        a.is_non_trivial() & b.is_non_trivial() & a.is_different(b)
    );

    // is there a factor i*i=-1 left?
    if((num_epsilon & 3u) > 1lu) {
        factor *= -1.0;
    }

    if(num_epsilon & 1lu) {
        factor *= 1.0i;
    }

    return make_pair(factor, FastPauliString(a.a ^ b.a, a.b ^ b.b));
}


inline decltype(auto) commutator(const FastPauliString& a, const FastPauliString& b) {
    const auto num_epsilon = bit_count(
        a.is_non_trivial() & b.is_non_trivial() & a.is_different(b)
    );

    if(!(num_epsilon & 1lu)) {
        return make_pair(complex<double>(0.0), FastPauliString());
    }

    complex<double> factor = 2.0i;
    if(bit_count(a.epsilon_is_negative(b)) & 1lu) {
        factor *= -1.0;
    }

    // is there a factor i*i=-1 left?
    if((num_epsilon & 3u) == 3u) {
        factor *= -1.0;
    }

    return make_pair(factor, FastPauliString(a.a ^ b.a, a.b ^ b.b));
}


}  // namespace quantum_expression


namespace std {

template<>
struct hash<quantum_expression::FastPauliString> {
    inline size_t operator()(const quantum_expression::FastPauliString& pauli_string) const {
        // it is crucial to have a different scrambling for a and b.
        // Otherwise all pure sigma_z strings would have the same hash.

        auto result = pauli_string.a ^ (
            (pauli_string.b << 20) | (pauli_string.b >> (64 - 20))
        );
        result ^= result >> 32;
        result ^= (result >> 16) | (result << 16);
        result ^= (result >> 8) | (result << 8);
        return result ^ 4222420308lu;
    }
};

}  // namespace std

