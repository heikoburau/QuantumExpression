#pragma once

#include "Spins.hpp"
#include <utility>
#include <complex>
#include <iostream>
#include <tuple>


namespace quantum_expression {

using namespace std;
using namespace std::complex_literals;

static const int fermion_table[] = {
    0, 1, 2, 3,
    1, -1, 3, -1,
    2, 4, -1, 2,
    3, 1, -1, 3,
};


struct FermionOperator {
    using Configuration = uint64_t;

    int type;

    FermionOperator(const int type) : type(type) {}
    FermionOperator(const char name) {
        if(name == '1') {
            this->type = 0;
        }
        else if(name == 'c') {
            this->type = 1;
        }
        else if(name == 'a') {
            this->type = 2;
        }
        else if(name == 'n') {
            this->type = 3;
        }
        else {
            cerr << "Invalid fermion-operator name: " << name << endl;
        }
    }

    inline FermionOperator dagger() const {
        if(this->type == 3) {
            return FermionOperator(3);
        }
        else if(this->type == 2) {
            return FermionOperator(1);
        }
        else if(this->type == 1) {
            return FermionOperator(2);
        }
        return FermionOperator(0);
    }

    inline pair<Configuration, complex<double>> apply(const Configuration& configuration, const int index) const {
        Configuration mask = 1 << index;

        if(this->type == 1) {
            if(configuration & mask) {
                return {configuration, 0.0};
            }
            return {configuration | mask, 1.0};
        }
        if(this->type == 2) {
            if(!(configuration & mask)) {
                return {configuration, 0.0};
            }
            return {configuration & ~mask, 1.0};
        }
        if(this->type == 3) {
            return {configuration, configuration & mask ? 1.0 : 0.0};
        }

        return {configuration, 1.0};
    }

    inline bool operator==(const FermionOperator& other) const {
        return this->type == other.type;
    }

    inline bool operator!=(const FermionOperator& other) const {
        return this->type != other.type;
    }

    inline bool is_identity() const {
        return this->type == 0;
    }

    inline char str() const {
        static const char type_to_char[] = {'1', 'c', 'a', 'n'};

        return type_to_char[this->type];
    }

    static constexpr bool is_pauli_operator() {
        return false;
    }
};

inline tuple<FermionOperator, complex<double>, double> operator*(const FermionOperator& a, const FermionOperator& b) {
    const auto fermion_type = fermion_table[a.type * 4 + b.type];
    if(fermion_type == -1) {
        return {FermionOperator(0), 0.0, 0.0};
    }
    if(fermion_type == 4) {
        return {FermionOperator(3), -1.0, 1.0};
    }

    return {FermionOperator(fermion_type), 1.0, 0.0};
}

} // quantum_expression
