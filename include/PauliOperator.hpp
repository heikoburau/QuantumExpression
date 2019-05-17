#pragma once

#include "Spins.hpp"
#include <utility>
#include <complex>
#include <iostream>
#include <tuple>


namespace quantum_expression {

using namespace std;
using namespace std::complex_literals;

static const int pauli_table[] = {
    0, 0, 0, 0,
    0, 0, 3, -2,
    0, -3, 0, 1,
    0, 2, -1, 0
};


struct PauliOperator {
    using Configuration = Spins;

    int type;

    PauliOperator(const int type) : type(type) {}
    PauliOperator(const char name) {
        if(name == '1') {
            this->type = 0;
        }
        else if(name == 'x') {
            this->type = 1;
        }
        else if(name == 'y') {
            this->type = 2;
        }
        else if(name == 'z') {
            this->type = 3;
        }
        else {
            cerr << "Invalid Pauli-operator name: " << name << endl;
        }
    }

    inline PauliOperator dagger() {
        return PauliOperator(this->type);
    }

    inline bool operator==(const PauliOperator& other) const {
        return this->type == other.type;
    }

    inline bool operator!=(const PauliOperator& other) const {
        return this->type != other.type;
    }

    inline bool is_identity() const {
        return this->type == 0;
    }

    inline char str() const {
        static const char type_to_char[] = {'1', 'x', 'y', 'z'};

        return type_to_char[this->type];
    }

    inline pair<Spins, complex<double>> apply(const Spins& spins, const int index) const {
        /*
        sigma_x | s > =       | -s >
        sigma_y | s > = i * s | -s >
        sigma_z | s > =     s | s >
        */

        if(this->type == 1)
            return {spins.flip(index), 1.0};
        if(this->type == 2)
            return {spins.flip(index), 1i * spins[index]};
        if(this->type == 3)
            return {spins, spins[index]};

        return {spins, 1.0};
    }

    static constexpr bool is_pauli_operator() {
        return true;
    }
};

inline tuple<PauliOperator, complex<double>, double> operator*(const PauliOperator& a, const PauliOperator& b) {
    if(a == b) {
        return {PauliOperator(0), 1.0, 0.0};
    }

    const auto pauli_type = pauli_table[a.type * 4 + b.type];

    return {PauliOperator(abs(pauli_type)), pauli_type >= 0 ? 1.0i : -1.0i, 0.0};
}

} // quantum_expression
