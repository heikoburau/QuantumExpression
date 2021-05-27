#pragma once

#include <cstdint>


namespace quantum_expression {

struct Spins {
    using type = uint64_t;
    type configuration;

    Spins() = default;
    Spins(type configuration) : configuration(configuration) {}

    inline operator unsigned int() const {
        return static_cast<unsigned int>(configuration);
    }

    inline Spins flip(const int position) const {
        return Spins(this->configuration ^ ((type)1 << position));
    }

    inline double operator[](const int position) const {
        return 2.0 * static_cast<double>(
            static_cast<bool>(this->configuration & ((type)1 << position))
        ) - 1.0;
    }

    inline bool operator==(const Spins& other) const {
        return this->configuration == other.configuration;
    }
};

} // quantum_expression
