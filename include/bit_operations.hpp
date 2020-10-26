#pragma once


namespace quantum_expression {
namespace detail {


inline unsigned int expand_bits_to_even_sites(unsigned int input) {
    return (
          ((input & (1lu <<  0)) <<  0)
        | ((input & (1lu <<  1)) <<  1)
        | ((input & (1lu <<  2)) <<  2)
        | ((input & (1lu <<  3)) <<  3)
        | ((input & (1lu <<  4)) <<  4)
        | ((input & (1lu <<  5)) <<  5)
        | ((input & (1lu <<  6)) <<  6)
        | ((input & (1lu <<  7)) <<  7)
        | ((input & (1lu <<  8)) <<  8)
        | ((input & (1lu <<  9)) <<  9)
        | ((input & (1lu << 10)) << 10)
    );
}

inline unsigned int pick_bits_at_even_sites(unsigned int input) {
    return (
          ((input & (1lu << 0)) >> 0)
        | ((input & (1lu << 2)) >> 1)
        | ((input & (1lu << 4)) >> 2)
        | ((input & (1lu << 6)) >> 3)
        | ((input & (1lu << 8)) >> 4)
        | ((input & (1lu << 10)) >> 5)
        | ((input & (1lu << 12)) >> 6)
        | ((input & (1lu << 14)) >> 7)
        | ((input & (1lu << 16)) >> 8)
        | ((input & (1lu << 18)) >> 9)
        | ((input & (1lu << 20)) >> 10)
    );
}


}  // namespace quantum_expression
}  // namespace detail
