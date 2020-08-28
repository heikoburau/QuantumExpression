#define CATCH_CONFIG_MAIN

#include "QuantumExpression.hpp"
#include "functions.hpp"
#include "catch2/catch.hpp"
#include <iostream>
#include <math.h>

using namespace quantum_expression;
using namespace std;
using namespace std::complex_literals;

using imap = map<int, int>;


TEST_CASE("Test PauliString") {

    PauliString a = {
        {0, 1}, {1, 2}
    };

    PauliString b = {
        {1, 2}, {0, 1}
    };

    REQUIRE( a == b );

    PauliString c = {
        {2, 2}, {0, 1}
    };

    REQUIRE(c != b);
    REQUIRE(c != a);
}

TEST_CASE("Test PauliString::commutes_with 1") {

    PauliString a = {{0, 1}};
    PauliString b = {{0, 1}};
    PauliString c = {{1, 1}};
    PauliString d = {{0, 2}};

    REQUIRE(a.commutes_with(b));
    REQUIRE(a.commutes_with(c));
    REQUIRE(c.commutes_with(d));

    REQUIRE(!b.commutes_with(d));
    REQUIRE(!a.commutes_with(d));
}

TEST_CASE("Test PauliString::commutes_with 2") {

    PauliString a = {{2, 1}, {3, 1}};
    PauliString b = {{2, 2}, {3, 2}};
    PauliString c = {{3, 2}, {4, 3}};
    PauliString d = {{2, 1}};

    REQUIRE(a.commutes_with(a));
    REQUIRE(a.commutes_with(b));
    REQUIRE(b.commutes_with(c));
    REQUIRE(!a.commutes_with(c));
    REQUIRE(c.commutes_with(d));
    REQUIRE(a.commutes_with(d));
    REQUIRE(!b.commutes_with(d));
}


TEST_CASE("Test FastPauliString::commutes_with 1") {

    FastPauliString a = {{0, 1}};
    FastPauliString b = {{0, 1}};
    FastPauliString c = {{1, 1}};
    FastPauliString d = {{0, 2}};

    REQUIRE(a.commutes_with(b));
    REQUIRE(a.commutes_with(c));
    REQUIRE(c.commutes_with(d));

    REQUIRE(!b.commutes_with(d));
    REQUIRE(!a.commutes_with(d));
}

TEST_CASE("Test FastPauliString::commutes_with 2") {

    FastPauliString a = {{2, 1}, {3, 1}};
    FastPauliString b = {{2, 2}, {3, 2}};
    FastPauliString c = {{3, 2}, {4, 3}};
    FastPauliString d = {{2, 1}};

    REQUIRE(a.commutes_with(a));
    REQUIRE(a.commutes_with(b));
    REQUIRE(b.commutes_with(c));
    REQUIRE(!a.commutes_with(c));
    REQUIRE(c.commutes_with(d));
    REQUIRE(a.commutes_with(d));
    REQUIRE(!b.commutes_with(d));
}

// TEST_CASE("Test PauliString multiplication 1") {

//     PauliString a = {{3, 3}};
//     PauliString b = {{3, 2}};

//     REQUIRE(a * a == make_pair(PauliString(), 1.0+0i));
//     REQUIRE(a * b == make_pair(PauliString({{3, 1}}), -1i));
//     REQUIRE(b * a == make_pair(PauliString({{3, 1}}), 1i));
// }

// TEST_CASE("Test PauliString multiplication 2") {

//     PauliString a = {{0, 2}};
//     PauliString b = {{0, 3}, {1, 1}};
//     PauliString c = {{4, 2}, {6, 1}};

//     REQUIRE(a * b == make_pair(
//         PauliString({{0, 1}, {1, 1}}), 1i
//     ));
//     REQUIRE(b * c == make_pair(
//         PauliString({{0, 3}, {1, 1}, {4, 2}, {6, 1}}), 1.0+0i
//     ));
//     REQUIRE(c * b == make_pair(
//         PauliString({{0, 3}, {1, 1}, {4, 2}, {6, 1}}), 1.0+0i
//     ));
// }

TEST_CASE("Test PauliExpression") {
    PauliExpression a({{0, 2}}, 1.0);
    PauliExpression b({{0, 3}, {1, 1}}, -1.0);
    PauliExpression c({{0, 2}, {6, 1}}, 0.5);
    PauliExpression d({{0, 2}, {1, 1}, {2, 2}, {3, 3}}, 0.5);
    PauliExpression e({{0, 1}, {1, 2}, {2, 3}, {3, 3}}, 0.5);

    using PauliString = typename PauliExpression::QuantumString;

    REQUIRE(a + b == PauliExpression{
        {PauliString{{0, 2}}, 1.0},
        {PauliString{{0, 3}, {1, 1}}, -1.0}
    });

    REQUIRE(a + a - a * 2.0 == PauliExpression());
    REQUIRE(b + b - 2.0 * b == PauliExpression());

    REQUIRE(b * c == 0.5 * PauliExpression{
        {PauliString{{0, 1}, {1, 1}, {6, 1}}, 1i},
    });

    REQUIRE((a * b) * c == a * (b * c));
    REQUIRE(a * (b + c) == a * b + a * c);
    REQUIRE(a * b == -b * a);

    REQUIRE(2 * PauliExpression(1.0) - PauliExpression(2.0) == PauliExpression());
    REQUIRE(PauliExpression(1.0) + 1.0 == PauliExpression(2.0));
    REQUIRE(3.0 - PauliExpression(1.0) + 1.0 == PauliExpression(3.0));

    REQUIRE(a.real() == a);
    REQUIRE((1i * (a + b)).imag() == a + b);

    REQUIRE(PauliExpression() == 0.0);
    REQUIRE(PauliExpression(0.0) == 0.0);
    REQUIRE(PauliExpression(1.0) == 1.0);
    REQUIRE(PauliExpression(0.0) != 1.0);
    REQUIRE(PauliExpression(1.0) != 2.0);
    REQUIRE(PauliExpression(imap{{2, 2}}) != 1.0);

    REQUIRE(PauliExpression().is_numeric());
    REQUIRE(PauliExpression(0.0).is_numeric());
    REQUIRE(PauliExpression(1.0).is_numeric());
    REQUIRE(!PauliExpression(imap{{3, 3}}).is_numeric());

    REQUIRE(PauliExpression(5.0).get_coefficient() == 5.0+0i);
    REQUIRE(((3.0+3.0i) * PauliExpression(imap{{3, 3}, {0, 1}})).get_coefficient() == 3.0+3i);

    REQUIRE(d * e == PauliExpression({{0, 3}, {1, 3}, {2, 1}}, 0.25i));

    REQUIRE(e * e == PauliExpression(0.25));
    REQUIRE(d * d * e == 0.25 * e);
}

TEST_CASE("Test rotate by") {
    PauliExpression a({{0, 1}}, 1.0);

    PauliExpression generator({{0, 2}}, 1i * M_PI / 4.0);

    REQUIRE(a.rotate_by(generator).apply_threshold(1e-10) == PauliExpression(
        {{0, 3}}, 1.0
    ));
}

TEST_CASE("Test max_norm") {
    const auto x = (
        PauliExpression({{1, 1}}, 1.0i) +
        PauliExpression({{3, 1}, {1, 3}}, -5.0i)
    );

    REQUIRE(x.max_norm() == 5.0);
}

TEST_CASE("Test operator[]") {
    const auto x = (
        PauliExpression({{1, 1}}, 1.0i) +
        PauliExpression({{3, 1}, {1, 3}}, -5.0i)
    );

    REQUIRE(x[{{1, 1}}] == 1.0i);
    REQUIRE(x[{{1, 1}, {5, 2}}] == 0.0);
}

TEST_CASE("Test commutator") {
    {
        PauliExpression a({{0, 1}}, 1.0);
        PauliExpression b({{0, 2}}, 1.0);

        REQUIRE(commutator(a, b) == PauliExpression(
            {{0, 3}}, 2i
        ));
    }
    {
        PauliExpression a({{0, 2}, {1, 2}}, 1.0);
        PauliExpression b({{0, 3}, {1, 3}}, 1.0);

        REQUIRE(commutator(a, b) == PauliExpression());
    }
    {
        PauliExpression a({{0, 2}, {1, 2}, {2, 2}}, 1.0);
        PauliExpression b({{0, 3}, {1, 3}, {2, 3}}, 1.0);

        REQUIRE(commutator(a, b) == PauliExpression(
            {{0, 1}, {1, 1}, {2, 1}}, -2i
        ));
    }
}

TEST_CASE("Test apply operator on spins") {
    PauliString a = {{0, 2}};
    PauliString b = {{0, 3}, {1, 1}};
    PauliString c = {{4, 2}, {6, 1}};

    Spins spins(0u);

    REQUIRE(a.apply(spins) == make_pair(Spins(1u), -1i));
    REQUIRE(b.apply(spins) == make_pair(Spins(2u), -1.0+0i));
    REQUIRE(c.apply(spins) == make_pair(Spins((1u << 4) | (1u << 6)), -1i));
}

TEST_CASE("Test apply operator on spins for FastPauliString") {
    FastPauliString a = {{0, 2}};
    FastPauliString b = {{0, 3}, {1, 1}};
    FastPauliString c = {{4, 2}, {6, 1}};

    Spins spins(0u);

    REQUIRE(a.apply(spins) == make_pair(Spins(1u), -1i));
    REQUIRE(b.apply(spins) == make_pair(Spins(2u), -1.0+0i));
    REQUIRE(c.apply(spins) == make_pair(Spins((1u << 4) | (1u << 6)), -1i));
}

TEST_CASE("Test extract_noncommuting_with") {
    PauliExpression a(1.0);
    const auto b = PauliExpression(imap{{0, 2}}) + PauliExpression(imap{{2, 2}});
    const auto c = b + PauliExpression(imap{{4, 2}});
    PauliExpression d(imap{{0, 1}, {2, 1}});

    REQUIRE(a.extract_noncommuting_with(b) == PauliExpression());
    REQUIRE(b.extract_noncommuting_with(c) == PauliExpression());
    REQUIRE(c.extract_noncommuting_with(d) == b);
}

TEST_CASE("Test term iterator") {
    PauliExpression a(imap{{0, 1}});
    PauliExpression b(imap{{0, 2}});
    PauliExpression c(imap{{0, 3}, {4, 2}});

    const auto e = a + b + c;

    auto term_it = e.begin_terms();

    REQUIRE(((*term_it == a) || (*term_it == b) || (*term_it == c)));
    term_it++;
    REQUIRE(((*term_it == a) || (*term_it == b) || (*term_it == c)));
    term_it++;
    REQUIRE(((*term_it == a) || (*term_it == b) || (*term_it == c)));
    term_it++;

    REQUIRE(term_it == e.end_terms());
}


TEST_CASE("Test expectation_value_of_plus_x_state") {
    PauliExpression a(imap{{0, 1}});
    PauliExpression b(imap{{0, 2}});
    PauliExpression c(imap{{0, 3}, {4, 2}});

    REQUIRE(PauliExpression(5.0).expectation_value_of_plus_x_state() == 5.0);
    REQUIRE(a.expectation_value_of_plus_x_state() == 1.0);
    REQUIRE(b.expectation_value_of_plus_x_state() == 0.0);
    REQUIRE((1 + a + b * c).expectation_value_of_plus_x_state() == 2.0);
}


TEST_CASE("Test FermionString") {

    FermionString a = {
        {0, 'c'}, {1, 'a'}
    };

    FermionString b = {
        {1, 'a'}, {0, 'c'}
    };

    REQUIRE( a == b );

    FermionString c = {
        {2, 'n'}, {0, 'a'}
    };

    REQUIRE(c != b);
    REQUIRE(c != a);
}

TEST_CASE("Test FermionExpression multiplication 1") {

    FermionExpression a({{3, 'c'}}, 1.0);
    FermionExpression b({{3, 'a'}}, 1.0);

    REQUIRE(a * a == FermionExpression());
    REQUIRE(a * b == FermionExpression({{3, 'n'}}, 1.0));
    REQUIRE(b * a == FermionExpression(1.0) - FermionExpression({{3, 'n'}}, 1.0));
}


TEST_CASE("Test FermionExpression") {
    FermionExpression a({{0, 'a'}}, 1.0);
    FermionExpression b({{0, 'n'}, {1, 'c'}}, -1.0);
    FermionExpression c({{0, 'a'}, {6, 'c'}}, 0.5);

    REQUIRE(a + b == FermionExpression{
        {FermionString{{0, 'a'}}, 1.0},
        {FermionString{{0, 'n'}, {1, 'c'}}, -1.0}
    });

    REQUIRE(a + a - a * 2.0 == FermionExpression());
    REQUIRE(b + b - 2.0 * b == FermionExpression());

    REQUIRE(c * b == 0.5 * FermionExpression{
        {FermionString{{0, 'a'}, {1, 'c'}, {6, 'c'}}, 1.0},
    });

    REQUIRE((a * b) * c == a * (b * c));
    REQUIRE(a * (b + c) == a * b + a * c);
    REQUIRE(a * b != b * a);

    REQUIRE(2 * FermionExpression(1.0) - FermionExpression(2.0) == FermionExpression());
    REQUIRE(FermionExpression(1.0) + 1.0 == FermionExpression(2.0));
    REQUIRE(3.0 - FermionExpression(1.0) + 1.0 == FermionExpression(3.0));

    REQUIRE(a.real() == a);

    REQUIRE(FermionExpression() == 0.0);
    REQUIRE(FermionExpression(0.0) == 0.0);
    REQUIRE(FermionExpression(1.0) == 1.0);
    REQUIRE(FermionExpression(0.0) != 1.0);
    REQUIRE(FermionExpression(1.0) != 2.0);
    REQUIRE(FermionExpression(imap{{2, 2}}) != 1.0);

    REQUIRE(FermionExpression().is_numeric());
    REQUIRE(FermionExpression(0.0).is_numeric());
    REQUIRE(FermionExpression(1.0).is_numeric());
    REQUIRE(!FermionExpression(imap{{3, 3}}).is_numeric());

    REQUIRE(FermionExpression(5.0).get_coefficient() == 5.0+0i);
    REQUIRE(FermionExpression({{3, 3}, {0, 1}}, 3.0+3.0i).get_coefficient() == 3.0+3i);
}
