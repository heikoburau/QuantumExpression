#include "functions.hpp"
#include "QuantumExpression.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>

#include <iostream>
#include <complex>


namespace py = pybind11;

using namespace quantum_expression;

// Python Module and Docstrings

PYBIND11_MODULE(QuantumExpression, m)
{
    xt::import_numpy();

    py::class_<PauliString>(m, "PauliString")
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("__bool__", &PauliString::cast_to_bool)
        .def("__len__", &PauliString::size)
        .def("__str__", &PauliString::str)
        .def("__repr__", &PauliString::str)
        .def(
            "__iter__",
            [](const PauliString& x){
                return py::make_iterator(x.begin(), x.end());
            },
            py::keep_alive<0, 1>()
        )
        .def_property_readonly("max_index", &PauliString::max_index)
        .def_property_readonly("min_index", &PauliString::min_index);

    py::class_<PauliExpression>(m, "PauliExpression")
        .def(py::init<const map<int, int>&>())
        .def(py::init<const PauliExpression&>())
        .def(py::self == py::self)
        .def(py::self == complex<double>())
        .def(py::self != py::self)
        .def(py::self != complex<double>())
        .def("__getitem__", &PauliExpression::__getitem__)
        .def("__len__", &PauliExpression::size)
        .def("__str__", &PauliExpression::str)
        .def("__repr__", &PauliExpression::str)
        .def("__abs__", &PauliExpression::absolute)
        .def(py::self + py::self)
        .def(py::self + complex<double>())
        .def(complex<double>() + py::self)
        .def(py::self += py::self)
        .def(py::self - py::self)
        .def(py::self - complex<double>())
        .def(complex<double>() - py::self)
        .def(py::self -= py::self)
        .def(py::self * py::self)
        .def(py::self * complex<double>())
        .def(complex<double>() * py::self)
        .def(+py::self)
        .def(-py::self)
        .def(
            "__iter__",
            [](const PauliExpression& x){
                return py::make_iterator(x.begin_terms(), x.end_terms());
            },
            py::keep_alive<0, 1>()
        )
        .def("exp", &PauliExpression::exp)
        .def("rotate_by", py::overload_cast<const PauliExpression&, const double>(&PauliExpression::rotate_by))
        .def("fast_rotate_by", &PauliExpression::fast_rotate_by)
        .def("apply_threshold", &PauliExpression::apply_threshold)
        .def_property_readonly("max_term", &PauliExpression::max_term)
        .def_property_readonly("max_norm", &PauliExpression::max_norm)
        .def_property_readonly("real", &PauliExpression::real)
        .def_property_readonly("imag", &PauliExpression::imag)
        .def_property_readonly("is_numeric", &PauliExpression::is_numeric)
        .def_property("coefficient", &PauliExpression::get_coefficient, &PauliExpression::set_coefficient)
        .def_property_readonly("pauli_string", &PauliExpression::get_quantum_string)
        .def("matrix", &PauliExpression::matrix)
        .def("commutes_with", &PauliExpression::commutes_with)
        .def("extract_noncommuting_with", &PauliExpression::extract_noncommuting_with)
        .def("expectation_value_of_plus_x_state", &PauliExpression::expectation_value_of_plus_x_state)
        .def("trace", &PauliExpression::trace);


    py::class_<FermionString>(m, "FermionString")
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("__bool__", &FermionString::cast_to_bool)
        .def("__len__", &FermionString::size)
        .def("__str__", &FermionString::str)
        .def("__repr__", &FermionString::str)
        .def(
            "__iter__",
            [](const FermionString& x){
                return py::make_iterator(x.begin(), x.end());
            },
            py::keep_alive<0, 1>()
        )
        .def_property_readonly("max_index", &FermionString::max_index)
        .def_property_readonly("min_index", &FermionString::min_index);

    py::class_<FermionExpression>(m, "FermionExpression")
        .def(py::init<const map<int, int>&>())
        .def(py::init<const FermionExpression&>())
        .def(py::self == py::self)
        .def(py::self == complex<double>())
        .def(py::self != py::self)
        .def(py::self != complex<double>())
        .def("__getitem__", &FermionExpression::__getitem__)
        .def("__len__", &FermionExpression::size)
        .def("__str__", &FermionExpression::str)
        .def("__repr__", &FermionExpression::str)
        .def("__abs__", &FermionExpression::absolute)
        .def(py::self + py::self)
        .def(py::self + complex<double>())
        .def(complex<double>() + py::self)
        .def(py::self += py::self)
        .def(py::self - py::self)
        .def(py::self - complex<double>())
        .def(complex<double>() - py::self)
        .def(py::self -= py::self)
        .def(py::self * py::self)
        .def(py::self * complex<double>())
        .def(complex<double>() * py::self)
        .def(+py::self)
        .def(-py::self)
        .def(
            "__iter__",
            [](const FermionExpression& x){
                return py::make_iterator(x.begin_terms(), x.end_terms());
            },
            py::keep_alive<0, 1>()
        )
        .def_property_readonly("dagger", &FermionExpression::dagger)
        .def("apply_threshold", &FermionExpression::apply_threshold)
        .def_property_readonly("max_term", &FermionExpression::max_term)
        .def_property_readonly("max_norm", &FermionExpression::max_norm)
        .def_property_readonly("real", &FermionExpression::real)
        .def_property_readonly("imag", &FermionExpression::imag)
        .def_property_readonly("is_numeric", &FermionExpression::is_numeric)
        .def_property("coefficient", &FermionExpression::get_coefficient, &FermionExpression::set_coefficient)
        .def_property_readonly("fermion_string", &FermionExpression::get_quantum_string)
        .def("commutes_with", &FermionExpression::commutes_with)
        .def("extract_noncommuting_with", &FermionExpression::extract_noncommuting_with);

    m.def("commutator", commutator<PauliExpression>);
    m.def("commutator", commutator<FermionExpression>);

    m.def("frobenius_norm", frobenius_norm);
    m.def("trace", trace);
    m.def("mul_trace", mul_trace);
}
