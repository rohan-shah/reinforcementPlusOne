#include "pybind11_wrapper.h"
#include "game.h"

namespace py = pybind11;

PYBIND11_MODULE(libplusOne_python, m) {
    {
        py::class_<plusOne::Game> c(m, "plusOne");
    }
}
