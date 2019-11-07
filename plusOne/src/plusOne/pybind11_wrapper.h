/**
 * Copyright (C) 2017 : Children's Medical Research Institute (CMRI)
 *
 * All Rights Reserved. Unauthorized copying of this file, via any medium is
 * strictly prohibited without the written permission of CMRI.
 *
 * Proprietary and confidential.
 */
#pragma once

// This exists to ensure that the compilation units all have the same specialisation
// of stl and eigen
// See: https://github.com/pybind/pybind11/issues/1055
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>