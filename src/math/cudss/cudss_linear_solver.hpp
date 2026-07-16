/*
 * Copyright (c) 2026 Battelle Memorial Institute
 * Licensed under the modified BSD License.
 */

#ifndef GRIDPACK_MATH_CUDSS_LINEAR_SOLVER_HPP_
#define GRIDPACK_MATH_CUDSS_LINEAR_SOLVER_HPP_

#include "gridpack/configuration/configuration.hpp"
#include "gridpack/math/linear_solver_implementation.hpp"

namespace gridpack {
namespace math {

LinearSolverImplementation<RealType, int> *
createCUDSSLinearSolver(
  RealMatrix& matrix,
  utility::Configuration::CursorPtr parentConfiguration);

} // namespace math
} // namespace gridpack

#endif
