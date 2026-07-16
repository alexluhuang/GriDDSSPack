// -------------------------------------------------------------
/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
// -------------------------------------------------------------
/**
 * @file   linear_solver.cpp
 * @author William A. Perkins
 * @date   2015-03-05 13:17:17 d3g096
 * 
 * @brief  
 * 
 * 
 */
// -------------------------------------------------------------

#include "linear_solver.hpp"
#include "cudss/cudss_linear_solver.hpp"
#include "petsc/petsc_linear_solver_implementation.hpp"
#include "gridpack/utilities/exception.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <string>

namespace gridpack {
namespace math {

namespace {

std::string normalizedBackend(const std::string& value)
{
  std::string result(value);
  for (std::string::iterator position = result.begin();
       position != result.end(); ++position) {
    *position = static_cast<char>(
      std::tolower(static_cast<unsigned char>(*position)));
  }
  return result;
}

bool environmentBoolean(const char *name, bool defaultValue)
{
  const char *value = std::getenv(name);
  if (value == NULL || *value == '\0') {
    return defaultValue;
  }

  const std::string normalized = normalizedBackend(value);
  if (normalized == "1" || normalized == "true" ||
      normalized == "yes" || normalized == "on") {
    return true;
  }
  if (normalized == "0" || normalized == "false" ||
      normalized == "no" || normalized == "off") {
    return false;
  }
  throw gridpack::Exception(
    std::string(name) + " must be a boolean value");
}

std::string selectedBackend(
  utility::Configuration::CursorPtr parent,
  const std::string& configurationKey)
{
  std::string backend("petsc");
  if (parent) {
    utility::Configuration::CursorPtr solver =
      parent->getCursor(configurationKey);
    if (solver) {
      backend = solver->get("Backend", backend);
    }
  }

  const char *environment =
    std::getenv("GRIDPACK_LINEAR_SOLVER_BACKEND");
  if (environment != NULL && *environment != '\0') {
    backend = environment;
  }
  return normalizedBackend(backend);
}

bool selectedStrict(
  utility::Configuration::CursorPtr parent,
  const std::string& configurationKey)
{
  bool strict = false;
  if (parent) {
    utility::Configuration::CursorPtr solver =
      parent->getCursor(configurationKey);
    if (solver) {
      strict = solver->get("CUDSSStrict", strict);
    }
  }
  return environmentBoolean("GRIDPACK_CUDSS_STRICT", strict);
}

template <typename T, typename I>
LinearSolverImplementation<T, I> *
createConfiguredImplementation(
  typename LinearSolverT<T, I>::MatrixType& matrix,
  utility::Configuration::CursorPtr parent,
  const std::string& configurationKey,
  const std::string& backend)
{
  if (backend == "petsc") {
    return new PETScLinearSolverImplementation<T, I>(matrix);
  }
  if (backend == "cudss") {
    if (selectedStrict(parent, configurationKey)) {
      throw gridpack::Exception(
        "cuDSS linear solver: strict cuDSS mode is only eligible for "
        "RealType/double/int32 systems");
    }
    return new PETScLinearSolverImplementation<T, I>(matrix);
  }
  throw gridpack::Exception(
    "LinearSolver Backend must be \"petsc\" or \"cudss\"");
}

template <>
LinearSolverImplementation<RealType, int> *
createConfiguredImplementation<RealType, int>(
  LinearSolverT<RealType, int>::MatrixType& matrix,
  utility::Configuration::CursorPtr parent,
  const std::string&,
  const std::string& backend)
{
  if (backend == "petsc") {
    return new PETScLinearSolverImplementation<RealType, int>(matrix);
  }
  if (backend == "cudss") {
    return createCUDSSLinearSolver(matrix, parent);
  }
  throw gridpack::Exception(
    "LinearSolver Backend must be \"petsc\" or \"cudss\"");
}

} // anonymous namespace

// -------------------------------------------------------------
//  class LinearSolver
// -------------------------------------------------------------

// -------------------------------------------------------------
// LinearSolver:: constructors / destructor
// -------------------------------------------------------------
template <typename T, typename I>
LinearSolverT<T, I>::LinearSolverT(LinearSolverT<T, I>::MatrixType& A)
  : parallel::WrappedDistributed(),
    utility::WrappedConfigurable(),
    utility::Uncopyable(),
    p_matrix(A),
    p_solver(new PETScLinearSolverImplementation<T, I>(A))
{
  p_setDistributed(p_solver.get());
  p_setConfigurable(p_solver.get());
  // empty
}

template <typename T, typename I>
void
LinearSolverT<T, I>::p_preconfigure(
  utility::Configuration::CursorPtr props)
{
  const std::string configurationKey = p_solver->configurationKey();
  const std::string backend =
    selectedBackend(props, configurationKey);
  std::unique_ptr<LinearSolverImplementation<T, I> > implementation(
    createConfiguredImplementation<T, I>(
      p_matrix, props, configurationKey, backend));
  implementation->configurationKey(configurationKey);
  p_solver.reset(implementation.release());
  p_setDistributed(p_solver.get());
  p_setConfigurable(p_solver.get());
}

template
LinearSolverT<ComplexType>::LinearSolverT(LinearSolverT<ComplexType>::MatrixType& A);

template
LinearSolverT<RealType>::LinearSolverT(LinearSolverT<RealType>::MatrixType& A);

template
void LinearSolverT<ComplexType>::p_preconfigure(
  utility::Configuration::CursorPtr props);

template
void LinearSolverT<RealType>::p_preconfigure(
  utility::Configuration::CursorPtr props);

} // namespace math
} // namespace gridpack
