// -------------------------------------------------------------
/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
// -------------------------------------------------------------
/**
 * @file   linear_solver_backend.cpp
 * @brief  Implementation of the runtime LinearSolver backend selection.
 */
// -------------------------------------------------------------

#include "linear_solver_backend.hpp"

#include <algorithm>
#include <cctype>

#ifdef GRIDPACK_WITH_CUDSS
#include <cuda_runtime.h>
#endif

namespace gridpack {
namespace math {

// The requested default backend (process-wide).  Defaults to PETSc so that any
// program which never calls setDefaultLinearSolverBackend() is unaffected.
static LinearSolverBackend p_defaultBackend = LinearSolverBackend::PETSc;

// -------------------------------------------------------------
// linearSolverBackendName
// -------------------------------------------------------------
const char *
linearSolverBackendName(LinearSolverBackend b)
{
  switch (b) {
  case LinearSolverBackend::CuDSS: return "cudss";
  case LinearSolverBackend::PETSc: /* fallthrough */
  default:                         return "petsc";
  }
}

// -------------------------------------------------------------
// linearSolverBackendFromString
// -------------------------------------------------------------
LinearSolverBackend
linearSolverBackendFromString(const std::string& name)
{
  std::string s(name);
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  // trim surrounding whitespace
  const std::string ws(" \t\r\n");
  const std::string::size_type b = s.find_first_not_of(ws);
  const std::string::size_type e = s.find_last_not_of(ws);
  if (b != std::string::npos) {
    s = s.substr(b, e - b + 1);
  } else {
    s.clear();
  }

  if (s == "cudss" || s == "cuds" || s == "gpu") {
    return LinearSolverBackend::CuDSS;
  }
  // "petsc", "cpu", empty, or anything unrecognized -> PETSc (safe default)
  return LinearSolverBackend::PETSc;
}

// -------------------------------------------------------------
// setDefaultLinearSolverBackend
// -------------------------------------------------------------
void
setDefaultLinearSolverBackend(LinearSolverBackend b)
{
  p_defaultBackend = b;
}

void
setDefaultLinearSolverBackend(const std::string& name)
{
  p_defaultBackend = linearSolverBackendFromString(name);
}

// -------------------------------------------------------------
// defaultLinearSolverBackend
// -------------------------------------------------------------
LinearSolverBackend
defaultLinearSolverBackend(void)
{
  return p_defaultBackend;
}

// -------------------------------------------------------------
// cudssBackendAvailable
// -------------------------------------------------------------
bool
cudssBackendAvailable(void)
{
#ifdef GRIDPACK_WITH_CUDSS
  // Memoize: querying the driver repeatedly is unnecessary and, once the
  // answer is known, must not change during a run.
  static int p_cached = -1;
  if (p_cached < 0) {
    int ndev = 0;
    cudaError_t err = cudaGetDeviceCount(&ndev);
    p_cached = (err == cudaSuccess && ndev > 0) ? 1 : 0;
    // Clear any sticky error so it does not leak into later CUDA calls.
    if (err != cudaSuccess) {
      cudaGetLastError();
    }
  }
  return p_cached == 1;
#else
  return false;
#endif
}

// -------------------------------------------------------------
// resolveLinearSolverBackend
// -------------------------------------------------------------
LinearSolverBackend
resolveLinearSolverBackend(void)
{
  if (p_defaultBackend == LinearSolverBackend::CuDSS && cudssBackendAvailable()) {
    return LinearSolverBackend::CuDSS;
  }
  return LinearSolverBackend::PETSc;
}

} // namespace math
} // namespace gridpack
