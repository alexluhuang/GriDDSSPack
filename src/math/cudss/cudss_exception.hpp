// Emacs Mode Line: -*- Mode:c++;-*-
// -------------------------------------------------------------
/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
// -------------------------------------------------------------
/**
 * @file   cudss_exception.hpp
 * @brief  Error-checking helpers for the CUDA runtime, cuDSS and PETSc calls
 *         made by the cuDSS LinearSolver backend.
 *
 * All failures are surfaced as gridpack::Exception so the runtime backend
 * selector can catch them and fall back to the PETSc path (see
 * petsc/petsc_linear_solver.cpp).  This header is only meaningful when the
 * cuDSS backend is compiled in (GRIDPACK_WITH_CUDSS).
 */
// -------------------------------------------------------------

#ifndef _cudss_exception_hpp_
#define _cudss_exception_hpp_

#ifdef GRIDPACK_WITH_CUDSS

#include <string>
#include <sstream>
#include <cuda_runtime.h>
#include <cudss.h>
#include "gridpack/utilities/exception.hpp"

namespace gridpack {
namespace math {

/// Throw a gridpack::Exception describing a failed CUDA runtime call.
inline void checkCuda(cudaError_t err, const char *what,
                      const char *file, int line)
{
  if (err != cudaSuccess) {
    std::ostringstream msg;
    msg << "cuDSS backend: CUDA error at " << file << ":" << line
        << " (" << what << "): " << cudaGetErrorString(err);
    throw gridpack::Exception(msg.str());
  }
}

/// Throw a gridpack::Exception describing a failed cuDSS call.
inline void checkCudss(cudssStatus_t status, const char *what,
                       const char *file, int line)
{
  if (status != CUDSS_STATUS_SUCCESS) {
    std::ostringstream msg;
    msg << "cuDSS backend: cuDSS error at " << file << ":" << line
        << " (" << what << "): status " << static_cast<int>(status);
    throw gridpack::Exception(msg.str());
  }
}

} // namespace math
} // namespace gridpack

/// Wrap a CUDA runtime call; throws gridpack::Exception on failure.
#define GP_CUDA_CHECK(call) \
  ::gridpack::math::checkCuda((call), #call, __FILE__, __LINE__)

/// Wrap a cuDSS call; throws gridpack::Exception on failure.
#define GP_CUDSS_CHECK(call) \
  ::gridpack::math::checkCudss((call), #call, __FILE__, __LINE__)

/// Wrap a PETSc call; throws gridpack::Exception on failure.  (Kept as a macro
/// so PETSc's own CHKERR machinery is not required in this translation unit.)
#define GP_PETSC_CHECK(ierr)                                                    \
  do {                                                                          \
    PetscErrorCode _gp_ierr = (ierr);                                          \
    if (_gp_ierr != 0) {                                                        \
      std::ostringstream _gp_msg;                                              \
      _gp_msg << "cuDSS backend: PETSc error " << _gp_ierr << " at "           \
              << __FILE__ << ":" << __LINE__;                                  \
      throw ::gridpack::Exception(_gp_msg.str());                             \
    }                                                                           \
  } while (0)

#endif // GRIDPACK_WITH_CUDSS

#endif
