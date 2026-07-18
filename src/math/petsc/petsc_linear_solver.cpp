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
 * @brief  Constructs a LinearSolverT, selecting the backend implementation at
 *         run time (PETSc by default, NVIDIA cuDSS when opted-in and available).
 *
 *
 */
// -------------------------------------------------------------

#include "linear_solver.hpp"
#include "linear_solver_backend.hpp"
#include "petsc/petsc_linear_solver_implementation.hpp"

#ifdef GRIDPACK_WITH_CUDSS
#include "cudss/cudss_linear_solver_implementation.hpp"
#endif

namespace gridpack {
namespace math {

// -------------------------------------------------------------
// p_makeLinearSolverImplementation
//
// Choose the concrete LinearSolverImplementation for a coefficient matrix.
// The choice honors the process-wide default backend (set from configuration)
// but degrades gracefully: cuDSS is used only when compiled in AND a CUDA
// device is available, and any failure constructing the cuDSS backend falls
// back to PETSc so the run still completes on the CPU path.
// -------------------------------------------------------------
template <typename T, typename I>
static LinearSolverImplementation<T, I> *
p_makeLinearSolverImplementation(MatrixT<T, I>& A)
{
#ifdef GRIDPACK_WITH_CUDSS
  if (resolveLinearSolverBackend() == LinearSolverBackend::CuDSS) {
    try {
      return new CuDSSLinearSolverImplementation<T, I>(A);
    } catch (const gridpack::Exception& e) {
      // cuDSS could not be initialized for this system -- fall back to PETSc.
    }
  }
#endif
  return new PETScLinearSolverImplementation<T, I>(A);
}


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
    p_solver(p_makeLinearSolverImplementation<T, I>(A))
{
  p_setDistributed(p_solver.get());
  p_setConfigurable(p_solver.get());
  // empty
}

template
LinearSolverT<ComplexType>::LinearSolverT(LinearSolverT<ComplexType>::MatrixType& A);

template
LinearSolverT<RealType>::LinearSolverT(LinearSolverT<RealType>::MatrixType& A);

} // namespace math
} // namespace gridpack
