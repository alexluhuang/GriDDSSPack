// Emacs Mode Line: -*- Mode:c++;-*-
// -------------------------------------------------------------
/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
// -------------------------------------------------------------
/**
 * @file   linear_solver_backend.hpp
 * @brief  Runtime selection of the LinearSolver implementation backend.
 *
 * GridPACK's math layer uses the pImpl idiom: a LinearSolverT owns a
 * LinearSolverImplementation chosen at construction time (see
 * petsc/petsc_linear_solver.cpp).  Historically the PETSc implementation was
 * hard-wired.  This small, dependency-free layer lets an application select the
 * GPU (NVIDIA cuDSS) direct-LU implementation instead, at run time, without
 * touching any application code or the LinearSolver public interface.
 *
 * Contract
 * --------
 *  * The default backend is PETSc, so a program that never touches this API
 *    behaves exactly as before.
 *  * An application (e.g. the contingency-analysis driver) reads the input.xml
 *    switch and calls setDefaultLinearSolverBackend() ONCE, BEFORE any
 *    LinearSolver is constructed (the pImpl is fixed at construction).
 *  * resolveLinearSolverBackend() never returns CuDSS unless the cuDSS backend
 *    was compiled in (GRIDPACK_WITH_CUDSS) AND a usable CUDA device is present;
 *    otherwise it downgrades to PETSc so the run still succeeds on CPU-only
 *    hardware or a CPU-only build.
 *
 * This header intentionally pulls in no CUDA/PETSc headers so it is safe to
 * include everywhere, including in builds without cuDSS.
 */
// -------------------------------------------------------------

#ifndef _linear_solver_backend_hpp_
#define _linear_solver_backend_hpp_

#include <string>

namespace gridpack {
namespace math {

// -------------------------------------------------------------
//  enum LinearSolverBackend
// -------------------------------------------------------------
/// Which underlying library implements a LinearSolver.
enum class LinearSolverBackend {
  PETSc,   ///< CPU direct/iterative solve via PETSc KSP (default, always available)
  CuDSS    ///< GPU direct-LU solve via NVIDIA cuDSS (opt-in, requires GRIDPACK_WITH_CUDSS + GPU)
};

/// Human-readable name of a backend ("petsc" / "cudss").
const char *linearSolverBackendName(LinearSolverBackend b);

/// Parse a backend name (case-insensitive).
/**
 * Accepts "petsc" and "cudss" (any case).  Unknown / empty strings map to
 * PETSc so a typo can never silently disable the CPU fallback.
 *
 * @param name backend name from configuration
 * @return the parsed backend (PETSc if unrecognized)
 */
LinearSolverBackend linearSolverBackendFromString(const std::string& name);

/// Set the process-wide default LinearSolver backend.
/**
 * Must be called before any LinearSolver is constructed to take effect.  Safe
 * to call from a program with no GPU: the request is only honored when usable
 * (see resolveLinearSolverBackend()).
 */
void setDefaultLinearSolverBackend(LinearSolverBackend b);

/// Set the process-wide default LinearSolver backend by name.
void setDefaultLinearSolverBackend(const std::string& name);

/// Get the requested process-wide default LinearSolver backend.
LinearSolverBackend defaultLinearSolverBackend(void);

/// True iff the cuDSS backend is compiled in AND a CUDA device is visible.
/**
 * Result is memoized after the first call.  Always false when built without
 * GRIDPACK_WITH_CUDSS.
 */
bool cudssBackendAvailable(void);

/// Resolve the backend actually used, honoring compile-time and runtime limits.
/**
 * Returns CuDSS only when the requested default is CuDSS and
 * cudssBackendAvailable() is true; otherwise returns PETSc.
 */
LinearSolverBackend resolveLinearSolverBackend(void);

} // namespace math
} // namespace gridpack

#endif
