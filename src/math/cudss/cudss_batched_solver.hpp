// Emacs Mode Line: -*- Mode:c++;-*-
// -------------------------------------------------------------
/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
// -------------------------------------------------------------
/**
 * @file   cudss_batched_solver.hpp
 * @brief  Solve MANY independent sparse linear systems that share one CSR
 *         sparsity pattern, with a SINGLE symbolic analysis.
 *
 * Every N-1 case is a rank-bounded perturbation of the base network, so all
 * cases in a wave share the base Jacobian pattern; cuDSS analyzes that pattern
 * once (CUDSS_PHASE_ANALYSIS) and then each case is factorized + solved as an
 * ordinary single system reusing that analysis (CUDSS_PHASE_FACTORIZATION /
 * CUDSS_PHASE_SOLVE).
 *
 * NOTE (2026): the earlier implementation drove NVIDIA's cuDSS *batch* API
 * (cudssMatrixCreateBatchCsr, one factorize/solve call over W systems).  On the
 * Texas7k-scale reduced Jacobian that path produced NaN solutions (validated by
 * forcing per-contingency flat output and diffing against the CPU KLU path --
 * the batch API result was NaN while the identical values through the
 * single-system path matched the CPU to 1e-3).  The single-system cuDSS path is
 * the proven-correct backend used by the per-contingency GPU solve, so the wave
 * engine now loops it per case.  The expensive symbolic analysis is still done
 * exactly ONCE for the whole wave (the shared pattern), which is the dominant
 * amortization; per-case factorization+solve of a ~13k reduced system is a few
 * ms and the assembly (not the solve) is the wave's real cost.
 *
 * Real double, 0-based, general matrices.  Decoupled from PETSc (plain int/double
 * CSR in) so it is independently testable.  Only compiled with GRIDPACK_WITH_CUDSS.
 */
// -------------------------------------------------------------

#ifndef _cudss_batched_solver_hpp_
#define _cudss_batched_solver_hpp_

#ifdef GRIDPACK_WITH_CUDSS

#include <vector>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cudss.h>
#include "cudss/cudss_exception.hpp"

namespace gridpack {
namespace math {

// -------------------------------------------------------------
//  class CuDSSBatchedSolver
// -------------------------------------------------------------
/// W independent sparse systems sharing one CSR pattern; one symbolic analysis,
/// per-case factorize+solve on the proven single-system cuDSS path.
class CuDSSBatchedSolver
{
public:

  /// Build a solver for @c batchCount systems sharing one CSR pattern.
  /**
   * @param n           rows/cols of each (square) system
   * @param nnz         nonzeros in the shared pattern
   * @param rowptr      host CSR row offsets (length n+1, 0-based)
   * @param colind      host CSR column indices (length nnz)
   * @param batchCount  number of systems solved together (wave size)
   */
  CuDSSBatchedSolver(int n, int nnz, const int *rowptr, const int *colind,
                     int batchCount)
    : p_n(n), p_nnz(nnz), p_W(batchCount),
      p_handle(NULL), p_config(NULL), p_data(NULL),
      p_A(NULL), p_b(NULL), p_x(NULL),
      p_d_rowptr(NULL), p_d_colind(NULL),
      p_d_values(NULL), p_d_rhs(NULL), p_d_sol(NULL),
      p_valsAll(NULL), p_analyzed(false)
  {
    GP_CUDSS_CHECK(cudssCreate(&p_handle));
    GP_CUDSS_CHECK(cudssConfigCreate(&p_config));
    GP_CUDSS_CHECK(cudssDataCreate(p_handle, &p_data));
    p_alloc(rowptr, colind);
  }

  ~CuDSSBatchedSolver(void)
  {
    if (p_A) cudssMatrixDestroy(p_A);
    if (p_b) cudssMatrixDestroy(p_b);
    if (p_x) cudssMatrixDestroy(p_x);
    cudaFree(p_d_rowptr); cudaFree(p_d_colind);
    cudaFree(p_d_values); cudaFree(p_d_rhs); cudaFree(p_d_sol);
    if (p_data)   cudssDataDestroy(p_handle, p_data);
    if (p_config) cudssConfigDestroy(p_config);
    if (p_handle) cudssDestroy(p_handle);
  }

  int rows(void) const       { return p_n; }
  int nnz(void) const        { return p_nnz; }
  int batchCount(void) const { return p_W; }

  /// Cache the wave's contiguous value buffer [W*nnz] (system k at [k*nnz,...)).
  /// Actual per-case upload happens in solve() so the current values are used.
  void setAllValues(const double *values) { p_valsAll = values; }

  /// (Compatibility) upload one system's values immediately.
  void setValues(int k, const double *values)
  {
    GP_CUDA_CHECK(cudaMemcpy(p_d_values, values, (std::size_t)p_nnz * sizeof(double),
                             cudaMemcpyHostToDevice));
    (void)k;
  }

  /// Symbolic analysis of the shared pattern.
  void analyze(void)
  {
    GP_CUDSS_CHECK(cudssExecute(p_handle, CUDSS_PHASE_ANALYSIS,
                                p_config, p_data, p_A, p_x, p_b));
    p_analyzed = true;
  }

  /// Recycle the solver data object and re-run the (cheap, ~ms) symbolic
  /// analysis.  cuDSS accumulates numeric-factorization state in the data
  /// object across repeated CUDSS_PHASE_FACTORIZATION calls; after a few dozen
  /// refactorizations on the SAME data object the factors degrade and solves
  /// return garbage (observed: case ~3 of a wave suddenly gets an 8e9 step from
  /// a well-conditioned system).  The per-contingency single-system path never
  /// hits this because it builds a fresh solver per contingency.  Calling
  /// reset() per case (or every few cases) gives each a clean data object; the
  /// pattern/analysis is cheap so this costs ~ms per case, not the expensive
  /// reordering re-run of a full rebuild.
  void reset(void)
  {
    if (p_data) { cudssDataDestroy(p_handle, p_data); p_data = NULL; }
    GP_CUDSS_CHECK(cudssDataCreate(p_handle, &p_data));
    analyze();
  }

  /// No-op: with per-case single-system solves the numeric factorization happens
  /// inside solve()/solveOne() reusing the single shared analysis.
  void factorize(void) {}

  /// Factorize + solve ONE system reusing the shared symbolic analysis.  vals
  /// [nnz], rhs [n] -> sol [n].  On a non-finite result sol is zeroed so the
  /// caller detects non-convergence.  Returns true iff the solution is finite.
  bool solveOne(const double *vals, const double *rhs, double *sol)
  {
    const std::size_t n = p_n, nnz = p_nnz;
    try {
      GP_CUDA_CHECK(cudaMemcpy(p_d_values, vals, nnz * sizeof(double),
                               cudaMemcpyHostToDevice));
      GP_CUDSS_CHECK(cudssExecute(p_handle, CUDSS_PHASE_FACTORIZATION,
                                  p_config, p_data, p_A, p_x, p_b));
      GP_CUDA_CHECK(cudaMemcpy(p_d_rhs, rhs, n * sizeof(double),
                               cudaMemcpyHostToDevice));
      GP_CUDSS_CHECK(cudssExecute(p_handle, CUDSS_PHASE_SOLVE,
                                  p_config, p_data, p_A, p_x, p_b));
      GP_CUDA_CHECK(cudaMemcpy(sol, p_d_sol, n * sizeof(double),
                               cudaMemcpyDeviceToHost));
      for (std::size_t i = 0; i < n; ++i)
        if (!std::isfinite(sol[i])) { for (std::size_t j = 0; j < n; ++j) sol[j] = 0.0; return false; }
      return true;
    } catch (...) {
      for (std::size_t j = 0; j < n; ++j) sol[j] = 0.0;
      return false;
    }
  }

  /// Numeric factorization of ONE system, reusing the shared symbolic analysis;
  /// the factors persist in p_data for subsequent solveReuse() calls.  This is the
  /// "constant-factorization" lever: for a WAVE of contingencies -- each a
  /// rank-bounded perturbation of the base -- the base Jacobian is factorized ONCE
  /// here and every case's chord (dishonest-Newton) iterations reuse it via
  /// solveReuse(), turning O(cases x iters) expensive cuDSS factorizations into a
  /// single factorization plus many cheap triangular solves.  A single 48k-reduced
  /// factorization on the GPU costs ~250 ms; a solve costs a few ms.  Returns true
  /// on success.  Only ONE factorization is done per wave, so the factor-drift the
  /// reset() note warns about (dozens of refactorizations on one data object)
  /// cannot occur here.
  bool factorizeValues(const double *vals)
  {
    const std::size_t nnz = p_nnz;
    try {
      GP_CUDA_CHECK(cudaMemcpy(p_d_values, vals, nnz * sizeof(double),
                               cudaMemcpyHostToDevice));
      GP_CUDSS_CHECK(cudssExecute(p_handle, CUDSS_PHASE_FACTORIZATION,
                                  p_config, p_data, p_A, p_x, p_b));
      return true;
    } catch (...) { return false; }
  }

  /// Solve reusing the factorization from the last factorizeValues() (NO
  /// refactor).  rhs [n] -> sol [n]; zeros sol and returns false on a non-finite
  /// result so the caller can route the case to the exact-Newton fallback.
  bool solveReuse(const double *rhs, double *sol)
  {
    const std::size_t n = p_n;
    try {
      GP_CUDA_CHECK(cudaMemcpy(p_d_rhs, rhs, n * sizeof(double),
                               cudaMemcpyHostToDevice));
      GP_CUDSS_CHECK(cudssExecute(p_handle, CUDSS_PHASE_SOLVE,
                                  p_config, p_data, p_A, p_x, p_b));
      GP_CUDA_CHECK(cudaMemcpy(sol, p_d_sol, n * sizeof(double),
                               cudaMemcpyDeviceToHost));
      for (std::size_t i = 0; i < n; ++i)
        if (!std::isfinite(sol[i])) { for (std::size_t j = 0; j < n; ++j) sol[j] = 0.0; return false; }
      return true;
    } catch (...) { for (std::size_t j = 0; j < n; ++j) sol[j] = 0.0; return false; }
  }

  /// Solve every system: rhs [W*n] in, sol [W*n] out (system k at k*n).  Each
  /// case is factorized (with its current values from setAllValues) and solved
  /// reusing the shared symbolic analysis.  A case whose factor/solve yields a
  /// non-finite entry has its solution zeroed so the wave engine detects
  /// non-convergence and routes it to the per-contingency CPU fallback.
  void solve(const double *rhs, double *sol)
  {
    const std::size_t n = p_n, nnz = p_nnz;
    for (int k = 0; k < p_W; ++k) {
      const double *vk = p_valsAll ? (p_valsAll + (std::size_t)k * nnz) : NULL;
      double *sk = sol + (std::size_t)k * n;
      try {
        if (vk)
          GP_CUDA_CHECK(cudaMemcpy(p_d_values, vk, nnz * sizeof(double),
                                   cudaMemcpyHostToDevice));
        GP_CUDSS_CHECK(cudssExecute(p_handle, CUDSS_PHASE_FACTORIZATION,
                                    p_config, p_data, p_A, p_x, p_b));
        GP_CUDA_CHECK(cudaMemcpy(p_d_rhs, rhs + (std::size_t)k * n,
                                 n * sizeof(double), cudaMemcpyHostToDevice));
        GP_CUDSS_CHECK(cudssExecute(p_handle, CUDSS_PHASE_SOLVE,
                                    p_config, p_data, p_A, p_x, p_b));
        GP_CUDA_CHECK(cudaMemcpy(sk, p_d_sol, n * sizeof(double),
                                 cudaMemcpyDeviceToHost));
        bool fin = true; double snorm = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
          if (!std::isfinite(sk[i])) { fin = false; break; }
          double a = sk[i] < 0 ? -sk[i] : sk[i]; if (a > snorm) snorm = a;
        }
        static bool warned2 = false;
        if (!warned2 && k == 0 && std::getenv("GRIDPACK_BATCH_DEBUG")) {
          double vn = 0.0, rn = 0.0;
          if (vk) for (std::size_t i = 0; i < nnz; ++i) { double a = vk[i]<0?-vk[i]:vk[i]; if (a>vn) vn=a; }
          const double *rk = rhs + (std::size_t)k * n;
          for (std::size_t i = 0; i < n; ++i) { double a = rk[i]; if (a<0)a=-a; if (a>rn) rn=a; }
          // residual ||A_k*sol - rhs|| using host pattern + this case's values
          double res = 0.0;
          if (vk && fin) {
            for (std::size_t i = 0; i < n; ++i) {
              double Ax = 0.0;
              for (int p = p_h_rowptr[i]; p < p_h_rowptr[i + 1]; ++p)
                Ax += vk[p] * sk[p_h_colind[p]];
              double d = Ax - rk[i]; if (d < 0) d = -d; if (d > res) res = d;
            }
          }
          std::fprintf(stderr, "[batched-solve dbg] case0 n=%zu nnz=%zu max|val|=%.3e max|rhs|=%.3e finite=%d max|sol|=%.3e residual=%.3e\n",
                       n, nnz, vn, rn, (int)fin, snorm, res);
          warned2 = true;
        }
        if (!fin) { for (std::size_t j = 0; j < n; ++j) sk[j] = 0.0; }
      } catch (const std::exception &e) {
        static bool warned = false;
        if (!warned && std::getenv("GRIDPACK_BATCH_DEBUG")) {
          std::fprintf(stderr, "[batched-solve] case %d failed: %s\n", k, e.what());
          warned = true;
        }
        for (std::size_t j = 0; j < n; ++j) sk[j] = 0.0;   // -> non-converge -> fallback
      }
    }
  }

private:

  int p_n, p_nnz, p_W;

  cudssHandle_t p_handle;
  cudssConfig_t p_config;
  cudssData_t   p_data;
  cudssMatrix_t p_A, p_b, p_x;

  // one shared single-system workspace, reused for every case in the wave
  void *p_d_rowptr, *p_d_colind, *p_d_values, *p_d_rhs, *p_d_sol;
  const double *p_valsAll;   // wave value buffer [W*nnz] (not owned)
  bool p_analyzed;
  std::vector<int> p_h_rowptr, p_h_colind;   // host pattern (for residual check)

  void p_alloc(const int *rowptr, const int *colind)
  {
    const std::size_t n = p_n, nnz = p_nnz;
    p_h_rowptr.assign(rowptr, rowptr + (n + 1));
    p_h_colind.assign(colind, colind + (nnz > 0 ? nnz : 0));
    GP_CUDA_CHECK(cudaMalloc(&p_d_rowptr, (n + 1) * sizeof(int)));
    GP_CUDA_CHECK(cudaMalloc(&p_d_colind, (nnz > 0 ? nnz : 1) * sizeof(int)));
    GP_CUDA_CHECK(cudaMalloc(&p_d_values, (nnz > 0 ? nnz : 1) * sizeof(double)));
    GP_CUDA_CHECK(cudaMalloc(&p_d_rhs,    (n > 0 ? n : 1) * sizeof(double)));
    GP_CUDA_CHECK(cudaMalloc(&p_d_sol,    (n > 0 ? n : 1) * sizeof(double)));
    GP_CUDA_CHECK(cudaMemcpy(p_d_rowptr, rowptr, (n + 1) * sizeof(int),
                             cudaMemcpyHostToDevice));
    if (nnz > 0)
      GP_CUDA_CHECK(cudaMemcpy(p_d_colind, colind, nnz * sizeof(int),
                               cudaMemcpyHostToDevice));

    // single-system CSR matrix + dense RHS/solution (n x 1), reused per case
    GP_CUDSS_CHECK(cudssMatrixCreateCsr(
        &p_A, p_n, p_n, p_nnz, p_d_rowptr, /*rowEnd=*/NULL, p_d_colind, p_d_values,
        CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
        CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
    GP_CUDSS_CHECK(cudssMatrixCreateDn(&p_b, p_n, 1, p_n, p_d_rhs,
                                       CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    GP_CUDSS_CHECK(cudssMatrixCreateDn(&p_x, p_n, 1, p_n, p_d_sol,
                                       CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
  }

  CuDSSBatchedSolver(const CuDSSBatchedSolver&);
  CuDSSBatchedSolver& operator=(const CuDSSBatchedSolver&);
};

} // namespace math
} // namespace gridpack

#endif // GRIDPACK_WITH_CUDSS

#endif
