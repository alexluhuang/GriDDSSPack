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
 * @brief  Phase-2 core: solve MANY independent sparse linear systems that share
 *         one CSR sparsity pattern, with a SINGLE symbolic analysis, via the
 *         NVIDIA cuDSS batch API.
 *
 * This is the reusable heart of the batched contingency engine.  Every N-1 case
 * is a rank-bounded perturbation of the base network, so all cases in a wave
 * share the base Jacobian pattern; cuDSS analyzes that pattern once
 * (CUDSS_PHASE_ANALYSIS) and then, per batched Newton step, refactorizes and
 * solves all W systems together with one call each
 * (CUDSS_PHASE_FACTORIZATION / CUDSS_PHASE_SOLVE over a batch object).
 *
 * Layout (matches the cuDSS batch examples):
 *   * host int arrays nrows/ncols/nnz/ld of length W (here every entry is n or
 *     nnz -- a uniform batch, but the batch API path is the non-uniform one so
 *     islanding-resized cases can be supported later);
 *   * one shared device pattern (row offsets + column indices), reused by every
 *     batch entry via device pointer-of-pointers;
 *   * one contiguous device values buffer (W*nnz), one RHS buffer (W*n), one
 *     solution buffer (W*n), sliced per system.
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
#include <cuda_runtime.h>
#include <cudss.h>
#include "cudss/cudss_exception.hpp"

namespace gridpack {
namespace math {

// -------------------------------------------------------------
//  class CuDSSBatchedSolver
// -------------------------------------------------------------
class CuDSSBatchedSolver
{
public:

  /// Build a batched solver for @c batchCount systems sharing one CSR pattern.
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
      p_d_valPtrs(NULL), p_d_rhsPtrs(NULL), p_d_solPtrs(NULL),
      p_d_offPtrs(NULL), p_d_colPtrs(NULL),
      p_analyzed(false)
  {
    p_nrows.assign(p_W, p_n);
    p_ncols.assign(p_W, p_n);
    p_nnzArr.assign(p_W, p_nnz);
    p_nrhs.assign(p_W, 1);
    p_ld.assign(p_W, p_n);

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
    cudaFree(p_d_valPtrs); cudaFree(p_d_rhsPtrs); cudaFree(p_d_solPtrs);
    cudaFree(p_d_offPtrs); cudaFree(p_d_colPtrs);
    if (p_data)   cudssDataDestroy(p_handle, p_data);
    if (p_config) cudssConfigDestroy(p_config);
    if (p_handle) cudssDestroy(p_handle);
  }

  int rows(void) const       { return p_n; }
  int nnz(void) const        { return p_nnz; }
  int batchCount(void) const { return p_W; }

  /// Upload all systems' values from a contiguous host buffer [W*nnz]
  /// (system k occupies [k*nnz, (k+1)*nnz)).
  void setAllValues(const double *values)
  {
    GP_CUDA_CHECK(cudaMemcpy(p_d_values, values,
                             static_cast<std::size_t>(p_W) * p_nnz * sizeof(double),
                             cudaMemcpyHostToDevice));
  }

  /// Upload one system's values (host, length nnz).
  void setValues(int k, const double *values)
  {
    GP_CUDA_CHECK(cudaMemcpy(static_cast<double *>(p_d_values) + (std::size_t)k * p_nnz,
                             values, (std::size_t)p_nnz * sizeof(double),
                             cudaMemcpyHostToDevice));
  }

  /// Symbolic analysis of the shared pattern -- run exactly once.
  void analyze(void)
  {
    GP_CUDSS_CHECK(cudssExecute(p_handle, CUDSS_PHASE_ANALYSIS,
                                p_config, p_data, p_A, p_x, p_b));
    p_analyzed = true;
  }

  /// Batched numeric (re)factorization of all systems with the current values.
  void factorize(void)
  {
    GP_CUDSS_CHECK(cudssExecute(p_handle, CUDSS_PHASE_FACTORIZATION,
                                p_config, p_data, p_A, p_x, p_b));
  }

  /// Batched solve: rhs [W*n] in, sol [W*n] out (system k at offset k*n).
  void solve(const double *rhs, double *sol)
  {
    GP_CUDA_CHECK(cudaMemcpy(p_d_rhs, rhs,
                             (std::size_t)p_W * p_n * sizeof(double),
                             cudaMemcpyHostToDevice));
    GP_CUDSS_CHECK(cudssExecute(p_handle, CUDSS_PHASE_SOLVE,
                                p_config, p_data, p_A, p_x, p_b));
    GP_CUDA_CHECK(cudaDeviceSynchronize());
    GP_CUDA_CHECK(cudaMemcpy(sol, p_d_sol,
                             (std::size_t)p_W * p_n * sizeof(double),
                             cudaMemcpyDeviceToHost));
  }

private:

  int p_n, p_nnz, p_W;

  cudssHandle_t p_handle;
  cudssConfig_t p_config;
  cudssData_t   p_data;
  cudssMatrix_t p_A, p_b, p_x;

  // shared pattern + per-system value/rhs/sol (contiguous)
  void *p_d_rowptr, *p_d_colind;
  void *p_d_values, *p_d_rhs, *p_d_sol;
  // device arrays of device pointers, one entry per system
  void **p_d_valPtrs, **p_d_rhsPtrs, **p_d_solPtrs, **p_d_offPtrs, **p_d_colPtrs;

  std::vector<int> p_nrows, p_ncols, p_nnzArr, p_nrhs, p_ld;
  bool p_analyzed;

  void p_alloc(const int *rowptr, const int *colind)
  {
    const std::size_t W = p_W, n = p_n, nnz = p_nnz;

    // shared pattern
    GP_CUDA_CHECK(cudaMalloc(&p_d_rowptr, (n + 1) * sizeof(int)));
    GP_CUDA_CHECK(cudaMalloc(&p_d_colind, (nnz > 0 ? nnz : 1) * sizeof(int)));
    GP_CUDA_CHECK(cudaMemcpy(p_d_rowptr, rowptr, (n + 1) * sizeof(int),
                             cudaMemcpyHostToDevice));
    if (nnz > 0)
      GP_CUDA_CHECK(cudaMemcpy(p_d_colind, colind, nnz * sizeof(int),
                               cudaMemcpyHostToDevice));

    // per-system contiguous storage
    GP_CUDA_CHECK(cudaMalloc(&p_d_values, W * (nnz > 0 ? nnz : 1) * sizeof(double)));
    GP_CUDA_CHECK(cudaMalloc(&p_d_rhs,    W * (n > 0 ? n : 1) * sizeof(double)));
    GP_CUDA_CHECK(cudaMalloc(&p_d_sol,    W * (n > 0 ? n : 1) * sizeof(double)));

    // build host arrays of device pointers, then copy to device void** arrays
    std::vector<void *> off(W), col(W), val(W), b(W), x(W);
    for (std::size_t k = 0; k < W; ++k) {
      off[k] = p_d_rowptr;                                   // shared pattern
      col[k] = p_d_colind;                                   // shared pattern
      val[k] = static_cast<char *>(p_d_values) + k * nnz * sizeof(double);
      b[k]   = static_cast<char *>(p_d_rhs)    + k * n   * sizeof(double);
      x[k]   = static_cast<char *>(p_d_sol)    + k * n   * sizeof(double);
    }
    p_d_offPtrs = p_allocPtrArray(off);
    p_d_colPtrs = p_allocPtrArray(col);
    p_d_valPtrs = p_allocPtrArray(val);
    p_d_rhsPtrs = p_allocPtrArray(b);
    p_d_solPtrs = p_allocPtrArray(x);

    // batch matrix objects (real double, int32 indices, general, 0-based)
    GP_CUDSS_CHECK(cudssMatrixCreateBatchCsr(
        &p_A, p_W, p_nrows.data(), p_ncols.data(), p_nnzArr.data(),
        p_d_offPtrs, /*rowEnd=*/NULL, p_d_colPtrs, p_d_valPtrs,
        CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
        CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
    GP_CUDSS_CHECK(cudssMatrixCreateBatchDn(
        &p_b, p_W, p_nrows.data(), p_nrhs.data(), p_ld.data(),
        p_d_rhsPtrs, CUDSS_R_32I, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    GP_CUDSS_CHECK(cudssMatrixCreateBatchDn(
        &p_x, p_W, p_ncols.data(), p_nrhs.data(), p_ld.data(),
        p_d_solPtrs, CUDSS_R_32I, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
  }

  void **p_allocPtrArray(const std::vector<void *>& hostPtrs)
  {
    void **d = NULL;
    GP_CUDA_CHECK(cudaMalloc(&d, hostPtrs.size() * sizeof(void *)));
    GP_CUDA_CHECK(cudaMemcpy(d, hostPtrs.data(), hostPtrs.size() * sizeof(void *),
                             cudaMemcpyHostToDevice));
    return d;
  }

  CuDSSBatchedSolver(const CuDSSBatchedSolver&);
  CuDSSBatchedSolver& operator=(const CuDSSBatchedSolver&);
};

} // namespace math
} // namespace gridpack

#endif // GRIDPACK_WITH_CUDSS

#endif
