// -------------------------------------------------------------
/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
// -------------------------------------------------------------
/**
 * @file   cudss_batched_test.cpp
 * @brief  Standalone GB10 check of the Phase-2 batched cuDSS solver.
 *
 * Solves W independent systems that SHARE one CSR pattern but have different
 * coefficient values AND right-hand sides, with a single symbolic analysis, and
 * verifies each solution against a known answer.  System k uses A_k = (k+1)*A
 * and a chosen solution s_k, with b_k = A_k s_k computed on the host, so the
 * batched solve must return s_k for every k.
 *
 * Only built with GRIDPACK_WITH_CUDSS.  Exit code 0 on success.
 */
// -------------------------------------------------------------

#include <cstdio>
#include <cmath>
#include <vector>

#ifdef GRIDPACK_WITH_CUDSS
#include "gridpack/math/cudss/cudss_batched_solver.hpp"
#endif

int main(int /*argc*/, char ** /*argv*/)
{
#ifndef GRIDPACK_WITH_CUDSS
  printf("cudss_batched_test: built without GRIDPACK_WITH_CUDSS -- skipping\n");
  return 0;
#else
  // A general nonsymmetric 5x5 pattern (from the cuDSS batch example, system 0).
  const int n = 5, nnz = 8;
  const int rowptr[6] = {0, 2, 4, 6, 7, 8};
  const int colind[8] = {0, 2, 1, 2, 2, 4, 3, 4};
  const double baseVals[8] = {4.0, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0};

  const int W = 4;

  // Host CSR mat-vec: y = A x for one system's value array.
  auto matvec = [&](const double *vals, const double *x, double *y) {
    for (int i = 0; i < n; ++i) {
      double s = 0.0;
      for (int p = rowptr[i]; p < rowptr[i + 1]; ++p) s += vals[p] * x[colind[p]];
      y[i] = s;
    }
  };

  std::vector<double> allVals((size_t)W * nnz), allRhs((size_t)W * n),
                      allSol((size_t)W * n), expect((size_t)W * n);

  for (int k = 0; k < W; ++k) {
    double *vals = &allVals[(size_t)k * nnz];
    double *rhs  = &allRhs[(size_t)k * n];
    double *exp  = &expect[(size_t)k * n];
    for (int p = 0; p < nnz; ++p) vals[p] = (k + 1) * baseVals[p];   // A_k = (k+1)A
    for (int i = 0; i < n; ++i)   exp[i]  = 1.0 + i + 0.5 * k;        // chosen s_k
    matvec(vals, exp, rhs);                                          // b_k = A_k s_k
  }

  try {
    gridpack::math::CuDSSBatchedSolver solver(n, nnz, rowptr, colind, W);
    solver.setAllValues(&allVals[0]);
    solver.analyze();     // ONE symbolic analysis for the whole batch
    solver.factorize();   // batched numeric factorization
    solver.solve(&allRhs[0], &allSol[0]);   // batched solve
  } catch (const std::exception &e) {
    printf("cudss_batched_test: FAILED with exception: %s\n", e.what());
    return 1;
  }

  double maxErr = 0.0;
  for (int k = 0; k < W; ++k) {
    for (int i = 0; i < n; ++i) {
      double d = std::fabs(allSol[(size_t)k * n + i] - expect[(size_t)k * n + i]);
      if (d > maxErr) maxErr = d;
    }
  }
  printf("cudss_batched_test: W=%d systems (shared pattern, distinct A_k & b_k), "
         "one analysis; max|x-x_exact| = %.3e\n", W, maxErr);
  if (maxErr > 1.0e-9) {
    printf("cudss_batched_test: FAILED (error too large)\n");
    return 1;
  }
  printf("cudss_batched_test: PASSED\n");
  return 0;
#endif
}
