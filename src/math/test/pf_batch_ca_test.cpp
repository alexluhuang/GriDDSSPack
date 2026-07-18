// -------------------------------------------------------------
/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
// -------------------------------------------------------------
/**
 * @file   pf_batch_ca_test.cpp
 * @brief  GB10 check of the Phase-2 batched Newton engine (PFBatchNR).
 *
 * Drives the engine with a synthetic wave of W independent 2x2 NONLINEAR systems
 * that share one dense 2x2 pattern but have different targets (hence different
 * Jacobians, RHS, and iteration counts).  Each case k has the known root
 * (u_k, v_k) of
 *      x0^2 + x1 = a_k ,   x0 + x1^2 = b_k
 * with a_k = u_k^2 + v_k, b_k = u_k + v_k^2.  The batched engine must converge
 * every case to its root using ONE shared cuDSS symbolic analysis, a batched
 * refactor+solve per Newton step, and per-case dropout.  This verifies the
 * engine's control flow and its use of the cuDSS batch path end to end on the
 * GPU, independently of the GridPACK network assembler.
 *
 * Only built with GRIDPACK_WITH_CUDSS.  Exit 0 on success.
 */
// -------------------------------------------------------------

#include <cstdio>
#include <cmath>
#include <vector>

#ifdef GRIDPACK_WITH_CUDSS
#include "gridpack/applications/modules/powerflow/pf_batch_ca.hpp"

namespace {

class QuadraticBatch : public gridpack::powerflow::BatchAssembler
{
public:
  explicit QuadraticBatch(int W) : p_W(W)
  {
    p_rowptr[0] = 0; p_rowptr[1] = 2; p_rowptr[2] = 4;
    p_colind[0] = 0; p_colind[1] = 1; p_colind[2] = 0; p_colind[3] = 1;
    p_x.assign((size_t)W * 2, 1.0);   // warm start x=(1,1) for every case
    p_a.resize(W); p_b.resize(W); p_u.resize(W); p_v.resize(W);
    for (int k = 0; k < W; ++k) {
      p_u[k] = 1.0 + 0.30 * k;         // distinct roots per case
      p_v[k] = 2.0 - 0.15 * k;
      p_a[k] = p_u[k] * p_u[k] + p_v[k];
      p_b[k] = p_u[k] + p_v[k] * p_v[k];
    }
  }

  int caseCount(void) const { return p_W; }
  int n(void) const         { return 2; }
  int nnz(void) const       { return 4; }
  const int *rowptr(void) const { return p_rowptr; }
  const int *colind(void) const { return p_colind; }

  double assemble(int k, double *jac, double *rhs)
  {
    double x0 = p_x[2*k], x1 = p_x[2*k+1];
    double f0 = x0*x0 + x1 - p_a[k];
    double f1 = x0 + x1*x1 - p_b[k];
    // Newton: J dx = -F
    jac[0] = 2.0*x0; jac[1] = 1.0;      // row 0
    jac[2] = 1.0;    jac[3] = 2.0*x1;   // row 1
    rhs[0] = -f0;    rhs[1] = -f1;
    return std::max(std::fabs(f0), std::fabs(f1));
  }

  double update(int k, const double *dx)
  {
    p_x[2*k]   += dx[0];
    p_x[2*k+1] += dx[1];
    double x0 = p_x[2*k], x1 = p_x[2*k+1];
    double f0 = x0*x0 + x1 - p_a[k];
    double f1 = x0 + x1*x1 - p_b[k];
    return std::max(std::fabs(f0), std::fabs(f1));
  }

  // Max over cases of the final residual ||F_k(x_k)||_inf.  Each case has its
  // own (a_k,b_k), so a tiny residual for every case proves the batch solved W
  // DISTINCT nonlinear systems (not the same one W times).  (The system has
  // multiple real roots, so we check the residual, not a specific root.)
  double residualErr(void) const
  {
    double e = 0.0;
    for (int k = 0; k < p_W; ++k) {
      double x0 = p_x[2*k], x1 = p_x[2*k+1];
      double f0 = x0*x0 + x1 - p_a[k];
      double f1 = x0 + x1*x1 - p_b[k];
      e = std::max(e, std::max(std::fabs(f0), std::fabs(f1)));
    }
    return e;
  }

private:
  int p_W;
  int p_rowptr[3], p_colind[4];
  std::vector<double> p_x, p_a, p_b, p_u, p_v;
};

} // namespace
#endif

int main(int /*argc*/, char ** /*argv*/)
{
#ifndef GRIDPACK_WITH_CUDSS
  printf("pf_batch_ca_test: built without GRIDPACK_WITH_CUDSS -- skipping\n");
  return 0;
#else
  const int W = 8;
  QuadraticBatch batch(W);
  try {
    gridpack::powerflow::PFBatchNR engine(batch, 1.0e-10, 50);
    const std::vector<gridpack::powerflow::BatchCaseStatus>& st = engine.solveWave();
    int nconv = 0, maxit = 0;
    for (int k = 0; k < W; ++k) { if (st[k].converged) ++nconv; if (st[k].iterations > maxit) maxit = st[k].iterations; }
    double err = batch.residualErr();
    printf("pf_batch_ca_test: W=%d distinct cases, %d converged, max iters=%d, "
           "max final residual=%.3e\n", W, nconv, maxit, err);
    if (nconv != W || err > 1.0e-8) {
      printf("pf_batch_ca_test: FAILED\n");
      return 1;
    }
  } catch (const std::exception &e) {
    printf("pf_batch_ca_test: FAILED with exception: %s\n", e.what());
    return 1;
  }
  printf("pf_batch_ca_test: PASSED\n");
  return 0;
#endif
}
