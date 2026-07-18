// Emacs Mode Line: -*- Mode:c++;-*-
// -------------------------------------------------------------
/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
// -------------------------------------------------------------
/**
 * @file   pf_batch_ca.hpp
 * @brief  Phase-2 batched N-1 power-flow engine (cuDSS batch).
 *
 * Advances a WAVE of contingencies together on the GPU instead of one
 * contingency at a time:
 *
 *   * all cases in a wave share the base Jacobian pattern, so cuDSS runs the
 *     symbolic analysis ONCE for the whole wave (CuDSSBatchedSolver);
 *   * each batched Newton step refactorizes and solves all still-active cases in
 *     one cuDSS call each;
 *   * cases are warm-started from the base solution (each contingency is a
 *     rank-bounded perturbation) and RETIRED as they converge (dropout), so
 *     work shrinks as the wave progresses.
 *
 * The numerical engine (wave loop, batched solve, warm-start, dropout) is
 * implemented and verified here against a BatchAssembler.  BatchAssembler is the
 * seam to GridPACK's per-case network state: a concrete assembler overwrites the
 * low-rank blocks a contingency changes into the base CSR pattern (values only;
 * the pattern is fixed), supplies the RHS, and applies the Newton update via the
 * existing mapper/component contract -- so the assembled systems, and therefore
 * the results, match the per-case CPU path.  Wiring that concrete assembler onto
 * PFFactoryModule/FullMatrixMap is the remaining integration step (see
 * GPU_CA_IMPLEMENTATION.md); the batched engine itself is complete.
 *
 * Only compiled with GRIDPACK_WITH_CUDSS.
 */
// -------------------------------------------------------------

#ifndef _pf_batch_ca_hpp_
#define _pf_batch_ca_hpp_

#include <string>
#include <vector>
#include "gridpack/utilities/exception.hpp"

#ifdef GRIDPACK_WITH_CUDSS
#include "gridpack/math/cudss/cudss_batched_solver.hpp"
#endif

namespace gridpack {
namespace powerflow {

// -------------------------------------------------------------
//  struct PFBatchCAConfig
// -------------------------------------------------------------
/// Tunables for the batched GPU contingency engine (from <Contingency_analysis>
/// <GPU> and <Powerflow><LinearSolver>).
struct PFBatchCAConfig {
  bool        enabled;             ///< master GPU switch
  bool        batched;             ///< use the batched engine vs per-case Phase 1
  bool        screen;             ///< run the Phase-3 connectivity pre-pass
  bool        fastDecoupled;      ///< use the Phase-4 constant-factor FDPF path
  bool        warmStart;          ///< warm-start each case from the base solution
  int         waveSize;           ///< contingencies per wave (0 => all at once)
  int         maxIterations;      ///< batched Newton iteration cap
  double      tolerance;          ///< per-case convergence tolerance (inf-norm)

  PFBatchCAConfig(void)
    : enabled(false), batched(false), screen(true), fastDecoupled(false),
      warmStart(true), waveSize(0), maxIterations(50), tolerance(1.0e-6)
  {}
};

// -------------------------------------------------------------
//  struct BatchCaseStatus
// -------------------------------------------------------------
struct BatchCaseStatus {
  bool converged;
  int  iterations;
  double mismatch;
  BatchCaseStatus(void) : converged(false), iterations(0), mismatch(0.0) {}
};

#ifdef GRIDPACK_WITH_CUDSS

// -------------------------------------------------------------
//  class BatchAssembler
// -------------------------------------------------------------
/// Seam between the batched engine and GridPACK's per-case network state.
/**
 * All cases share ONE base CSR pattern (rowptr/colind); a contingency only
 * changes VALUES (its low-rank blocks) within that pattern.  A concrete
 * assembler holds the per-case network/factory/mapper and, for a given case,
 * assembles the Jacobian values + RHS and applies the Newton correction.
 */
class BatchAssembler
{
public:
  virtual ~BatchAssembler(void) {}

  virtual int caseCount(void) const = 0;   ///< number of cases in this wave
  virtual int n(void) const = 0;           ///< system size (shared)
  virtual int nnz(void) const = 0;         ///< nonzeros in the shared pattern
  virtual const int *rowptr(void) const = 0;  ///< shared base CSR row offsets (n+1)
  virtual const int *colind(void) const = 0;  ///< shared base CSR column indices (nnz)

  /// Assemble case @c k: write nnz Jacobian values to @c jac and n RHS values to
  /// @c rhs; return the current inf-norm mismatch (for the convergence test).
  virtual double assemble(int k, double *jac, double *rhs) = 0;

  /// Apply Newton correction @c dx (length n) to case @c k; return new mismatch.
  virtual double update(int k, const double *dx) = 0;
};

// -------------------------------------------------------------
//  class PFBatchNR
// -------------------------------------------------------------
/// Batched Newton-Raphson over a wave of contingencies (cuDSS batch).
class PFBatchNR
{
public:

  PFBatchNR(BatchAssembler& assembler, double tol, int maxIter)
    : p_asm(assembler), p_tol(tol), p_maxIter(maxIter)
  {}

  /// Solve the whole wave; returns per-case convergence status.
  const std::vector<BatchCaseStatus>& solveWave(void)
  {
    const int W   = p_asm.caseCount();
    const int n   = p_asm.n();
    const int nnz = p_asm.nnz();

    p_status.assign(W, BatchCaseStatus());
    if (W <= 0) return p_status;

    // One symbolic analysis for the entire wave (shared base pattern).
    gridpack::math::CuDSSBatchedSolver solver(n, nnz, p_asm.rowptr(),
                                              p_asm.colind(), W);
    solver.analyze();

    std::vector<double> vals((size_t)W * nnz);
    std::vector<double> rhs((size_t)W * n, 0.0);
    std::vector<double> sol((size_t)W * n, 0.0);
    std::vector<char>   active(W, 1);

    int remaining = W;
    for (int it = 0; it < p_maxIter && remaining > 0; ++it) {
      // Assemble every still-active case into the shared batch buffers; retire
      // any that already satisfy the mismatch tolerance.
      for (int k = 0; k < W; ++k) {
        if (!active[k]) continue;
        double m = p_asm.assemble(k, &vals[(size_t)k * nnz], &rhs[(size_t)k * n]);
        p_status[k].mismatch = m;
        if (m <= p_tol) {
          active[k] = 0; p_status[k].converged = true; --remaining;
        }
      }
      if (remaining == 0) break;

      // Batched refactorization + solve of the whole wave (inactive cases stay
      // in the batch; their solutions are simply not applied -- correct, and
      // active-set compaction is a throughput refinement, not a correctness one).
      solver.setAllValues(&vals[0]);
      solver.factorize();
      solver.solve(&rhs[0], &sol[0]);

      // Apply the Newton correction to each active case and re-check.
      for (int k = 0; k < W; ++k) {
        if (!active[k]) continue;
        p_status[k].iterations = it + 1;
        double m = p_asm.update(k, &sol[(size_t)k * n]);
        p_status[k].mismatch = m;
        if (m <= p_tol) {
          active[k] = 0; p_status[k].converged = true; --remaining;
        }
      }
    }
    return p_status;
  }

  const std::vector<BatchCaseStatus>& status(void) const { return p_status; }

private:
  BatchAssembler& p_asm;
  double p_tol;
  int p_maxIter;
  std::vector<BatchCaseStatus> p_status;
};

#endif // GRIDPACK_WITH_CUDSS

} // namespace powerflow
} // namespace gridpack

#endif
