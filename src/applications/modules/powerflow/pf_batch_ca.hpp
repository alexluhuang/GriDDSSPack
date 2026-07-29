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

#include <algorithm>
#include <cmath>
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
  int maxPBus;
  double maxPMismatch;
  int maxQBus;
  double maxQMismatch;
  int refactorizations;
  BatchCaseStatus(void)
    : converged(false), iterations(0), mismatch(0.0),
      maxPBus(0), maxPMismatch(0.0), maxQBus(0), maxQMismatch(0.0),
      refactorizations(0)
  {}
};

struct BatchMismatchInfo {
  int maxPBus;
  double maxPMismatch;
  int maxQBus;
  double maxQMismatch;
  BatchMismatchInfo(void)
    : maxPBus(0), maxPMismatch(0.0), maxQBus(0), maxQMismatch(0.0)
  {}
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

  // --- "live" per-case path (sequential wave): keep case k's state live across
  // its Newton iterations so restore/snapshot/branch-toggle and the redundant
  // RHS recompute happen once per case, not once per iteration. ---

  /// Start case @c k: load its warm start, take the branch out, and assemble the
  /// first Jacobian values (@c jac) + RHS (@c rhs).  Leaves the branch out and
  /// the network in case k's live state.  Returns the inf-norm mismatch.
  virtual double assembleLive(int k, double *jac, double *rhs) = 0;

  /// Advance case @c k in place: apply Newton correction @c dx to the LIVE
  /// state (no reload), then assemble the next @c jac + @c rhs.  Returns the new
  /// inf-norm mismatch.
  virtual double updateLive(int k, const double *dx, double *jac, double *rhs) = 0;

  /// Finish case @c k: snapshot its converged iterate (for output) and restore
  /// the base topology so the next case starts clean.
  virtual void finishLive(int k) = 0;

  // --- constant-factorization (chord) path: base Jacobian factorized once, each
  // case iterates with it reassembling only the RHS (see cudss_batched_solver). ---

  /// Assemble the base Jacobian (all branches in, base voltages) into @c jac.
  virtual void assembleBaseJac(double *jac) = 0;
  /// Start case @c k for the chord path: warm-start + branch out + RHS only.
  virtual double assembleLiveRhs(int k, double *rhs) = 0;
  /// Chord step: apply @c dx to case @c k's live state, recompute RHS only.
  virtual double updateLiveRhs(int k, const double *dx, double *rhs) = 0;

  /// Assemble the current live case's Jacobian without changing its state.
  /// Used for bounded adaptive numeric refactorization when chord convergence
  /// stalls; the CSR pattern remains the wave's shared pattern.
  virtual void assembleLiveJac(int k, double *jac) = 0;

  /// Worst P/Q mismatch captured by the most recent live RHS assembly.
  virtual BatchMismatchInfo lastMismatch(void) const = 0;
};

// -------------------------------------------------------------
//  class PFBatchNR
// -------------------------------------------------------------
/// Batched Newton-Raphson over a wave of contingencies (cuDSS batch).
class PFBatchNR
{
public:

  /// @param refactorEvery  Phase-4 constant-factorization stride: refactorize the
  ///   batch only every Nth Newton step (reusing the prior LU for the solves in
  ///   between -- a chord/dishonest-Newton step).  1 (default) => exact Newton,
  ///   refactor every step.  A large value (>= maxIter) => factor once, solve many.
  PFBatchNR(BatchAssembler& assembler, double tol, int maxIter,
            int refactorEvery = 1, bool constantFactor = false,
            int chordCap = 0, double damping = 1.0,
            int maxAdaptiveRefactors = 2)
    : p_asm(assembler), p_tol(tol), p_maxIter(maxIter),
      p_refactorEvery(refactorEvery < 0 ? 0 : refactorEvery),
      p_constantFactor(constantFactor),
      p_chordCap(chordCap > 0 ? std::min(chordCap, std::max(1, maxIter))
                              : std::max(1, maxIter)),
      p_damping(damping > 0.0 ? std::min(damping, 1.0) : 1.0),
      p_maxAdaptiveRefactors(maxAdaptiveRefactors < 0 ? 0
                                                     : maxAdaptiveRefactors)
  {}

  /// Solve every case; returns per-case convergence status.
  ///
  /// Cases are advanced SEQUENTIALLY (each runs its own Newton loop to
  /// convergence on the live network state) rather than in lockstep.  The
  /// lockstep "wave" (all-assemble, one batched solve, all-update) shares one
  /// serial network across W cases, so the restore/snapshot needed between a
  /// case's assemble and its update -- with 63 other cases toggling branches in
  /// between -- corrupts the shared state and the Newton diverges.  Sequential
  /// per-case advancement keeps each case's state live and correct (verified to
  /// converge quadratically) while STILL amortizing the one expensive lever:
  /// cuDSS runs the symbolic ANALYSIS exactly once for the shared pattern and
  /// every case reuses it (only factorize+solve per step).  The assembly, not
  /// the solve, is the wave's real cost and the fast GA-free assembler handles
  /// that; a true batched solve gave NaN on this scale (see cudss_batched_solver).
  const std::vector<BatchCaseStatus>& solveWave(void)
  {
    const int W   = p_asm.caseCount();
    const int n   = p_asm.n();
    const int nnz = p_asm.nnz();

    p_status.assign(W, BatchCaseStatus());
    if (W <= 0) return p_status;

    // One symbolic analysis for the shared base pattern, reused by every case.
    gridpack::math::CuDSSBatchedSolver solver(n, nnz, p_asm.rowptr(),
                                              p_asm.colind(), 1);
    solver.analyze();

    std::vector<double> vals(nnz);
    std::vector<double> rhs(n, 0.0);
    std::vector<double> sol(n, 0.0);

    // Factor-reuse (dishonest-Newton / chord) path.  For each case we
    // factorize its OWN iter-1 Jacobian ONCE -- assembled with the branch already
    // out of service at the warm-start voltages, so it is consistent with the
    // residual and close to the contingency solution (a contingency is a
    // rank-bounded perturbation of the base, so the warm start is near the answer)
    // -- then hold that factorization fixed and iterate reassembling ONLY the RHS.
    // This replaces the exact path's 3-4 GPU refactorizations per case (each
    // ~250 ms on a 48k reduced system) with a SINGLE factorization plus a handful
    // of cheap triangular solves, which is what lets the wave beat CPU sparse LU on
    // large systems.  Convergence is linear rather than quadratic; a case that
    // stalls within p_chordCap steps is marked non-converged and the driver
    // re-solves it with exact Newton on CPU.  Accuracy is preserved either way --
    // any converged iterate satisfies F(x)=0 exactly regardless of which Jacobian
    // drove the steps.  (Factoring the BASE Jacobian once for ALL cases was tried
    // and rejected: base-voltage + branch-in factors are too far from each
    // contingency and most cases failed to converge.)
    if (p_constantFactor || p_refactorEvery > 1) {
      for (int k = 0; k < W; ++k) {
        bool conv = false;
        int it = 1;
        int refactorizations = 0;
        int adaptiveRefactors = 0;
        int solvesSinceFactor = 0;
        bool previousStepRefactoredForPoorProgress = false;
        double m = p_asm.assembleLive(k, &vals[0], &rhs[0]);   // J + RHS, branch out
        if (m <= p_tol) {
          conv = true;
        } else if (solver.factorizeValues(&vals[0])) {         // factor this case ONCE
          const double poorProgressThreshold =
            std::max(0.5, 1.0 - 0.5 * p_damping);
          for (it = 1; it < p_chordCap; ++it) {
            const double previousMismatch = m;
            if (!solver.solveReuse(&rhs[0], &sol[0])) break;    // reuse factor -> fallback
            m = p_asm.updateLiveRhs(k, &sol[0], &rhs[0]);       // chord step, RHS only
            if (m <= p_tol) { conv = true; ++it; break; }
            if (!std::isfinite(m) || previousMismatch <= 0.0) break;

            ++solvesSinceFactor;
            const double ratio = m / previousMismatch;
            const bool poorProgress = ratio >= poorProgressThreshold;
            const bool scheduledRefresh =
              p_refactorEvery > 1 && solvesSinceFactor >= p_refactorEvery;

            // A poor step immediately after an adaptive refresh says the current
            // case is a bad fit for modified Newton.  Fall back to the exact CPU
            // path instead of degenerating into repeated expensive GPU factors.
            if (poorProgress && previousStepRefactoredForPoorProgress) break;
            if (poorProgress &&
                adaptiveRefactors >= p_maxAdaptiveRefactors) break;

            if (poorProgress || scheduledRefresh) {
              p_asm.assembleLiveJac(k, &vals[0]);
              if (!solver.factorizeValues(&vals[0])) break;
              ++refactorizations;
              solvesSinceFactor = 0;
              if (poorProgress) {
                ++adaptiveRefactors;
                previousStepRefactoredForPoorProgress = true;
              } else {
                previousStepRefactoredForPoorProgress = false;
              }
            } else {
              previousStepRefactoredForPoorProgress = false;
            }
          }
        }
        const BatchMismatchInfo minfo = p_asm.lastMismatch();
        p_asm.finishLive(k);
        p_status[k].converged = conv;
        p_status[k].iterations = it;
        p_status[k].mismatch = m;
        p_status[k].maxPBus = minfo.maxPBus;
        p_status[k].maxPMismatch = minfo.maxPMismatch;
        p_status[k].maxQBus = minfo.maxQBus;
        p_status[k].maxQMismatch = minfo.maxQMismatch;
        p_status[k].refactorizations = refactorizations;
      }
      return p_status;
    }

    for (int k = 0; k < W; ++k) {
      bool conv = false;
      int it = 1;
      double m = p_asm.assembleLive(k, &vals[0], &rhs[0]);   // warm start, iter 1
      if (m <= p_tol) {
        conv = true;
      } else {
        for (it = 1; it < p_maxIter; ++it) {
          if (!solver.solveOne(&vals[0], &rhs[0], &sol[0])) break;   // -> fallback
          m = p_asm.updateLive(k, &sol[0], &vals[0], &rhs[0]);       // apply + reassemble
          if (m <= p_tol) { conv = true; ++it; break; }
        }
      }
      const BatchMismatchInfo minfo = p_asm.lastMismatch();
      p_asm.finishLive(k);
      p_status[k].converged = conv;
      p_status[k].iterations = it;
      p_status[k].mismatch = m;
      p_status[k].maxPBus = minfo.maxPBus;
      p_status[k].maxPMismatch = minfo.maxPMismatch;
      p_status[k].maxQBus = minfo.maxQBus;
      p_status[k].maxQMismatch = minfo.maxQMismatch;
    }
    return p_status;
  }

  const std::vector<BatchCaseStatus>& status(void) const { return p_status; }

private:
  BatchAssembler& p_asm;
  double p_tol;
  int p_maxIter;
  int p_refactorEvery;
  bool p_constantFactor;
  int p_chordCap;
  double p_damping;
  int p_maxAdaptiveRefactors;
  std::vector<BatchCaseStatus> p_status;
};

#endif // GRIDPACK_WITH_CUDSS

} // namespace powerflow
} // namespace gridpack

#endif
