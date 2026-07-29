// Emacs Mode Line: -*- Mode:c++;-*-
// -------------------------------------------------------------
/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
// -------------------------------------------------------------
/**
 * @file   pf_batch_ca_assembler.hpp
 * @brief  Concrete GridPACK BatchAssembler for the Phase-2 batched GPU engine.
 *
 * Bridges PFBatchNR (pf_batch_ca.hpp) to GridPACK's per-case network state.  A
 * single serial network (grp_size==1 => the whole network lives on one rank)
 * is time-shared across a WAVE of contingencies: each case carries its own
 * (V,angle) iterate, and the shared base Jacobian ALLOCATION is reused so cuDSS
 * runs the symbolic analysis ONCE for the whole wave.
 *
 * The shared base pattern is only valid when a case leaves every bus's
 * PV/PQ/slack/isolated status unchanged (see prepare()).  With LARGE_MATRIX off
 * a PV bus contributes 1 row and a PQ bus 2, so a generator outage, a qlim
 * PV->PQ switch, a slack transfer, or islanding would change the dimension and
 * the pattern.  prepare() therefore classifies each case: pure branch outages
 * that preserve the per-bus structure signature go in the cuDSS batch; every
 * other case (gen, islanded, no-slack, slack-transfer, structure-changed) is
 * handed back to the driver's exact per-contingency CPU path.  This keeps the
 * batched results identical (within tolerance) to the per-case path.
 *
 * Only compiled with GRIDPACK_WITH_CUDSS.
 */
// -------------------------------------------------------------

#ifndef _pf_batch_ca_assembler_hpp_
#define _pf_batch_ca_assembler_hpp_

#ifdef GRIDPACK_WITH_CUDSS

#include <vector>
#include <map>
#include <string>
#include <utility>
#include <cstdlib>
#include <cstdio>
#include <climits>
#include <cmath>
#include <chrono>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>

// Env-gated micro-profiler for the batched assembler hot path.  Enabled by
// setting GRIDPACK_BATCH_PROFILE=1; prints a per-sub-op cumulative breakdown at
// destruction.  Zero cost when the accumulators are not read.
#define BT_T0() std::chrono::high_resolution_clock::now()
#define BT_ADD(acc,t0) do { (acc) += std::chrono::duration<double>( \
    std::chrono::high_resolution_clock::now() - (t0)).count(); } while(0)

#include "gridpack/applications/modules/powerflow/pf_app_module.hpp"
#include "gridpack/applications/modules/powerflow/pf_batch_ca.hpp"
#include "gridpack/applications/modules/powerflow/pf_screen.hpp"
#include "gridpack/applications/components/pf_matrix/pf_components.hpp"
#include "gridpack/mapper/full_map.hpp"
#include "gridpack/mapper/bus_vector_map.hpp"
#include "gridpack/math/matrix.hpp"
#include "gridpack/math/vector.hpp"
#include "gridpack/math/cudss/cudss_csr_extractor.hpp"

namespace gridpack {
namespace powerflow {

// -------------------------------------------------------------
//  class GridpackBatchAssembler
// -------------------------------------------------------------
class GridpackBatchAssembler : public BatchAssembler
{
public:

  /// @param app        the powerflow application (base case already solved)
  /// @param events     the full contingency list (indexed by task id)
  /// @param taskIds    this rank's assigned task ids (subset of events)
  /// @param warmStart  true => each case starts from the base solution;
  ///                    false => from the parser-initial voltages (cold, exactly
  ///                    like the per-contingency CPU path).
  /// @param damping    Newton step damping factor (matches PFAppModule::solve's
  ///                    p_dampingFactor; 1.0 = no damping).
  GridpackBatchAssembler(PFAppModule& app,
                         std::vector<Contingency>& events,
                         const std::vector<int>& taskIds,
                         bool warmStart, double damping = 1.0,
                         bool connectivityScreen = true)
    : p_app(app),
      p_net(app.getNetwork()),
      p_factory(app.getFactory()),
      p_events(events),
      p_taskIds(taskIds),
      p_n(0), p_nnz(0), p_nbus(0), p_warmStart(warmStart),
      p_damping(damping),
      p_tRestore(0), p_tYbus(0), p_tMapJ(0), p_tCsr(0), p_tMapV(0),
      p_tRhs(0), p_tUpd(0), p_tSnap(0), p_nAsm(0), p_nUpd(0),
      p_useFast(true), p_validateFast(false),
      p_screenEnabled(connectivityScreen),
      p_screenBridgeCount(0)
  {
    // Opt out of the fast assembler (fall back to the GA mapper) for A/B
    // validation via GRIDPACK_BATCH_NOFAST=1.
    if (std::getenv("GRIDPACK_BATCH_NOFAST")) p_useFast = false;
    p_validateFast = (std::getenv("GRIDPACK_BATCH_VALIDATE") != NULL);
    p_dbg = (std::getenv("GRIDPACK_BATCH_DEBUG") != NULL);
    p_nbus = p_net->numBuses();

    // Snapshot the per-case starting voltages.  For a cold start match the CPU
    // path exactly (raw parser values); otherwise reuse the base solution.
    std::vector<double> startV(p_nbus, 0.0), startA(p_nbus, 0.0);
    if (!warmStart) p_app.resetVoltages();
    p_isActiveBus.assign(p_nbus, 0);
    for (int i = 0; i < p_nbus; i++) {
      p_isActiveBus[i] = p_net->getActiveBus(i) ? 1 : 0;
      PFBus* bus = dynamic_cast<PFBus*>(p_net->getBus(i).get());
      if (!bus) continue;
      double v = 0.0, a = 0.0;
      bus->getVoltageState(v, a);
      startV[i] = v; startA[i] = a;
    }
    p_startV.swap(startV);
    p_startA.swap(startA);

    // Base per-bus structure signature (0 = ref/isolated/none, 1 = PV, 2 = PQ)
    // and the shared base Jacobian pattern.  Built from the base topology so the
    // allocation contains every branch slot a contingency might zero out.
    // NOTE: the mappers size themselves from the factory mode AT CONSTRUCTION
    // (exactly like PFAppModule::solve), so vMap must be built after setMode(RHS)
    // and jMap after setMode(Jacobian) -- otherwise the matrix has 0 rows.
    p_factory->setYBus();
    p_factory->setMode(YBus);
    p_factory->setSBus();
    p_factory->setMode(RHS);
    p_vMap.reset(new gridpack::mapper::BusVectorMap<PFNetwork>(p_net));
    p_PQ = p_vMap->mapToRealVector();
    p_factory->setMode(Jacobian);
    p_jMap.reset(new gridpack::mapper::FullMatrixMap<PFNetwork>(p_net));
    p_J = p_jMap->mapToRealMatrix();
    p_X.reset(p_PQ->clone());
    p_baseSig = p_structureSignature();
    {
      gridpack::math::PetscSeqCSRView<gridpack::RealType, int> view(*p_J);
      const PetscInt rows = view.rows();
      const PetscInt nnz = view.nnz();
      if (rows < 0 || rows >= static_cast<PetscInt>(INT_MAX) ||
          nnz < 0 || nnz > static_cast<PetscInt>(INT_MAX)) {
        throw gridpack::Exception(
            "pf_batch_ca: reduced Jacobian does not fit 32-bit cuDSS indices");
      }
      p_n   = static_cast<int>(rows);
      p_nnz = static_cast<int>(nnz);
      const PetscInt* ia = view.rowptr();
      const PetscInt* ja = view.colind();
      p_rowptr.resize(p_n + 1);
      for (int i = 0; i <= p_n; i++) {
        if (ia[i] < 0 || ia[i] > nnz || (i > 0 && ia[i] < ia[i - 1])) {
          throw gridpack::Exception("pf_batch_ca: invalid CSR row offsets");
        }
        p_rowptr[i] = static_cast<int>(ia[i]);
      }
      if (p_rowptr[p_n] != p_nnz) {
        throw gridpack::Exception("pf_batch_ca: CSR row offsets do not end at nnz");
      }
      p_colind.resize(p_nnz);
      for (int i = 0; i < p_nnz; i++) {
        if (ja[i] < 0 || ja[i] >= rows) {
          throw gridpack::Exception("pf_batch_ca: invalid CSR column index");
        }
        p_colind[i] = static_cast<int>(ja[i]);
      }
    }
    p_corr.assign(p_n, 0.0);
    // Precompute the CSR scatter map (component block -> CSR slot) from the base
    // topology, which is exactly the fixed pattern extracted above.
    if (p_useFast) p_buildScatterMap();
    if (p_screenEnabled) p_buildConnectivityScreen();
  }

  ~GridpackBatchAssembler(void)
  {
    if (std::getenv("GRIDPACK_BATCH_PROFILE")) {
      std::fprintf(stderr,
        "[BATCH_PROFILE] assemble=%ld update=%ld | restore=%.2fs ybus/setMode=%.2fs "
        "mapJ=%.2fs csrCopy=%.2fs mapV=%.2fs rhs=%.2fs update-map=%.2fs snap=%.2fs\n",
        p_nAsm, p_nUpd, p_tRestore, p_tYbus, p_tMapJ, p_tCsr, p_tMapV,
        p_tRhs, p_tUpd, p_tSnap);
    }
  }

  /// Reuse the immutable base pattern, mapper, scatter map and connectivity
  /// screen for another rank-local wave.  CPU fallback cases leave cached
  /// Ybus/Sbus values at their contingency state, so repair those caches once
  /// here before any local GPU-path patches are applied.
  void beginWave(const std::vector<int>& taskIds)
  {
    p_restoreBaseState();
    if (p_structureSignature() != p_baseSig) {
      throw gridpack::Exception(
          "pf_batch_ca: bus structure did not return to the base signature");
    }
    p_taskIds = taskIds;
  }

  /// Restore all case branches, immutable base voltages, and full base caches.
  /// Used both at wave boundaries and before routing an exceptional GPU wave to
  /// the exact CPU path.
  void restoreBaseState(void)
  {
    p_restoreBaseState();
  }

  // -----------------------------------------------------------
  //  Classify this rank's cases into "batchable" vs "fall back to the
  //  per-contingency CPU path".  Must be called once before solveWave().
  // -----------------------------------------------------------
  void prepare(void)
  {
    p_batchTaskIds.clear();
    p_batchBranches.clear();
    p_nonBatchTaskIds.clear();
    p_screenSkipped = 0;

    // GRIDPACK_BATCH_NOFAST is an accuracy/reference mode.  Do not send its
    // cases through the live GPU protocol: the GA mapper restores topology
    // after each assembly, whereas that protocol intentionally keeps the
    // outage live between chord steps.  Route every task to the established
    // exact CPU path instead of mixing the two state machines.
    if (!p_useFast) {
      p_nonBatchTaskIds = p_taskIds;
      return;
    }

    for (size_t t = 0; t < p_taskIds.size(); t++) {
      int tid = p_taskIds[t];
      Contingency& evt = p_events[tid];
      // Only pure branch outages can share the base pattern.
      if (evt.p_type != Branch) { p_nonBatchTaskIds.push_back(tid); continue; }

      // Cache the branch pointers + their BASE in-service status.  CRITICAL:
      // capture baseStatus BEFORE setContingency toggles the branch out --
      // reading it afterwards records the OUTAGE status (false), so the hot
      // loop's "restore" would leave every case's branch permanently out and
      // outages would accumulate across the wave (the classic silent corruption
      // that made case ~3 onward diverge).
      std::vector<BranchTog> brs;
      int nline = static_cast<int>(evt.p_to.size());
      for (int i = 0; i < nline; i++) {
        std::vector<int> lids =
          p_net->getLocalBranchIndices(evt.p_from[i], evt.p_to[i]);
        for (size_t j = 0; j < lids.size(); j++) {
          PFBranch* br = dynamic_cast<PFBranch*>(p_net->getBranch(lids[j]).get());
          if (br) {
            BranchTog bt;
            bt.br = br; bt.tag = evt.p_ckt[i];
            bt.localIndex = lids[j];
            bt.baseStatus = br->getBranchStatus(evt.p_ckt[i]);  // TRUE base status
            brs.push_back(bt);
          }
        }
      }

      bool eligible = false;
      bool screened = false;
      if (brs.size() == 1 && evt.p_to.size() == 1) {
        const std::pair<int,std::string> key =
          std::make_pair(brs[0].localIndex, brs[0].tag);
        std::map<std::pair<int,std::string>, bool>::const_iterator pos =
          p_lineIsBridge.find(key);
        if (pos != p_lineIsBridge.end()) {
          screened = true;
          eligible = !pos->second;
          ++p_screenSkipped;
        }
      }
      if (!screened) {
        p_restoreStart();
        bool found = p_app.setContingency(evt);
        if (found && p_app.getIslandCount() <= 1 && !p_app.hasLoneBus() &&
            p_structureSignature() == p_baseSig) {
          eligible = true;
        }
        p_app.unSetContingency(evt);
      }
      if (eligible) {
        p_batchTaskIds.push_back(tid);
        p_batchBranches.push_back(brs);
      } else {
        p_nonBatchTaskIds.push_back(tid);
      }
    }

    // Per-case iterate storage for the batch (initialised to the chosen start).
    const size_t W = p_batchTaskIds.size();
    p_caseV.assign(W, p_startV);
    p_caseA.assign(W, p_startA);
  }

  // -----------------------------------------------------------
  //  BatchAssembler interface (over the batchable cases only)
  // -----------------------------------------------------------
  int caseCount(void) const { return static_cast<int>(p_batchTaskIds.size()); }
  int n(void) const   { return p_n; }
  int nnz(void) const { return p_nnz; }
  const int* rowptr(void) const { return p_rowptr.empty() ? NULL : &p_rowptr[0]; }
  const int* colind(void) const { return p_colind.empty() ? NULL : &p_colind[0]; }

  double assemble(int k, double* jac, double* rhs)
  {
    ++p_nAsm;
    if (p_useFast) {
      return p_assembleFast(k, jac, rhs);
    }
    auto t0 = BT_T0();
    p_restoreCase(k);
    p_toggleBranches(k, false);          // branch out of service
    BT_ADD(p_tRestore, t0);
    t0 = BT_T0();
    p_factory->setYBus();
    p_factory->setMode(YBus);
    p_factory->setSBus();
    p_factory->setMode(Jacobian);
    BT_ADD(p_tYbus, t0);
    t0 = BT_T0();
    p_jMap->mapToRealMatrix(p_J);        // reassemble into the base allocation
    BT_ADD(p_tMapJ, t0);
    t0 = BT_T0();
    {
      gridpack::math::PetscSeqCSRView<gridpack::RealType, int> view(*p_J);
      if (static_cast<int>(view.nnz()) != p_nnz) {
        p_toggleBranches(k, true);
        throw gridpack::Exception("pf_batch_ca: contingency changed the "
                                  "Jacobian nonzero pattern (batch invalid)");
      }
      const PetscScalar* a = view.values();
      for (int i = 0; i < p_nnz; i++) jac[i] = static_cast<double>(a[i]);
    }
    BT_ADD(p_tCsr, t0);
    t0 = BT_T0();
    p_factory->setMode(RHS);
    p_vMap->mapToRealVector(p_PQ);
    BT_ADD(p_tMapV, t0);
    t0 = BT_T0();
    double mism = p_extractRhs(rhs);
    BT_ADD(p_tRhs, t0);
    p_toggleBranches(k, true);           // restore base topology
    return mism;
  }

  double update(int k, const double* dx)
  {
    ++p_nUpd;
    if (p_useFast) return p_updateFast(k, dx);
    auto t0 = BT_T0();
    p_restoreCase(k);
    p_toggleBranches(k, false);
    // Load dx into p_X (mapper ordering == CSR row ordering) and apply the
    // Newton correction exactly as PFAppModule::solve() does (setValues: p_v/p_a
    // -= dx).
    {
      PetscInt m = 0;
      PetscScalar* xa =
        gridpack::math::petscVecArray<gridpack::RealType, int>(*p_X, m);
      int lim = (static_cast<int>(m) < p_n) ? static_cast<int>(m) : p_n;
      // Apply the same Newton-step damping PFAppModule::solve() uses
      // (X->scale(p_dampingFactor)); 1.0 => no damping.
      for (int i = 0; i < lim; i++) xa[i] = dx[i] * p_damping;
      gridpack::math::petscVecRestoreArray<gridpack::RealType, int>(*p_X, xa);
    }
    BT_ADD(p_tRestore, t0);
    t0 = BT_T0();
    p_factory->setYBus();
    p_factory->setMode(YBus);
    p_factory->setSBus();
    p_factory->setMode(RHS);
    p_vMap->mapToBus(p_X);
    p_net->updateBuses();
    BT_ADD(p_tUpd, t0);
    t0 = BT_T0();
    p_snapshotCase(k);                   // save the updated iterate
    BT_ADD(p_tSnap, t0);
    t0 = BT_T0();
    p_vMap->mapToRealVector(p_PQ);
    BT_ADD(p_tMapV, t0);
    t0 = BT_T0();
    double mism = p_extractRhs(NULL);
    BT_ADD(p_tRhs, t0);
    p_toggleBranches(k, true);
    return mism;
  }

  // -----------------------------------------------------------
  //  "Live" per-case path (used by the sequential wave): keep case k's state
  //  live across its Newton iterations so restore / branch-toggle / RHS-refresh
  //  happen once per case instead of once per assemble+update.  Always uses the
  //  fast GA-free scatter (validated byte-identical to the GA mapper).
  // -----------------------------------------------------------
  double assembleLive(int k, double* jac, double* rhs)
  {
    if (!p_useFast) return assemble(k, jac, rhs);
    ++p_nAsm;
    auto t0 = BT_T0();
    p_restoreCase(k);
    p_localSetYBus(k, false);            // branch out (stays out for the case)
    BT_ADD(p_tRestore, t0);
    t0 = BT_T0();
    p_factory->setMode(RHS);
    double mism = p_fastRhs(rhs);         // sets Pinj/Qinj + refreshes p_theta
    BT_ADD(p_tMapV, t0);
    t0 = BT_T0();
    p_factory->setMode(Jacobian);
    p_fastJac(jac);
    BT_ADD(p_tMapJ, t0);
    if (p_validateFast) p_validateLiveAssembly(jac, rhs, true);
    return mism;
  }

  double updateLive(int k, const double* dx, double* jac, double* rhs)
  {
    if (!p_useFast) {
      update(k, dx);
      return assemble(k, jac, rhs);
    }
    ++p_nUpd;
    auto t0 = BT_T0();
    for (int i = 0; i < p_n; i++) p_corr[i] = dx[i] * p_damping;
    for (size_t i = 0; i < p_diag.size(); i++)     // apply Newton step (live)
      p_diag[i].bus->setValues(&p_corr[p_diag[i].row0]);
    BT_ADD(p_tUpd, t0);
    t0 = BT_T0();
    p_factory->setMode(RHS);
    double mism = p_fastRhs(rhs);          // new mismatch + Pinj/Qinj + p_theta
    BT_ADD(p_tMapV, t0);
    t0 = BT_T0();
    p_factory->setMode(Jacobian);
    p_fastJac(jac);                        // next Jacobian at the updated state
    BT_ADD(p_tMapJ, t0);
    if (p_validateFast) p_validateLiveAssembly(jac, rhs, true);
    return mism;
  }

  void finishLive(int k)
  {
    p_snapshotCase(k);                     // save converged iterate for output
    if (p_useFast) {
      p_localSetYBus(k, true);             // restore base topology for next case
    } else {
      p_toggleBranches(k, true);
      p_factory->setYBus();
      p_factory->setMode(YBus);
      p_factory->setSBus();
    }
  }

  // -----------------------------------------------------------
  //  Constant-factorization (chord / dishonest-Newton) path.  The base Jacobian
  //  is assembled+factorized ONCE by the engine; each case then iterates with the
  //  base factors, reassembling only the (cheap) RHS.  Because the fixed point is
  //  F(x)=0 regardless of which Jacobian drives the step, any case that converges
  //  lands on the exact contingency solution -- accuracy is identical to Newton;
  //  only the convergence RATE differs, and cases that stall are routed to the
  //  exact-Newton CPU fallback.  This trades O(cases x iters) GPU refactorizations
  //  (each ~250 ms at 48k) for one factorization + many ~ms triangular solves,
  //  which is what lets the GPU wave beat CPU KLU on large systems.
  // -----------------------------------------------------------

  /// Assemble the BASE Jacobian (all branches in service, base voltages) into
  /// @c jac.  Call once, before the case loop, to seed the constant factorization.
  void assembleBaseJac(double* jac)
  {
    p_restoreStart();                      // base solution voltages
    p_factory->setYBus();                  // full base YBus (all branches in)
    p_factory->setMode(YBus);
    p_factory->setSBus();
    p_factory->setMode(Jacobian);
    p_fastJac(jac);
  }

  /// Start case @c k for the chord path: warm-start, take the branch out, and
  /// compute ONLY the RHS (no per-case Jacobian).  Returns the inf-norm mismatch.
  double assembleLiveRhs(int k, double* rhs)
  {
    ++p_nAsm;
    auto t0 = BT_T0();
    p_restoreCase(k);
    p_localSetYBus(k, false);              // branch out (stays out for the case)
    BT_ADD(p_tRestore, t0);
    t0 = BT_T0();
    p_factory->setMode(RHS);
    double mism = p_fastRhs(rhs);
    BT_ADD(p_tMapV, t0);
    return mism;
  }

  /// Chord step for case @c k: apply Newton correction @c dx to the LIVE state and
  /// recompute ONLY the RHS (Jacobian stays the base factorization).  Returns the
  /// new inf-norm mismatch.
  double updateLiveRhs(int k, const double* dx, double* rhs)
  {
    if (!p_useFast) {
      const double mism = update(k, dx);
      p_extractRhs(rhs);
      return mism;
    }
    ++p_nUpd;
    auto t0 = BT_T0();
    for (int i = 0; i < p_n; i++) p_corr[i] = dx[i] * p_damping;
    for (size_t i = 0; i < p_diag.size(); i++)
      p_diag[i].bus->setValues(&p_corr[p_diag[i].row0]);
    BT_ADD(p_tUpd, t0);
    t0 = BT_T0();
    p_factory->setMode(RHS);
    double mism = p_fastRhs(rhs);
    BT_ADD(p_tMapV, t0);
    if (p_validateFast) p_validateLiveAssembly(NULL, rhs, false);
    (void)k;
    return mism;
  }

  void assembleLiveJac(int k, double* jac)
  {
    if (p_useFast) {
      p_factory->setMode(Jacobian);
      p_fastJac(jac);
      if (p_validateFast) p_validateLiveAssembly(jac, NULL, true);
      return;
    }
    std::vector<double> rhs(p_n, 0.0);
    assemble(k, jac, &rhs[0]);
  }

  BatchMismatchInfo lastMismatch(void) const
  {
    return p_lastMismatch;
  }

  // -----------------------------------------------------------
  //  Output helpers used by the driver to capture each batched case's result
  //  through the exact same code path as the per-contingency loop.
  // -----------------------------------------------------------
  int batchTaskId(int k) const { return p_batchTaskIds[k]; }
  const std::vector<int>& nonBatchTaskIds(void) const { return p_nonBatchTaskIds; }
  int screenSkipped(void) const { return p_screenSkipped; }
  int bridgeCount(void) const { return p_screenBridgeCount; }

  /// Put the network into batched case @c k's converged state + topology so the
  /// driver's writeBusString/writeBranchString capture the right numbers.
  Contingency& applyCaseForOutput(int k, bool fullRefresh = false)
  {
    p_restoreCase(k);
    if (fullRefresh) {
      p_toggleBranches(k, false);
      p_factory->setYBus();
      p_factory->setMode(YBus);
      p_factory->setSBus();
    } else {
      p_localSetYBus(k, false);
    }
    // Refresh each bus's p_Pinj/p_Qinj at the converged voltages.  These are
    // recomputed only inside the RHS map (rhsValues); the CPU path gets them
    // from solve(), but the batched overlay skips solve(), so without this the
    // "power" output (generator P/Q, writeStats, csv/json) would report a stale
    // p_Qinj left over from another case in the wave.
    p_factory->setMode(RHS);
    p_fastRhs(NULL);
    return p_events[p_batchTaskIds[k]];
  }

  /// Undo applyCaseForOutput(): return the branch(es) to service.
  void clearCaseForOutput(int k, bool fullRefresh = false)
  {
    if (fullRefresh) {
      p_toggleBranches(k, true);
      p_factory->setYBus();
      p_factory->setMode(YBus);
      p_factory->setSBus();
    } else {
      p_localSetYBus(k, true);
    }
  }

private:

  void p_buildConnectivityScreen(void)
  {
    std::vector<int> from, to;
    std::vector<std::pair<int,std::string> > keys;
    std::vector<int> screenIndex(p_nbus, -1);
    int activeBusCount = 0;
    for (int i = 0; i < p_nbus; ++i) {
      if (!p_net->getActiveBus(i)) continue;
      PFBus* bus = dynamic_cast<PFBus*>(p_net->getBus(i).get());
      if (!bus || bus->isIsolated()) continue;
      screenIndex[i] = activeBusCount++;
    }

    const int nbranch = p_net->numBranches();
    for (int bi = 0; bi < nbranch; ++bi) {
      if (!p_net->getActiveBranch(bi)) continue;
      PFBranch* branch =
        dynamic_cast<PFBranch*>(p_net->getBranch(bi).get());
      if (!branch) continue;
      int u = -1, v = -1;
      p_net->getBranchEndpoints(bi, &u, &v);
      if (u < 0 || u >= p_nbus || v < 0 || v >= p_nbus ||
          screenIndex[u] < 0 || screenIndex[v] < 0) {
        continue;
      }
      const std::vector<std::string> tags = branch->getLineIDs();
      for (size_t j = 0; j < tags.size(); ++j) {
        if (!branch->getBranchStatus(tags[j])) continue;
        from.push_back(screenIndex[u]);
        to.push_back(screenIndex[v]);
        keys.push_back(std::make_pair(bi, tags[j]));
      }
    }
    N1ConnectivityScreen screen(activeBusCount, from, to);
    const int baseComponents = screen.componentsWithout(-1);
    // The bridge shortcut is authoritative only for a connected active base.
    // On an already-disconnected base, leave the lookup empty so prepare()
    // performs GridPACK's complete per-case topology/structure probe.
    if (baseComponents != 1) return;
    const std::vector<int> outageComponents = screen.screenAllBranchOutages();
    for (size_t e = 0; e < keys.size(); ++e) {
      const bool bridge = outageComponents[e] > baseComponents;
      p_lineIsBridge[keys[e]] = bridge;
      if (bridge) ++p_screenBridgeCount;
    }
  }

  // --- per-bus structure signature (matches how the mapper sizes the system)
  std::vector<int> p_structureSignature(void)
  {
    p_factory->setMode(Jacobian);
    std::vector<int> sig(p_nbus, 0);
    for (int i = 0; i < p_nbus; i++) {
      if (!p_isActiveBus[i]) continue;
      PFBus* bus = dynamic_cast<PFBus*>(p_net->getBus(i).get());
      if (!bus) continue;
      int is = 0, js = 0;
      sig[i] = bus->matrixDiagSize(&is, &js) ? is : 0;
    }
    return sig;
  }

  void p_restoreStart(void)
  {
    for (int i = 0; i < p_nbus; i++) {
      if (!p_isActiveBus[i]) continue;
      PFBus* bus = dynamic_cast<PFBus*>(p_net->getBus(i).get());
      if (bus) bus->setVoltageState(p_startV[i], p_startA[i]);
    }
    p_net->updateBuses();
  }

  void p_restoreBaseState(void)
  {
    for (size_t k = 0; k < p_batchBranches.size(); ++k) {
      p_toggleBranches(static_cast<int>(k), true);
    }
    p_restoreStart();
    p_factory->setYBus();
    p_factory->setMode(YBus);
    p_factory->setSBus();
  }

  void p_restoreCase(int k)
  {
    for (int i = 0; i < p_nbus; i++) {
      if (!p_isActiveBus[i]) continue;
      PFBus* bus = dynamic_cast<PFBus*>(p_net->getBus(i).get());
      if (bus) bus->setVoltageState(p_caseV[k][i], p_caseA[k][i]);
    }
    p_net->updateBuses();
  }

  void p_snapshotCase(int k)
  {
    for (int i = 0; i < p_nbus; i++) {
      if (!p_isActiveBus[i]) continue;
      PFBus* bus = dynamic_cast<PFBus*>(p_net->getBus(i).get());
      if (!bus) continue;
      double v = 0.0, a = 0.0;
      bus->getVoltageState(v, a);
      p_caseV[k][i] = v; p_caseA[k][i] = a;
    }
  }

  // inService=false => take the contingency branch(es) out of service;
  // inService=true  => restore them to their BASE status (which may itself be
  // out-of-service -- so a contingency on an already-out circuit is a correct
  // no-op, never a spurious energization).
  void p_toggleBranches(int k, bool inService)
  {
    const std::vector<BranchTog>& brs = p_batchBranches[k];
    for (size_t j = 0; j < brs.size(); j++) {
      brs[j].br->setBranchStatus(brs[j].tag, inService ? brs[j].baseStatus : false);
    }
  }

  double p_extractRhs(double* rhs)
  {
    PetscInt m = 0;
    const PetscScalar* a =
      gridpack::math::petscVecArrayRead<gridpack::RealType, int>(*p_PQ, m);
    double inf = 0.0;
    for (int i = 0; i < static_cast<int>(m); i++) {
      double val = static_cast<double>(a[i]);
      if (rhs && i < p_n) rhs[i] = val;
      double av = (val < 0.0) ? -val : val;
      if (av > inf) inf = av;
    }
    gridpack::math::petscVecRestoreArrayRead<gridpack::RealType, int>(*p_PQ, a);
    return inf;
  }

  // ===========================================================================
  //  GA-free direct-CSR-scatter fast assembler
  // ===========================================================================

  /// CSR slot holding (row,col) in the fixed pattern (build-once linear scan).
  int p_findSlot(int row, int col) const
  {
    if (row < 0 || row >= p_n || col < 0 || col >= p_n) {
      throw gridpack::Exception(
          "pf_batch_ca: component entry is outside the reduced Jacobian");
    }
    for (int k = p_rowptr[row]; k < p_rowptr[row + 1]; k++)
      if (p_colind[k] == col) return k;
    throw gridpack::Exception(
        "pf_batch_ca: component entry is missing from the base CSR pattern");
  }

  /// Precompute the component-block -> CSR-slot map from the base topology (which
  /// is exactly the fixed pattern captured in the ctor).  Runs once.
  void p_buildScatterMap(void)
  {
    p_factory->setYBus();
    p_factory->setMode(YBus);
    p_factory->setSBus();
    p_factory->setMode(Jacobian);

    // Row/col offset per MatVecIndex = prefix sum of block sizes over lower
    // indices (PV bus -> 1, PQ bus -> 2, slack/isolated -> 0).
    int maxmvi = -1;
    std::vector<int> sizeByMVI;
    std::vector<PFBus*> busByMVI;
    for (int i = 0; i < p_nbus; i++) {
      if (!p_isActiveBus[i]) continue;
      PFBus* bus = dynamic_cast<PFBus*>(p_net->getBus(i).get());
      if (!bus) continue;
      int mvi = -1; bus->getMatVecIndex(&mvi);
      if (mvi < 0) continue;
      if (mvi > maxmvi) {
        sizeByMVI.resize(mvi + 1, 0);
        busByMVI.resize(mvi + 1, (PFBus*)NULL);
        maxmvi = mvi;
      }
      int is = 0, js = 0;
      sizeByMVI[mvi] = bus->matrixDiagSize(&is, &js) ? is : 0;
      busByMVI[mvi] = bus;
    }
    p_offByMVI.assign(maxmvi + 2, 0);
    for (int m = 0; m <= maxmvi; m++)
      p_offByMVI[m + 1] = p_offByMVI[m] + sizeByMVI[m];
    if (p_offByMVI.empty() || p_offByMVI.back() != p_n) {
      throw gridpack::Exception(
          "pf_batch_ca: component block sizes do not cover the reduced Jacobian");
    }

    // Diagonal (bus) blocks.
    p_diag.clear();
    for (int m = 0; m <= maxmvi; m++) {
      PFBus* bus = busByMVI[m];
      if (!bus) continue;
      int is = 0, js = 0;
      if (!bus->matrixDiagSize(&is, &js)) continue;   // slack/isolated
      DiagRec r; r.bus = bus; r.row0 = p_offByMVI[m]; r.vn = is; r.n = is * js;
      int icnt = 0;
      for (int k = 0; k < js; k++)          // column-major (equation index fastest)
        for (int j = 0; j < is; j++)
          r.slot[icnt++] = p_findSlot(p_offByMVI[m] + j, p_offByMVI[m] + k);
      p_diag.push_back(r);
    }

    // Off-diagonal (branch) blocks: forward at (bus1 rows, bus2 cols), reverse at
    // (bus2 rows, bus1 cols).  Sizes come from the component (PV endpoints shrink).
    p_off.clear();
    int nbr = p_net->numBranches();
    for (int b = 0; b < nbr; b++) {
      PFBranch* br = dynamic_cast<PFBranch*>(p_net->getBranch(b).get());
      if (!br) continue;
      int m1 = -1, m2 = -1; br->getMatVecIndices(&m1, &m2);
      if (m1 < 0 || m2 < 0 || m1 > maxmvi || m2 > maxmvi) continue;
      BrRec r; r.br = br; r.nf = 0; r.nr = 0;
      int is = 0, js = 0;
      if (br->matrixForwardSize(&is, &js)) {
        r.nf = is * js; int icnt = 0;
        for (int k = 0; k < js; k++)
          for (int j = 0; j < is; j++)
            r.fslot[icnt++] = p_findSlot(p_offByMVI[m1] + j, p_offByMVI[m2] + k);
      }
      if (br->matrixReverseSize(&is, &js)) {
        r.nr = is * js; int icnt = 0;
        for (int k = 0; k < js; k++)
          for (int j = 0; j < is; j++)
            r.rslot[icnt++] = p_findSlot(p_offByMVI[m2] + j, p_offByMVI[m1] + k);
      }
      if (r.nf || r.nr) p_off.push_back(r);
    }
  }

  /// Toggle case k's branch(es) and patch ONLY the two endpoint buses' Ybus
  /// diagonals (O(deg), not a full-network setYBus).  Ybus is voltage-independent
  /// so this is exact; getPQ later refreshes p_theta from the live angles.
  void p_localSetYBus(int k, bool inService)
  {
    const std::vector<BranchTog>& brs = p_batchBranches[k];
    for (size_t j = 0; j < brs.size(); j++) {
      brs[j].br->setBranchStatus(brs[j].tag,
                                 inService ? brs[j].baseStatus : false);
      brs[j].br->setYBus();
      PFBus* b1 = dynamic_cast<PFBus*>(brs[j].br->getBus1().get());
      PFBus* b2 = dynamic_cast<PFBus*>(brs[j].br->getBus2().get());
      if (b1) b1->setYBus();
      if (b2) b2->setYBus();
    }
  }

  /// RHS pass (must precede p_fastJac each iteration): rhsValues stores each
  /// bus's Pinj/Qinj (read by the diagonal Jacobian) and getPQ refreshes every
  /// incident branch's p_theta from the live angles.  Writes the reduced
  /// mismatch (if rhs != NULL) and returns its inf-norm.
  double p_fastRhs(double* rhs) const
  {
    double vals[2]; double inf = 0.0;
    p_lastMismatch = BatchMismatchInfo();
    for (size_t i = 0; i < p_diag.size(); i++) {
      const DiagRec& r = p_diag[i];
      if (!r.bus->vectorValues(vals)) {
        throw gridpack::Exception(
            "pf_batch_ca: RHS structure no longer matches the Jacobian scatter map");
      }
      for (int t = 0; t < r.vn; t++) {
        double v = vals[t];
        if (rhs) rhs[r.row0 + t] = v;
        double av = v < 0.0 ? -v : v;
        if (av > inf) inf = av;
        const double engineeringMismatch = av * r.bus->getSBase();
        if (t == 0 && engineeringMismatch > p_lastMismatch.maxPMismatch) {
          p_lastMismatch.maxPMismatch = engineeringMismatch;
          p_lastMismatch.maxPBus = r.bus->getOriginalIndex();
        } else if (t == 1 &&
                   engineeringMismatch > p_lastMismatch.maxQMismatch) {
          p_lastMismatch.maxQMismatch = engineeringMismatch;
          p_lastMismatch.maxQBus = r.bus->getOriginalIndex();
        }
      }
    }
    return inf;
  }

  /// Jacobian pass: scatter each component's block straight into the CSR value
  /// array.  Toggled-out branches (matrix*Values -> false) leave their slots at
  /// the zero set by std::fill, exactly reproducing the removed contribution.
  void p_fastJac(double* jac) const
  {
    std::fill(jac, jac + p_nnz, 0.0);
    double vals[4];
    for (size_t i = 0; i < p_diag.size(); i++) {
      const DiagRec& r = p_diag[i];
      if (!r.bus->matrixDiagValues(vals)) continue;
      for (int t = 0; t < r.n; t++)
        jac[r.slot[t]] += vals[t];
    }
    for (size_t i = 0; i < p_off.size(); i++) {
      const BrRec& r = p_off[i];
      if (r.nf && r.br->matrixForwardValues(vals))
        for (int t = 0; t < r.nf; t++)
          jac[r.fslot[t]] += vals[t];
      if (r.nr && r.br->matrixReverseValues(vals))
        for (int t = 0; t < r.nr; t++)
          jac[r.rslot[t]] += vals[t];
    }
  }

  double p_assembleFast(int k, double* jac, double* rhs)
  {
    auto t0 = BT_T0();
    p_restoreCase(k);
    p_localSetYBus(k, false);
    BT_ADD(p_tRestore, t0);
    t0 = BT_T0();
    p_factory->setMode(RHS);
    double mism = p_fastRhs(rhs);
    BT_ADD(p_tMapV, t0);
    t0 = BT_T0();
    p_factory->setMode(Jacobian);
    p_fastJac(jac);
    BT_ADD(p_tMapJ, t0);
    p_localSetYBus(k, true);
    if (p_dbg && k < 8)
      std::fprintf(stderr, "[trace k%d] assemble mism=%.6e\n", k, mism);
    return mism;
  }

  double p_updateFast(int k, const double* dx)
  {
    auto t0 = BT_T0();
    p_restoreCase(k);
    p_localSetYBus(k, false);
    for (int i = 0; i < p_n; i++) p_corr[i] = dx[i] * p_damping;
    // Apply the Newton correction directly (replaces GA mapToBus): each
    // contributing bus consumes p_corr[row0 .. row0+size-1] via PFBus::setValues.
    for (size_t i = 0; i < p_diag.size(); i++)
      p_diag[i].bus->setValues(&p_corr[p_diag[i].row0]);
    BT_ADD(p_tUpd, t0);
    t0 = BT_T0();
    p_snapshotCase(k);
    BT_ADD(p_tSnap, t0);
    t0 = BT_T0();
    p_factory->setMode(RHS);
    double mism = p_fastRhs(NULL);
    BT_ADD(p_tMapV, t0);
    p_localSetYBus(k, true);
    if (p_dbg && k < 8) {
      double dxn = 0.0; for (int i = 0; i < p_n; i++) { double a = dx[i]<0?-dx[i]:dx[i]; if (a>dxn) dxn=a; }
      std::fprintf(stderr, "[trace k%d] update max|dx|=%.6e -> mism=%.6e\n", k, dxn, mism);
    }
    return mism;
  }

  // Expensive, opt-in reference check.  The contingency and voltage state are
  // already live here, so rebuild the same state through GridPACK's GA mappers
  // and require both the CSR structure and numeric values to match.  This runs
  // on every requested assembly (GRIDPACK_BATCH_VALIDATE=1), including Release
  // builds, and fails closed rather than merely printing a warning.
  void p_validateLiveAssembly(const double* fastJac, const double* fastRhs,
                              bool checkJac)
  {
    std::vector<double> referenceJac;
    std::vector<double> referenceRhs;

    p_factory->setYBus();
    p_factory->setMode(YBus);
    p_factory->setSBus();

    if (checkJac) {
      p_factory->setMode(Jacobian);
      p_jMap->mapToRealMatrix(p_J);
      gridpack::math::PetscSeqCSRView<gridpack::RealType, int> view(*p_J);
      if (view.rows() != p_n || view.nnz() != p_nnz) {
        throw gridpack::Exception(
            "pf_batch_ca validation: Jacobian dimensions changed");
      }
      const PetscInt* ia = view.rowptr();
      const PetscInt* ja = view.colind();
      const PetscScalar* values = view.values();
      referenceJac.resize(p_nnz);
      for (int i = 0; i <= p_n; ++i) {
        if (ia[i] != p_rowptr[i]) {
          throw gridpack::Exception(
              "pf_batch_ca validation: Jacobian row pattern changed");
        }
      }
      for (int i = 0; i < p_nnz; ++i) {
        if (ja[i] != p_colind[i]) {
          throw gridpack::Exception(
              "pf_batch_ca validation: Jacobian column pattern changed");
        }
        referenceJac[i] = static_cast<double>(values[i]);
      }
    }

    if (fastRhs) {
      referenceRhs.resize(p_n);
      p_factory->setMode(RHS);
      p_vMap->mapToRealVector(p_PQ);
      p_extractRhs(&referenceRhs[0]);
    }

    double maxDifference = 0.0;
    double scale = 1.0;
    if (checkJac) {
      if (!fastJac) {
        throw gridpack::Exception(
            "pf_batch_ca validation: missing fast Jacobian values");
      }
      for (int i = 0; i < p_nnz; ++i) {
        maxDifference =
          std::max(maxDifference, std::fabs(fastJac[i] - referenceJac[i]));
        scale = std::max(scale, std::fabs(referenceJac[i]));
      }
    }
    if (fastRhs) {
      for (int i = 0; i < p_n; ++i) {
        maxDifference =
          std::max(maxDifference, std::fabs(fastRhs[i] - referenceRhs[i]));
        scale = std::max(scale, std::fabs(referenceRhs[i]));
      }
    }
    if (!std::isfinite(maxDifference) || maxDifference > 1.0e-12 * scale) {
      char message[256];
      std::snprintf(message, sizeof(message),
                    "pf_batch_ca validation: fast assembly differs from "
                    "GridPACK reference (max |difference| %.6e, scale %.6e)",
                    maxDifference, scale);
      throw gridpack::Exception(message);
    }
  }

  PFAppModule& p_app;
  boost::shared_ptr<PFNetwork> p_net;
  boost::shared_ptr<PFFactoryModule> p_factory;
  std::vector<Contingency>& p_events;
  std::vector<int> p_taskIds;

  boost::scoped_ptr<gridpack::mapper::FullMatrixMap<PFNetwork> > p_jMap;
  boost::scoped_ptr<gridpack::mapper::BusVectorMap<PFNetwork> > p_vMap;
  boost::shared_ptr<gridpack::math::RealMatrix> p_J;
  boost::shared_ptr<gridpack::math::RealVector> p_PQ, p_X;

  int p_n, p_nnz, p_nbus;
  bool p_warmStart;
  double p_damping;

  // env-gated hot-path micro-profile accumulators (seconds / counts)
  mutable double p_tRestore, p_tYbus, p_tMapJ, p_tCsr, p_tMapV, p_tRhs,
                 p_tUpd, p_tSnap;
  mutable long p_nAsm, p_nUpd;

  // ---- GA-free direct-CSR-scatter fast assembler --------------------------
  // Replaces the Global-Arrays FullMatrixMap/BusVectorMap in the hot loop: the
  // reduced-Jacobian PATTERN is fixed across a wave, so we precompute, once, the
  // CSR slot each component block-entry lands in (from getMatVecIndex offsets)
  // and then per iteration call the SAME PFBus/PFBranch matrixValues/vectorValues
  // (numerically identical to the mapper) writing straight into the CSR arrays.
  // Ybus is voltage-independent, so a branch toggle is a LOCAL O(deg) setYBus on
  // the two endpoints instead of a full-network recompute.
  bool p_useFast;                 ///< use the fast scatter path (default true)
  bool p_validateFast;            ///< compare live fast values with GA reference
  bool p_dbg;                     ///< env-gated convergence trace
  std::vector<int> p_offByMVI;    ///< reduced-J row/col offset per MatVecIndex
  struct DiagRec { PFBus* bus; int row0; int vn; int n; int slot[4]; };
  struct BrRec   { PFBranch* br; int nf; int fslot[4]; int nr; int rslot[4]; };
  std::vector<DiagRec> p_diag;    ///< one per contributing bus (diagonal block)
  std::vector<BrRec>   p_off;     ///< one per contributing branch (fwd+rev blocks)
  mutable std::vector<double> p_corr;  ///< reusable Newton-correction scratch (n)
  std::vector<int> p_rowptr, p_colind;
  std::vector<int> p_baseSig;
  std::vector<double> p_startV, p_startA;
  std::vector<char> p_isActiveBus;
  mutable BatchMismatchInfo p_lastMismatch;

  // A contingency branch to toggle, with its BASE status for correct restore.
  struct BranchTog {
    PFBranch* br;
    std::string tag;
    bool baseStatus;
    int localIndex;
  };

  // batchable cases
  std::vector<int> p_batchTaskIds;
  std::vector<std::vector<BranchTog> > p_batchBranches;
  std::vector<std::vector<double> > p_caseV, p_caseA;
  // everything else -> driver runs it through the per-contingency path
  std::vector<int> p_nonBatchTaskIds;
  int p_screenSkipped;
  bool p_screenEnabled;
  int p_screenBridgeCount;
  std::map<std::pair<int,std::string>, bool> p_lineIsBridge;

  GridpackBatchAssembler(const GridpackBatchAssembler&);
  GridpackBatchAssembler& operator=(const GridpackBatchAssembler&);
};

} // namespace powerflow
} // namespace gridpack

#endif // GRIDPACK_WITH_CUDSS

#endif
