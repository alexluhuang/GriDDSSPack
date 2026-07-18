// Emacs Mode Line: -*- Mode:c++;-*-
// -------------------------------------------------------------
/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
// -------------------------------------------------------------
/**
 * @file   cudss_linear_solver_implementation.hpp
 * @brief  GPU direct-LU LinearSolver backend built on NVIDIA cuDSS.
 *
 * This class is the GPU counterpart of PETScLinearSolverImplementation and
 * implements the same LinearSolverImplementation<T,I> contract.  It maps
 * GridPACK's solver lifecycle onto cuDSS's three phases and exploits the fact
 * that a power-flow Jacobian keeps a FIXED sparsity pattern across Newton
 * iterations:
 *
 *   construction / first solve : cudssExecute(CUDSS_PHASE_ANALYSIS)  -- ONCE
 *                                (reordering + symbolic factorization; depends
 *                                 only on the pattern, which never changes)
 *   each Newton iteration      : cudssExecute(CUDSS_PHASE_FACTORIZATION)
 *                                (numeric refactorization, reusing the analysis)
 *   solve(b,x)                 : cudssExecute(CUDSS_PHASE_SOLVE)
 *                                (optionally with iterative refinement)
 *
 * This is the Phase-1 "drop-in" backend: a fresh instance is created per
 * contingency (mirroring the existing per-contingency LinearSolver), so the
 * symbolic analysis is amortized across that contingency's Newton iterations.
 * Amortizing ONE analysis across the whole N-1 sweep, plus batching many
 * contingencies together, is the Phase-2 batched engine (see
 * applications/modules/powerflow/pf_batch_ca.hpp).
 *
 * Because both p_solveImpl() and p_resolveImpl() are const (per the base
 * interface), all GPU-side state that changes between calls is `mutable`, in
 * exactly the same spirit as PETScLinearSolverImplementation's mutable KSP
 * bookkeeping.
 *
 * Numerical parity: cuDSS performs an exact sparse LU factorization, the same
 * numerical method as the CPU KLU/SuperLU_DIST path, so per-case results (and
 * therefore the CA output CSVs) match the CPU run to FP64 round-off.
 *
 * Only compiled when GRIDPACK_WITH_CUDSS is defined.
 */
// -------------------------------------------------------------

#ifndef _cudss_linear_solver_implementation_hpp_
#define _cudss_linear_solver_implementation_hpp_

#ifdef GRIDPACK_WITH_CUDSS

#include <cstddef>
#include <cuda_runtime.h>
#include <cudss.h>

#include "linear_solver_implementation.hpp"
#include "cudss/cudss_exception.hpp"
#include "cudss/cudss_csr_extractor.hpp"

namespace gridpack {
namespace math {

// -------------------------------------------------------------
//  class CuDSSLinearSolverImplementation
// -------------------------------------------------------------
template <typename T, typename I>
class CuDSSLinearSolverImplementation
  : public LinearSolverImplementation<T, I>
{
public:

  typedef typename LinearSolverImplementation<T, I>::MatrixType MatrixType;
  typedef typename LinearSolverImplementation<T, I>::VectorType VectorType;

  /// Default constructor.
  /**
   * Creates the cuDSS library context.  Throws gridpack::Exception if cuDSS or
   * the CUDA runtime cannot be initialized, which the runtime backend selector
   * catches to fall back to the PETSc path.  (Device availability is checked
   * before construction via cudssBackendAvailable().)
   */
  CuDSSLinearSolverImplementation(MatrixType& A)
    : LinearSolverImplementation<T, I>(A),
      p_handle(NULL), p_config(NULL), p_data(NULL),
      p_Amat(NULL), p_bMat(NULL), p_xMat(NULL),
      p_d_rowptr(NULL), p_d_colind(NULL), p_d_values(NULL),
      p_d_rhs(NULL), p_d_sol(NULL),
      p_analyzed(false), p_matricesCreated(false),
      p_nCached(-1), p_nnzCached(-1),
      p_irSteps(0), p_deterministic(false), p_hybridMemory(false),
      p_refactorEvery(1), p_solveCount(0), p_factored(false)
  {
    p_valType = cudssPetscValueType();
    p_idxType = cudssPetscIndexType();

    // Create the cuDSS context.  On partial failure destroy whatever already
    // succeeded before rethrowing, so the PETSc fallback path (which catches
    // this exception) leaks no cuDSS handle/config.
    try {
      GP_CUDSS_CHECK(cudssCreate(&p_handle));
      GP_CUDSS_CHECK(cudssConfigCreate(&p_config));
      GP_CUDSS_CHECK(cudssDataCreate(p_handle, &p_data));
    } catch (...) {
      if (p_data)   { cudssDataDestroy(p_handle, p_data); p_data = NULL; }
      if (p_config) { cudssConfigDestroy(p_config);       p_config = NULL; }
      if (p_handle) { cudssDestroy(p_handle);             p_handle = NULL; }
      throw;
    }

    // The cuDSS backend requires a serial (SEQAIJ) coefficient matrix.  In a
    // multi-process task group each rank collects the full system (via the base
    // class' serial path) and solves it on its GPU.  groupSize=1 (the
    // recommended GPU configuration) avoids this redundancy entirely.
    this->p_doSerial = (this->processor_size() > 1);
    this->p_constSerialMatrix = false;
  }

  /// Destructor
  ~CuDSSLinearSolverImplementation(void)
  {
    // Never throw from a destructor: release everything best-effort.
    p_freeDeviceAndMatrices();
    if (p_data)   { cudssDataDestroy(p_handle, p_data); p_data = NULL; }
    if (p_config) { cudssConfigDestroy(p_config);       p_config = NULL; }
    if (p_handle) { cudssDestroy(p_handle);             p_handle = NULL; }
  }

protected:

  // ---- cuDSS library objects -------------------------------------------
  cudssHandle_t p_handle;
  cudssConfig_t p_config;
  cudssData_t   p_data;

  // ---- cuDSS matrix wrappers (recreated when the pattern size changes) --
  mutable cudssMatrix_t p_Amat;   ///< sparse system matrix (CSR)
  mutable cudssMatrix_t p_bMat;   ///< dense right-hand side
  mutable cudssMatrix_t p_xMat;   ///< dense solution

  // ---- device buffers ---------------------------------------------------
  mutable void *p_d_rowptr;   ///< row offsets  (n+1) of PetscInt
  mutable void *p_d_colind;   ///< column indices (nnz) of PetscInt
  mutable void *p_d_values;   ///< nonzero values (nnz) of PetscScalar
  mutable void *p_d_rhs;      ///< RHS (n) of PetscScalar
  mutable void *p_d_sol;      ///< solution (n) of PetscScalar

  // ---- cached factorization state --------------------------------------
  mutable bool     p_analyzed;         ///< has ANALYSIS been run for the pattern
  mutable bool     p_matricesCreated;  ///< have the cudssMatrix wrappers been created
  mutable PetscInt p_nCached;          ///< rows the buffers/analysis were sized for
  mutable PetscInt p_nnzCached;        ///< nnz the buffers/analysis were sized for

  // ---- configuration ----------------------------------------------------
  int  p_irSteps;        ///< iterative-refinement steps (0 = off)
  bool p_deterministic;  ///< reproducible (bit-wise) factorization
  bool p_hybridMemory;   ///< prefer host/unified pointers (reserved)
  cudssDataType_t p_valType;   ///< cuDSS value type of the PETSc scalar
  cudssDataType_t p_idxType;   ///< cuDSS integer type of the PETSc index

  // ---- Phase-4 constant-factorization ("factor-once, solve-many") -------
  // Refactorize only every p_refactorEvery-th solve; intervening solves reuse
  // the existing LU factors (modified/chord Newton).  1 = refactor every solve
  // (exact Newton, the default).  A large value = factor once and reuse -- the
  // constant-factor throughput mode the fast-decoupled method exploits, here
  // realized on the full Jacobian without touching the verified assembly.  It
  // converges to the SAME solution (possibly in more iterations).
  int  p_refactorEvery;
  mutable long p_solveCount;   ///< solves since construction
  mutable bool p_factored;     ///< do valid LU factors exist for the current pattern

  // ---------------------------------------------------------------------
  // p_freeDeviceAndMatrices
  // ---------------------------------------------------------------------
  void p_freeDeviceAndMatrices(void) const
  {
    // Destroy per-pointer (NOT gated on p_matricesCreated) so partially-created
    // wrappers are released too -- e.g. if cudssMatrixCreateDn threw after
    // cudssMatrixCreateCsr succeeded, p_matricesCreated is still false but
    // p_Amat is non-NULL and must be freed.
    if (p_Amat) { cudssMatrixDestroy(p_Amat); p_Amat = NULL; }
    if (p_bMat) { cudssMatrixDestroy(p_bMat); p_bMat = NULL; }
    if (p_xMat) { cudssMatrixDestroy(p_xMat); p_xMat = NULL; }
    p_matricesCreated = false;
    if (p_d_rowptr) { cudaFree(p_d_rowptr); p_d_rowptr = NULL; }
    if (p_d_colind) { cudaFree(p_d_colind); p_d_colind = NULL; }
    if (p_d_values) { cudaFree(p_d_values); p_d_values = NULL; }
    if (p_d_rhs)    { cudaFree(p_d_rhs);    p_d_rhs = NULL; }
    if (p_d_sol)    { cudaFree(p_d_sol);    p_d_sol = NULL; }
    p_analyzed = false;
    p_factored = false;   // factors are invalid once the pattern is rebuilt
  }

  // ---------------------------------------------------------------------
  // p_prepare : (re)allocate device buffers, upload the fixed pattern, and
  // run the symbolic ANALYSIS exactly once for a given pattern size.
  // ---------------------------------------------------------------------
  void p_prepare(const PetscSeqCSRView<T, I>& csr) const
  {
    const PetscInt n   = csr.rows();
    const PetscInt nnz = csr.nnz();

    if (p_analyzed && n == p_nCached && nnz == p_nnzCached) {
      return;  // pattern unchanged -- reuse existing analysis
    }

    // Pattern (size) changed: rebuild everything.
    p_freeDeviceAndMatrices();

    const std::size_t szInt = sizeof(PetscInt);
    const std::size_t szVal = sizeof(PetscScalar);

    GP_CUDA_CHECK(cudaMalloc(&p_d_rowptr, (n + 1) * szInt));
    GP_CUDA_CHECK(cudaMalloc(&p_d_colind, (nnz > 0 ? nnz : 1) * szInt));
    GP_CUDA_CHECK(cudaMalloc(&p_d_values, (nnz > 0 ? nnz : 1) * szVal));
    GP_CUDA_CHECK(cudaMalloc(&p_d_rhs,    (n > 0 ? n : 1) * szVal));
    GP_CUDA_CHECK(cudaMalloc(&p_d_sol,    (n > 0 ? n : 1) * szVal));

    // Upload the fixed sparsity pattern (row offsets + column indices) once.
    GP_CUDA_CHECK(cudaMemcpy(p_d_rowptr, csr.rowptr(), (n + 1) * szInt,
                             cudaMemcpyHostToDevice));
    if (nnz > 0) {
      GP_CUDA_CHECK(cudaMemcpy(p_d_colind, csr.colind(), nnz * szInt,
                               cudaMemcpyHostToDevice));
    }

    // Create the cuDSS matrix wrappers around the device buffers.  Standard
    // compressed CSR: rowEnd = NULL (rowStart holds n+1 offsets).  The Jacobian
    // is a general nonsymmetric matrix, viewed in full, 0-based.
    GP_CUDSS_CHECK(cudssMatrixCreateCsr(
        &p_Amat, n, n, nnz,
        p_d_rowptr, /*rowEnd=*/NULL, p_d_colind, p_d_values,
        p_idxType, p_idxType, p_valType,
        CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
    GP_CUDSS_CHECK(cudssMatrixCreateDn(
        &p_bMat, n, 1, n, p_d_rhs, p_valType, CUDSS_LAYOUT_COL_MAJOR));
    GP_CUDSS_CHECK(cudssMatrixCreateDn(
        &p_xMat, n, 1, n, p_d_sol, p_valType, CUDSS_LAYOUT_COL_MAJOR));
    p_matricesCreated = true;

    // Symbolic analysis (reordering + symbolic factorization) -- depends only
    // on the pattern, so it happens exactly once per pattern size.
    GP_CUDSS_CHECK(cudssExecute(p_handle, CUDSS_PHASE_ANALYSIS,
                                p_config, p_data, p_Amat, p_xMat, p_bMat));

    p_analyzed  = true;
    p_nCached   = n;
    p_nnzCached = nnz;
  }

  // ---------------------------------------------------------------------
  // p_solveImpl : refactorize with the current coefficients, then solve.
  // ---------------------------------------------------------------------
  void p_solveImpl(MatrixType& A, const VectorType& b, VectorType& x) const
  {
    PetscSeqCSRView<T, I> csr(A);

    p_prepare(csr);   // alloc + pattern upload + ANALYSIS (once per pattern)

    // Constant-factorization (Phase 4): refactorize only every p_refactorEvery
    // solves; otherwise reuse the existing LU factors (modified Newton).  Always
    // (re)factor if no valid factors exist yet for the current pattern.
    const bool refactor = (!p_factored) || (p_refactorEvery <= 1) ||
                          (p_solveCount % p_refactorEvery == 0);
    ++p_solveCount;
    if (refactor) {
      // Upload the (changed) nonzero values and numerically refactorize.
      const PetscInt nnz = csr.nnz();
      if (nnz > 0) {
        GP_CUDA_CHECK(cudaMemcpy(p_d_values, csr.values(),
                                 nnz * sizeof(PetscScalar),
                                 cudaMemcpyHostToDevice));
      }
      GP_CUDSS_CHECK(cudssExecute(p_handle, CUDSS_PHASE_FACTORIZATION,
                                  p_config, p_data, p_Amat, p_xMat, p_bMat));
      p_factored = true;
    }

    // Solve (with fresh or reused factors).
    this->p_resolveImpl(b, x);
  }

  // ---------------------------------------------------------------------
  // p_resolveImpl : solve again using the existing factorization.
  // ---------------------------------------------------------------------
  void p_resolveImpl(const VectorType& b, VectorType& x) const
  {
    if (!p_analyzed) {
      throw gridpack::Exception(
        "cuDSS backend: resolve() called before any factorization");
    }
    const PetscInt n = p_nCached;

    // Upload RHS.
    PetscInt nb = 0;
    const PetscScalar *barr = petscVecArrayRead(b, nb);
    if (nb != n) {
      petscVecRestoreArrayRead(b, barr);
      throw gridpack::Exception("cuDSS backend: RHS size does not match matrix");
    }
    cudaError_t cerr = cudaMemcpy(p_d_rhs, barr, n * sizeof(PetscScalar),
                                  cudaMemcpyHostToDevice);
    petscVecRestoreArrayRead(b, barr);
    GP_CUDA_CHECK(cerr);

    // Solve (optionally with iterative refinement, set via config).
    GP_CUDSS_CHECK(cudssExecute(p_handle, CUDSS_PHASE_SOLVE,
                                p_config, p_data, p_Amat, p_xMat, p_bMat));
    GP_CUDA_CHECK(cudaDeviceSynchronize());

    // Download solution into the PETSc vector's array (same pattern as the
    // PETSc backend writing directly into the underlying Vec).
    PetscInt nx = 0;
    PetscScalar *xarr = petscVecArray(x, nx);
    if (nx != n) {
      petscVecRestoreArray(x, xarr);
      throw gridpack::Exception("cuDSS backend: solution size does not match matrix");
    }
    cerr = cudaMemcpy(xarr, p_d_sol, n * sizeof(PetscScalar),
                      cudaMemcpyDeviceToHost);
    petscVecRestoreArray(x, xarr);
    GP_CUDA_CHECK(cerr);
  }

  // ---------------------------------------------------------------------
  // p_configure : read cuDSS-specific options and apply them to the config.
  // ---------------------------------------------------------------------
  void p_configure(utility::Configuration::CursorPtr props)
  {
    // Base reads tolerances / MaxIterations / ForceSerial etc.
    LinearSolverImplementation<T, I>::p_configure(props);

    if (props) {
      // Iterative refinement: accept either a step count or a boolean.
      p_irSteps = props->get("iterativeRefinementSteps", p_irSteps);
      if (props->get("iterativeRefinement", false) && p_irSteps <= 0) {
        p_irSteps = 2;
      }
      p_deterministic = props->get("deterministic", p_deterministic);
      p_hybridMemory  = props->get("hybridMemory", p_hybridMemory);

      // Phase-4 constant-factor throughput mode.  refactorEvery=K refactors
      // every K-th solve; constantFactor=true factors once and reuses.
      p_refactorEvery = props->get("refactorEvery", p_refactorEvery);
      if (props->get("constantFactor", false)) p_refactorEvery = 1000000000;
      if (p_refactorEvery < 1) p_refactorEvery = 1;
    }

    // The base p_configure sets p_doSerial from the "ForceSerial" option
    // (default false); re-assert the cuDSS requirement AFTERWARDS so the
    // coefficient matrix reaching p_solveImpl() is always serial (SEQAIJ).
    this->p_doSerial = (this->processor_size() > 1);
    this->p_constSerialMatrix = false;

    p_applyConfig();
  }

  // ---------------------------------------------------------------------
  // p_applyConfig : push tunables into the cuDSS config object.
  // ---------------------------------------------------------------------
  void p_applyConfig(void)
  {
    if (p_irSteps > 0) {
      int steps = p_irSteps;
      GP_CUDSS_CHECK(cudssConfigSet(p_config, CUDSS_CONFIG_IR_N_STEPS,
                                    &steps, sizeof(steps)));
    }
    if (p_deterministic) {
      int det = 1;
      GP_CUDSS_CHECK(cudssConfigSet(p_config, CUDSS_CONFIG_DETERMINISTIC_MODE,
                                    &det, sizeof(det)));
    }
    // NOTE: hybrid (host/unified) memory mode -- optimized for Grace Blackwell
    // -- is intentionally left as a documented tunable; enabling it depends on
    // the cuDSS version's CUDSS_CONFIG_HYBRID_MODE support and is wired up in
    // the Phase-2 batched engine where the 128 GB pool is the dominant lever.
  }

private:

  // non-copyable (owns GPU resources)
  CuDSSLinearSolverImplementation(const CuDSSLinearSolverImplementation&);
  CuDSSLinearSolverImplementation& operator=(const CuDSSLinearSolverImplementation&);
};

} // namespace math
} // namespace gridpack

#endif // GRIDPACK_WITH_CUDSS

#endif
