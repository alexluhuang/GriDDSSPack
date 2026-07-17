/*
 * Copyright (c) 2026 Battelle Memorial Institute
 * Licensed under the modified BSD License.
 */

#ifndef GRIDPACK_MATH_CUDSS_BATCH_SOLVER_HPP_
#define GRIDPACK_MATH_CUDSS_BATCH_SOLVER_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gridpack/math/petsc/petsc_csr_exporter.hpp"

namespace gridpack {
namespace math {

struct CUDSSBatchOptions
{
  CUDSSBatchOptions(void)
    : device(0), batchSize(16), maximumCachedPatterns(16),
      validateResiduals(false), residualTolerance(1.0e-10)
  {}

  int device;
  std::size_t batchSize;
  std::size_t maximumCachedPatterns;
  bool validateResiduals;
  double residualTolerance;
};

/**
 * Non-owning numeric buffers for one system in a uniform batch.
 * Both backing vectors must outlive the CUDSSBatchSolver::solve() call.
 */
struct CUDSSBatchSystemView
{
  CUDSSBatchSystemView(const std::vector<double>& matrixValues,
                       const std::vector<double>& rightHandSideValues)
    : values(&matrixValues), rightHandSide(&rightHandSideValues)
  {}

  const std::vector<double> *values;
  const std::vector<double> *rightHandSide;

  CUDSSBatchSystemView(std::vector<double>&& matrixValues,
                       const std::vector<double>& rightHandSideValues) = delete;
  CUDSSBatchSystemView(const std::vector<double>& matrixValues,
                       std::vector<double>&& rightHandSideValues) = delete;
  CUDSSBatchSystemView(std::vector<double>&& matrixValues,
                       std::vector<double>&& rightHandSideValues) = delete;
};

struct CUDSSBatchStatistics
{
  CUDSSBatchStatistics(void)
    : submittedSystems(0), completedSystems(0), batchExecutions(0),
      analyses(0), factorizations(0), refactorizations(0), solves(0),
      cacheHits(0), cacheMisses(0), cacheEvictions(0),
      structureUploadBytes(0), numericUploadBytes(0),
      solutionDownloadBytes(0)
  {}

  std::uint64_t submittedSystems;
  std::uint64_t completedSystems;
  std::uint64_t batchExecutions;
  std::uint64_t analyses;
  std::uint64_t factorizations;
  std::uint64_t refactorizations;
  std::uint64_t solves;
  std::uint64_t cacheHits;
  std::uint64_t cacheMisses;
  std::uint64_t cacheEvictions;
  std::uint64_t structureUploadBytes;
  std::uint64_t numericUploadBytes;
  std::uint64_t solutionDownloadBytes;
};

/**
 * Device-mode cuDSS uniform-batch executor.
 *
 * Every system submitted in one call must have exactly the same CSR pattern.
 * Symbolic analysis and device CSR structure are retained in a bounded
 * exact-pattern LRU cache. Hybrid execution is deliberately not supported by
 * the cuDSS uniform-batch API.
 */
class CUDSSBatchSolver
{
  public:
    explicit CUDSSBatchSolver(const CUDSSBatchOptions& options);
    ~CUDSSBatchSolver(void);

    bool available(void) const;
    std::string unavailableReason(void) const;

    void solve(const std::vector<RealCsrSystem>& systems,
               std::vector<std::vector<double> >& solutions);

    /**
     * Solve numeric systems sharing one immutable CSR structure without
     * copying that structure into every batch slot.
     */
    void solve(const RealCsrSystem& pattern,
               const std::vector<CUDSSBatchSystemView>& systems,
               std::vector<std::vector<double> >& solutions);

    CUDSSBatchStatistics statistics(void) const;

  private:
    class Impl;
    std::unique_ptr<Impl> p_impl;

    CUDSSBatchSolver(const CUDSSBatchSolver&);
    CUDSSBatchSolver& operator=(const CUDSSBatchSolver&);
};

} // namespace math
} // namespace gridpack

#endif
