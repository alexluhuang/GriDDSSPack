/*
 * Copyright (c) 2026 Battelle Memorial Institute
 * Licensed under the modified BSD License.
 */

#ifndef GRIDPACK_MATH_CUDSS_MPI_BROKER_HPP_
#define GRIDPACK_MATH_CUDSS_MPI_BROKER_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <mpi.h>

#include "gridpack/math/cudss/cudss_batch_solver.hpp"

namespace gridpack {
namespace math {

struct CUDSSBrokerOptions
{
  CUDSSBrokerOptions(void)
    : ownerRank(0), device(0), batchSize(8), minimumGpuBatchSize(8),
      batchWaitMicroseconds(0), maximumRegisteredPatterns(64),
      maximumDevicePatterns(16), validateResiduals(false),
      residualTolerance(1.0e-10), strict(false)
  {}

  int ownerRank;
  int device;
  std::size_t batchSize;
  std::size_t minimumGpuBatchSize;
  std::uint64_t batchWaitMicroseconds;
  std::size_t maximumRegisteredPatterns;
  std::size_t maximumDevicePatterns;
  bool validateResiduals;
  double residualTolerance;
  bool strict;
};

struct CUDSSBrokerStatistics
{
  CUDSSBrokerStatistics(void)
    : registrations(0), solveRequests(0), retryResponses(0),
      fallbackResponses(0), errorResponses(0), fullBatches(0),
      partialBatches(0), completedWorkers(0)
  {}

  std::uint64_t registrations;
  std::uint64_t solveRequests;
  std::uint64_t retryResponses;
  std::uint64_t fallbackResponses;
  std::uint64_t errorResponses;
  std::uint64_t fullBatches;
  std::uint64_t partialBatches;
  std::uint64_t completedWorkers;
  CUDSSBatchStatistics batch;
};

/**
 * Client side of the single-owner cuDSS broker protocol.
 *
 * The communicator is non-owning and must be dedicated to this protocol. It
 * must remain valid until done() has completed and the client is destroyed.
 * Destruction does not send DONE; callers must invoke done() explicitly.
 */
class CUDSSBrokerClient
{
  public:
    CUDSSBrokerClient(MPI_Comm communicator,
                      const CUDSSBrokerOptions& options);
    ~CUDSSBrokerClient(void);

    /**
     * Return true with solution filled when the broker solved the system.
     * Return false when the broker directs this worker to local KLU fallback.
     */
    bool solve(const RealCsrSystem& system, std::uint64_t taskId,
               std::vector<double>& solution);

    /** Send this worker's terminal message after all requests have completed. */
    void done(void);

  private:
    class Impl;
    std::unique_ptr<Impl> p_impl;
    CUDSSBrokerClient(const CUDSSBrokerClient&);
    CUDSSBrokerClient& operator=(const CUDSSBrokerClient&);
};

/**
 * Owner-side event loop. Exactly one rank calls run(). The non-owning
 * communicator must be isolated from all other GridPACK/MPI traffic.
 */
class CUDSSBrokerServer
{
  public:
    CUDSSBrokerServer(MPI_Comm communicator,
                      const CUDSSBrokerOptions& options);
    ~CUDSSBrokerServer(void);

    /** Run until every non-owner rank sends DONE and all queues drain. */
    void run(void);

    CUDSSBrokerStatistics statistics(void) const;

  private:
    class Impl;
    std::unique_ptr<Impl> p_impl;
    CUDSSBrokerServer(const CUDSSBrokerServer&);
    CUDSSBrokerServer& operator=(const CUDSSBrokerServer&);
};

} // namespace math
} // namespace gridpack

#endif
