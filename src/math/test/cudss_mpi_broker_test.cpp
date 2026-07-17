/*
 * Copyright (c) 2026 Battelle Memorial Institute
 * Licensed under the modified BSD License.
 */

#include "cudss/cudss_mpi_broker.hpp"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>

#include "test_main.cpp"

namespace {

gridpack::math::RealCsrSystem makeSystem(double shift)
{
  gridpack::math::RealCsrSystem result;
  result.rows = 3;
  result.columns = 3;
  result.nonzeros = 7;
  result.rightHandSideCount = 1;
  result.rowOffsets = {0, 2, 5, 7};
  result.columnIndices = {0, 1, 0, 1, 2, 1, 2};
  result.values = {
    4.0 + shift, 1.0, 1.0, 3.0 + shift, 1.0,
    1.0, 2.0 + shift
  };
  result.rightHandSides = {1.0, 2.0, 3.0};
  return result;
}

gridpack::math::RealCsrSystem makeSameShapeDifferentPatternSystem(
    double shift)
{
  gridpack::math::RealCsrSystem result = makeSystem(shift);
  result.columnIndices = {0, 2, 0, 1, 2, 1, 2};
  return result;
}

double residual(const gridpack::math::RealCsrSystem& system,
                const std::vector<double>& solution)
{
  double squared = 0.0;
  for (std::size_t row = 0; row < system.rows; ++row) {
    double value = 0.0;
    for (std::uint32_t entry = system.rowOffsets[row];
         entry < system.rowOffsets[row + 1]; ++entry) {
      value += system.values[entry] *
        solution[system.columnIndices[entry]];
    }
    const double difference = value - system.rightHandSides[row];
    squared += difference * difference;
  }
  return std::sqrt(squared);
}

} // anonymous namespace

BOOST_AUTO_TEST_SUITE(CUDSSMpiBrokerTest)

BOOST_AUTO_TEST_CASE(SingleRequestTimesOutToFallbackAndTerminates)
{
  gridpack::parallel::Communicator world;
  if (world.size() < 2) return;

  gridpack::math::CUDSSBrokerOptions options;
  options.ownerRank = 0;
  options.batchSize = 2;
  options.minimumGpuBatchSize = 2;
  options.batchWaitMicroseconds = 50;
  options.strict = false;

  if (world.rank() == options.ownerRank) {
    gridpack::math::CUDSSBrokerServer server(
        static_cast<MPI_Comm>(world), options);
    server.run();
    const gridpack::math::CUDSSBrokerStatistics statistics =
      server.statistics();
    BOOST_CHECK_EQUAL(statistics.solveRequests, 1);
    BOOST_CHECK_EQUAL(statistics.fallbackResponses, 1);
    BOOST_CHECK_EQUAL(statistics.completedWorkers, world.size() - 1);
  } else {
    gridpack::math::CUDSSBrokerClient client(
        static_cast<MPI_Comm>(world), options);
    if (world.rank() == 1) {
      std::vector<double> solution;
      BOOST_CHECK(!client.solve(makeSystem(0.0), 7, solution));
    }
    client.done();
  }
  world.sync();
}

BOOST_AUTO_TEST_CASE(RequestIsSolvedOnGpuOrExplicitlyFallsBack)
{
  gridpack::parallel::Communicator world;
  if (world.size() < 2) return;

  gridpack::math::CUDSSBrokerOptions options;
  options.ownerRank = 0;
  options.batchSize = static_cast<std::size_t>(world.size() - 1);
  options.minimumGpuBatchSize = options.batchSize;
  options.batchWaitMicroseconds = 50;
  options.validateResiduals = true;
  options.strict = false;

  const char *requiredValue =
    std::getenv("GRIDPACK_TEST_REQUIRE_CUDSS_GPU");
  const bool gpuRequired =
    requiredValue != NULL && std::string(requiredValue) == "1";

  if (world.rank() == options.ownerRank) {
    gridpack::math::CUDSSBrokerServer server(
        static_cast<MPI_Comm>(world), options);
    server.run();
    const gridpack::math::CUDSSBrokerStatistics statistics =
      server.statistics();
    BOOST_CHECK_EQUAL(statistics.solveRequests, 2 * (world.size() - 1));
    BOOST_CHECK_EQUAL(statistics.completedWorkers, world.size() - 1);
    BOOST_CHECK_EQUAL(statistics.errorResponses, 0);
    BOOST_CHECK_EQUAL(statistics.registrations, 1);
    if (gpuRequired) {
      BOOST_CHECK_EQUAL(statistics.fallbackResponses, 0);
      BOOST_CHECK_EQUAL(statistics.fullBatches, 2);
      BOOST_CHECK_EQUAL(statistics.partialBatches, 0);
      BOOST_CHECK_EQUAL(statistics.batch.submittedSystems,
                        2 * (world.size() - 1));
      BOOST_CHECK_EQUAL(statistics.batch.completedSystems,
                        2 * (world.size() - 1));
      BOOST_CHECK_EQUAL(statistics.batch.cacheMisses, 1);
      BOOST_CHECK_EQUAL(statistics.batch.cacheHits, 1);
    }
  } else {
    gridpack::math::CUDSSBrokerClient client(
        static_cast<MPI_Comm>(world), options);
    for (int iteration = 0; iteration < 2; ++iteration) {
      const gridpack::math::RealCsrSystem system = makeSystem(
          0.25 * world.rank() + 0.1 * iteration);
      std::vector<double> solution;
      const bool solved = client.solve(
          system, 100 * iteration + world.rank(), solution);
      if (gpuRequired) BOOST_REQUIRE(solved);
      if (solved) {
        BOOST_CHECK_SMALL(residual(system, solution), 1.0e-10);
      }
    }
    client.done();
  }
  world.sync();
}

BOOST_AUTO_TEST_CASE(EqualShapeDifferentPatternsAreNeverCoalesced)
{
  gridpack::parallel::Communicator world;
  if (world.size() != 3) return;

  gridpack::math::CUDSSBrokerOptions options;
  options.ownerRank = 0;
  options.batchSize = 2;
  options.minimumGpuBatchSize = 2;
  options.batchWaitMicroseconds = 50000;
  options.strict = false;

  if (world.rank() == options.ownerRank) {
    gridpack::math::CUDSSBrokerServer server(
        static_cast<MPI_Comm>(world), options);
    server.run();
    const gridpack::math::CUDSSBrokerStatistics statistics =
      server.statistics();
    BOOST_CHECK_EQUAL(statistics.solveRequests, 2);
    BOOST_CHECK_EQUAL(statistics.registrations, 2);
    BOOST_CHECK_EQUAL(statistics.fallbackResponses, 2);
    BOOST_CHECK_EQUAL(statistics.errorResponses, 0);
    BOOST_CHECK_EQUAL(statistics.fullBatches, 0);
    BOOST_CHECK_EQUAL(statistics.partialBatches, 0);
    BOOST_CHECK_EQUAL(statistics.batch.submittedSystems, 0);
    BOOST_CHECK_EQUAL(statistics.completedWorkers, 2);
  } else {
    const gridpack::math::RealCsrSystem system = world.rank() == 1
      ? makeSystem(0.0)
      : makeSameShapeDifferentPatternSystem(0.0);
    BOOST_REQUIRE_EQUAL(system.rows, 3);
    BOOST_REQUIRE_EQUAL(system.nonzeros, 7);

    gridpack::math::CUDSSBrokerClient client(
        static_cast<MPI_Comm>(world), options);
    std::vector<double> solution;
    BOOST_CHECK(!client.solve(system, world.rank(), solution));
    client.done();
  }
  world.sync();
}

BOOST_AUTO_TEST_CASE(StrictUnavailableDeviceReturnsErrorsAndDrains)
{
  gridpack::parallel::Communicator world;
  if (world.size() < 2) return;

  gridpack::math::CUDSSBrokerOptions options;
  options.ownerRank = 0;
  options.device = std::numeric_limits<int>::max();
  options.batchSize = static_cast<std::size_t>(world.size() - 1);
  options.minimumGpuBatchSize = 1;
  options.batchWaitMicroseconds = 50;
  options.strict = true;

  if (world.rank() == options.ownerRank) {
    gridpack::math::CUDSSBrokerServer server(
        static_cast<MPI_Comm>(world), options);
    server.run();
    const gridpack::math::CUDSSBrokerStatistics statistics =
      server.statistics();
    BOOST_CHECK_EQUAL(statistics.solveRequests, world.size() - 1);
    BOOST_CHECK_EQUAL(statistics.errorResponses, world.size() - 1);
    BOOST_CHECK_EQUAL(statistics.completedWorkers, world.size() - 1);
  } else {
    gridpack::math::CUDSSBrokerClient client(
        static_cast<MPI_Comm>(world), options);
    std::vector<double> solution;
    BOOST_CHECK_THROW(client.solve(makeSystem(0.1 * world.rank()),
                                   world.rank(), solution),
                      std::exception);
    client.done();
  }
  world.sync();
}

BOOST_AUTO_TEST_SUITE_END()
