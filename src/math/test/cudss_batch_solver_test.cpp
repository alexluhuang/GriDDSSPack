/*
 * Copyright (c) 2026 Battelle Memorial Institute
 * Licensed under the modified BSD License.
 */

#include "cudss/cudss_batch_solver.hpp"

#include <cmath>
#include <cstdlib>
#include <string>

#include "test_main.cpp"

namespace {

gridpack::math::RealCsrSystem makeSystem(double diagonalShift)
{
  gridpack::math::RealCsrSystem result;
  result.rows = 3;
  result.columns = 3;
  result.nonzeros = 7;
  result.rightHandSideCount = 1;
  result.rowOffsets = {0, 2, 5, 7};
  result.columnIndices = {0, 1, 0, 1, 2, 1, 2};
  result.values = {
    4.0 + diagonalShift, 1.0,
    1.0, 3.0 + diagonalShift, 1.0,
    1.0, 2.0 + diagonalShift
  };
  result.rightHandSides = {1.0, 2.0, 3.0};
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

BOOST_AUTO_TEST_SUITE(CUDSSBatchSolverTest)

BOOST_AUTO_TEST_CASE(UniformBatchReusesAnalysisAndHandlesPartialBatch)
{
  gridpack::math::CUDSSBatchOptions options;
  options.batchSize = 4;
  options.maximumCachedPatterns = 2;
  options.validateResiduals = true;
  gridpack::math::CUDSSBatchSolver solver(options);

  if (!solver.available()) {
    const char *required = std::getenv("GRIDPACK_TEST_REQUIRE_CUDSS_GPU");
    if (required != NULL && std::string(required) == "1") {
      BOOST_FAIL("cuDSS GPU was required but unavailable: " +
                 solver.unavailableReason());
    }
    std::vector<gridpack::math::RealCsrSystem> unavailableSystems(
        1, makeSystem(0.0));
    std::vector<std::vector<double> > unavailableSolutions;
    BOOST_CHECK_THROW(solver.solve(unavailableSystems, unavailableSolutions),
                      std::exception);
    return;
  }

  std::vector<gridpack::math::RealCsrSystem> first;
  first.push_back(makeSystem(0.0));
  first.push_back(makeSystem(0.5));
  first.push_back(makeSystem(1.0));
  std::vector<std::vector<double> > solutions;
  solver.solve(first, solutions);
  BOOST_REQUIRE_EQUAL(solutions.size(), first.size());
  for (std::size_t index = 0; index < first.size(); ++index) {
    BOOST_CHECK_SMALL(residual(first[index], solutions[index]), 1.0e-10);
  }

  std::vector<gridpack::math::RealCsrSystem> second;
  second.push_back(makeSystem(1.5));
  second.push_back(makeSystem(2.0));
  solver.solve(second, solutions);
  BOOST_REQUIRE_EQUAL(solutions.size(), second.size());
  for (std::size_t index = 0; index < second.size(); ++index) {
    BOOST_CHECK_SMALL(residual(second[index], solutions[index]), 1.0e-10);
  }

  const gridpack::math::CUDSSBatchStatistics statistics = solver.statistics();
  BOOST_CHECK_EQUAL(statistics.submittedSystems, 5);
  BOOST_CHECK_EQUAL(statistics.completedSystems, 5);
  BOOST_CHECK_EQUAL(statistics.batchExecutions, 2);
  BOOST_CHECK_EQUAL(statistics.cacheMisses, 1);
  BOOST_CHECK_EQUAL(statistics.cacheHits, 1);
  BOOST_CHECK_EQUAL(statistics.analyses, 1);
  BOOST_CHECK_EQUAL(statistics.factorizations, 1);
  BOOST_CHECK_EQUAL(statistics.refactorizations, 1);
  BOOST_CHECK_EQUAL(statistics.solves, 2);
  BOOST_CHECK(statistics.structureUploadBytes > 0);
  BOOST_CHECK(statistics.numericUploadBytes > 0);
  BOOST_CHECK(statistics.solutionDownloadBytes > 0);
}

BOOST_AUTO_TEST_SUITE_END()
