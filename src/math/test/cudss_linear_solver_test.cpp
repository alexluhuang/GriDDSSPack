/*
 * Copyright (c) 2026 Battelle Memorial Institute
 * Licensed under the modified BSD License.
 */

#include <cstdlib>
#include <memory>
#include <string>

#include "linear_solver.hpp"

#include "test_main.cpp"

namespace {

bool gpuRequired(void)
{
  const char *value = std::getenv("GRIDPACK_TEST_REQUIRE_CUDSS_GPU");
  return value != NULL && std::string(value) == "1";
}

void assembleSystem(gridpack::math::RealMatrix& matrix,
                    gridpack::math::RealVector& rhs)
{
  matrix.setElement(0, 0, 4.0);
  matrix.setElement(0, 1, 1.0);
  matrix.setElement(1, 0, 1.0);
  matrix.setElement(1, 1, 3.0);
  matrix.setElement(1, 2, 1.0);
  matrix.setElement(2, 1, 1.0);
  matrix.setElement(2, 2, 2.0);
  rhs.setElement(0, 1.0);
  rhs.setElement(1, 2.0);
  rhs.setElement(2, 3.0);
  matrix.ready();
  rhs.ready();
}

void checkResidual(const gridpack::math::RealMatrix& matrix,
                   const gridpack::math::RealVector& rhs,
                   const gridpack::math::RealVector& solution)
{
  std::unique_ptr<gridpack::math::RealVector> residual(
    multiply(matrix, solution));
  residual->add(rhs, -1.0);
  BOOST_CHECK_SMALL(residual->norm2(), 1.0e-10);
}

gridpack::utility::Configuration::CursorPtr
configuration(const std::string& name)
{
  BOOST_REQUIRE(test_config);
  gridpack::utility::Configuration::CursorPtr result =
    test_config->getCursor(name);
  BOOST_REQUIRE(result);
  return result;
}

gridpack::math::LinearSolverStatistics
solvePatternVariant(const gridpack::parallel::Communicator& world,
                    int offDiagonalColumn)
{
  gridpack::math::RealMatrix matrix(
    world, 4, 4, gridpack::math::Sparse);
  gridpack::math::RealVector rhs(world, 4);
  gridpack::math::RealVector solution(world, 4);
  for (int row = 0; row < 4; ++row) {
    matrix.setElement(row, row, 4.0 + row);
    rhs.setElement(row, 1.0 + row);
  }
  matrix.setElement(0, offDiagonalColumn, 0.25);
  matrix.ready();
  rhs.ready();
  solution.fill(0.0);
  solution.ready();

  gridpack::math::RealLinearSolver solver(matrix);
  solver.configure(configuration("CUDSSDeviceLRUValidation"));
  solver.solve(rhs, solution);
  checkResidual(matrix, rhs, solution);
  return solver.statistics();
}

} // anonymous namespace

BOOST_AUTO_TEST_SUITE(CUDSSLinearSolverTest)

BOOST_AUTO_TEST_CASE(DeviceModeCachesExactPattern)
{
  gridpack::parallel::Communicator world;
  BOOST_REQUIRE_EQUAL(world.size(), 1);

  gridpack::math::RealMatrix matrix(
    world, 3, 3, gridpack::math::Sparse);
  gridpack::math::RealVector rhs(world, 3);
  gridpack::math::RealVector solution(world, 3);
  assembleSystem(matrix, rhs);
  solution.fill(0.0);
  solution.ready();

  gridpack::math::LinearSolverStatistics statistics;
  {
    gridpack::math::RealLinearSolver solver(matrix);
    solver.configure(configuration("CUDSSDeviceValidation"));
    solver.solve(rhs, solution);
    checkResidual(matrix, rhs, solution);

    solution.fill(0.0);
    solution.ready();
    solver.resolve(rhs, solution);
    checkResidual(matrix, rhs, solution);

    matrix.setElement(0, 0, 5.0);
    matrix.ready();
    solution.fill(0.0);
    solution.ready();
    solver.solve(rhs, solution);
    checkResidual(matrix, rhs, solution);
    statistics = solver.statistics();
  }

  BOOST_CHECK_EQUAL(statistics.backend, "cudss");
  if (gpuRequired()) BOOST_REQUIRE(statistics.deviceOwner);
  if (statistics.deviceOwner) {
    BOOST_CHECK(statistics.cudssCompiled);
    BOOST_CHECK_EQUAL(statistics.ownerWorldRank, 0);
    BOOST_CHECK_EQUAL(statistics.cacheMisses, 1);
    BOOST_CHECK_EQUAL(statistics.cacheHits, 2);
    BOOST_CHECK_EQUAL(statistics.analyses, 1);
    BOOST_CHECK_EQUAL(statistics.factorizations, 1);
    BOOST_CHECK_EQUAL(statistics.refactorizations, 1);
    BOOST_CHECK_EQUAL(statistics.solves, 3);
    BOOST_CHECK_EQUAL(statistics.fallbacks, 0);
    BOOST_CHECK_SMALL(statistics.lastScaledResidual, 1.0e-10);
  } else {
    BOOST_CHECK_EQUAL(statistics.fallbacks, 3);
  }

  solution.fill(0.0);
  solution.ready();
  gridpack::math::RealLinearSolver secondSolver(matrix);
  secondSolver.configure(configuration("CUDSSDeviceValidation"));
  secondSolver.solve(rhs, solution);
  checkResidual(matrix, rhs, solution);

  const gridpack::math::LinearSolverStatistics secondStatistics =
    secondSolver.statistics();
  if (gpuRequired()) BOOST_REQUIRE(secondStatistics.deviceOwner);
  if (statistics.deviceOwner) {
    BOOST_REQUIRE(secondStatistics.deviceOwner);
    BOOST_CHECK_EQUAL(secondStatistics.cacheMisses, 0);
    BOOST_CHECK_EQUAL(secondStatistics.cacheHits, 1);
    BOOST_CHECK_EQUAL(secondStatistics.analyses, 0);
    BOOST_CHECK_EQUAL(secondStatistics.factorizations, 0);
    BOOST_CHECK_EQUAL(secondStatistics.refactorizations, 0);
    BOOST_CHECK_EQUAL(secondStatistics.solves, 1);
    BOOST_CHECK_EQUAL(secondStatistics.fallbacks, 0);
  } else {
    BOOST_CHECK_EQUAL(secondStatistics.fallbacks, 1);
  }
}

BOOST_AUTO_TEST_CASE(HybridModeSolvesOrFallsBack)
{
  gridpack::parallel::Communicator world;
  BOOST_REQUIRE_EQUAL(world.size(), 1);

  gridpack::math::RealMatrix matrix(
    world, 3, 3, gridpack::math::Sparse);
  gridpack::math::RealVector rhs(world, 3);
  gridpack::math::RealVector solution(world, 3);
  assembleSystem(matrix, rhs);
  solution.fill(0.0);
  solution.ready();

  gridpack::math::RealLinearSolver solver(matrix);
  solver.configure(configuration("CUDSSHybridValidation"));
  solver.solve(rhs, solution);
  checkResidual(matrix, rhs, solution);

  const gridpack::math::LinearSolverStatistics statistics =
    solver.statistics();
  BOOST_CHECK_EQUAL(statistics.backend, "cudss");
  if (statistics.deviceOwner) {
    BOOST_CHECK_EQUAL(statistics.analyses, 1);
    BOOST_CHECK_EQUAL(statistics.factorizations, 1);
    BOOST_CHECK_EQUAL(statistics.solves, 1);
    BOOST_CHECK_EQUAL(statistics.fallbacks, 0);
  } else {
    BOOST_CHECK_EQUAL(statistics.fallbacks, 1);
  }
}

BOOST_AUTO_TEST_CASE(StrictModeNeverFallsBack)
{
  gridpack::parallel::Communicator world;
  BOOST_REQUIRE_EQUAL(world.size(), 1);

  gridpack::math::RealMatrix matrix(
    world, 3, 3, gridpack::math::Sparse);
  gridpack::math::RealVector rhs(world, 3);
  gridpack::math::RealVector solution(world, 3);
  assembleSystem(matrix, rhs);
  solution.fill(0.0);
  solution.ready();

  bool failed = false;
  try {
    gridpack::math::RealLinearSolver solver(matrix);
    solver.configure(configuration("CUDSSStrictValidation"));
    solver.solve(rhs, solution);
  } catch (const std::exception&) {
    failed = true;
  }
  BOOST_CHECK(failed);
}

BOOST_AUTO_TEST_CASE(DeviceModeCacheIsBoundedLRU)
{
  gridpack::parallel::Communicator world;
  BOOST_REQUIRE_EQUAL(world.size(), 1);

  const gridpack::math::LinearSolverStatistics first =
    solvePatternVariant(world, 1);
  const gridpack::math::LinearSolverStatistics second =
    solvePatternVariant(world, 2);
  const gridpack::math::LinearSolverStatistics firstAgain =
    solvePatternVariant(world, 1);
  const gridpack::math::LinearSolverStatistics third =
    solvePatternVariant(world, 3);
  const gridpack::math::LinearSolverStatistics secondAgain =
    solvePatternVariant(world, 2);

  if (gpuRequired()) BOOST_REQUIRE(first.deviceOwner);
  if (first.deviceOwner) {
    BOOST_REQUIRE(second.deviceOwner);
    BOOST_REQUIRE(firstAgain.deviceOwner);
    BOOST_REQUIRE(third.deviceOwner);
    BOOST_REQUIRE(secondAgain.deviceOwner);
    BOOST_CHECK_EQUAL(first.cacheMisses, 1);
    BOOST_CHECK_EQUAL(second.cacheMisses, 1);
    BOOST_CHECK_EQUAL(firstAgain.cacheHits, 1);
    BOOST_CHECK_EQUAL(firstAgain.analyses, 0);
    BOOST_CHECK_EQUAL(third.cacheMisses, 1);
    BOOST_CHECK_EQUAL(secondAgain.cacheMisses, 1);
    BOOST_CHECK_EQUAL(secondAgain.analyses, 1);
  } else {
    BOOST_CHECK_EQUAL(first.fallbacks, 1);
    BOOST_CHECK_EQUAL(second.fallbacks, 1);
    BOOST_CHECK_EQUAL(firstAgain.fallbacks, 1);
    BOOST_CHECK_EQUAL(third.fallbacks, 1);
    BOOST_CHECK_EQUAL(secondAgain.fallbacks, 1);
  }
}

BOOST_AUTO_TEST_SUITE_END()
