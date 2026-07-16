/*
 * Copyright (c) 2026 Battelle Memorial Institute
 * Licensed under the modified BSD License.
 */

#include <memory>

#include "linear_solver.hpp"

#include "test_main.cpp"

namespace {

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

  const gridpack::math::LinearSolverStatistics statistics =
    solver.statistics();
  BOOST_CHECK_EQUAL(statistics.backend, "cudss");
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

BOOST_AUTO_TEST_SUITE_END()
