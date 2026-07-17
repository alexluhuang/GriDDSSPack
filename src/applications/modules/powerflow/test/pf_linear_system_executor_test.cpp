/*
 * Copyright (c) 2026 Battelle Memorial Institute
 * Licensed under the modified BSD License.
 */

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gridpack/configuration/configuration.hpp"
#include "gridpack/environment/environment.hpp"
#include "gridpack/math/math.hpp"
#include "gridpack/parallel/parallel.hpp"
#include "pf_app_module.hpp"

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API
#include <boost/test/included/unit_test.hpp>

namespace {

const char *backendEnvironment = "GRIDPACK_LINEAR_SOLVER_BACKEND";

class ScopedEnvironment
{
  public:
    explicit ScopedEnvironment(const char *name)
      : p_name(name), p_hadValue(false)
    {
      const char *value = std::getenv(p_name.c_str());
      if (value != NULL) {
        p_hadValue = true;
        p_value = value;
      }
      unsetenv(p_name.c_str());
    }

    ~ScopedEnvironment(void)
    {
      if (p_hadValue) {
        setenv(p_name.c_str(), p_value.c_str(), 1);
      } else {
        unsetenv(p_name.c_str());
      }
    }

  private:
    std::string p_name;
    std::string p_value;
    bool p_hadValue;
};

gridpack::utility::Configuration::CursorPtr executorConfiguration(void)
{
  gridpack::utility::Configuration *configuration =
    gridpack::utility::Configuration::configuration();
  gridpack::utility::Configuration::CursorPtr cursor =
    configuration->getCursor("Configuration.Powerflow.Executor");
  BOOST_REQUIRE(cursor);
  return cursor;
}

void solveWithPetsc(gridpack::powerflow::PFLinearSystem& system)
{
  std::unique_ptr<gridpack::math::RealMatrix> matrix(system.matrix.clone());
  gridpack::math::RealLinearSolver solver(*matrix);
  solver.configure(executorConfiguration());
  solver.solve(system.rightHandSide, system.solution);
}

class RecordingExecutor : public gridpack::powerflow::PFLinearSystemExecutor
{
  public:
    enum Behavior {
      Handle,
      RequestFallback,
      ThrowOnSecondCall
    };

    explicit RecordingExecutor(Behavior behavior)
      : p_behavior(behavior), p_calls(0)
    {}

    bool solve(gridpack::powerflow::PFLinearSystem& system)
    {
      ++p_calls;
      p_newtonIterations.push_back(system.newtonIteration);
      p_ordinals.push_back(system.linearSolveOrdinal);

      if (p_behavior == RequestFallback) {
        if (p_calls == 1) {
          // The application-level backend is deliberately invalid. Setting
          // this here proves fallback solver configuration happens only after
          // the executor declines the first system.
          setenv(backendEnvironment, "petsc", 1);
        }
        return false;
      }

      if (p_behavior == ThrowOnSecondCall && p_calls == 2) {
        throw std::runtime_error("executor failure on second Newton solve");
      }

      solveWithPetsc(system);
      return true;
    }

    int calls(void) const
    {
      return p_calls;
    }

    const std::vector<int>& newtonIterations(void) const
    {
      return p_newtonIterations;
    }

    const std::vector<int>& ordinals(void) const
    {
      return p_ordinals;
    }

  private:
    Behavior p_behavior;
    int p_calls;
    std::vector<int> p_newtonIterations;
    std::vector<int> p_ordinals;
};

class PowerflowCase
{
  public:
    PowerflowCase(void)
      : world(), network(new gridpack::powerflow::PFNetwork(world))
    {
      app.suppressOutput(true);
      gridpack::utility::Configuration *configuration =
        gridpack::utility::Configuration::configuration();
      app.readNetwork(network, configuration);
      app.initialize();
    }

    gridpack::parallel::Communicator world;
    boost::shared_ptr<gridpack::powerflow::PFNetwork> network;
    gridpack::powerflow::PFAppModule app;
};

void checkBothNewtonSolveSites(const RecordingExecutor& executor)
{
  BOOST_REQUIRE_GE(executor.calls(), 2);
  BOOST_REQUIRE_GE(executor.newtonIterations().size(), 2U);
  BOOST_CHECK_EQUAL(executor.newtonIterations()[0], 1);
  BOOST_CHECK_EQUAL(executor.newtonIterations()[1], 2);
  for (size_t i = 0; i < executor.ordinals().size(); ++i) {
    BOOST_CHECK_EQUAL(executor.ordinals()[i], static_cast<int>(i + 1));
  }
}

bool hasExecutorMessage(const std::runtime_error& error)
{
  return std::string(error.what()) ==
    "executor failure on second Newton solve";
}

} // anonymous namespace

BOOST_AUTO_TEST_SUITE(PFLinearSystemExecutorTest)

BOOST_AUTO_TEST_CASE(HandledSystemsBypassConfiguredLocalSolver)
{
  ScopedEnvironment environment(backendEnvironment);
  PowerflowCase powerflow;
  BOOST_REQUIRE_EQUAL(powerflow.world.size(), 1);
  RecordingExecutor executor(RecordingExecutor::Handle);
  powerflow.app.setLinearSystemExecutor(&executor);

  BOOST_REQUIRE(powerflow.app.solve());
  BOOST_CHECK(powerflow.app.getConvergence().converged);
  checkBothNewtonSolveSites(executor);

  const gridpack::powerflow::PFSolveMetrics metrics =
    powerflow.app.getSolveMetrics();
  BOOST_CHECK_EQUAL(metrics.linearSolveCalls, executor.calls());
  BOOST_CHECK_GE(metrics.completedNewtonUpdates, 1);
}

BOOST_AUTO_TEST_CASE(FalseReturnLazilyUsesPetscKluFallback)
{
  ScopedEnvironment environment(backendEnvironment);
  PowerflowCase powerflow;
  BOOST_REQUIRE_EQUAL(powerflow.world.size(), 1);
  RecordingExecutor executor(RecordingExecutor::RequestFallback);
  powerflow.app.setLinearSystemExecutor(&executor);

  BOOST_REQUIRE(powerflow.app.solve());
  BOOST_CHECK(powerflow.app.getConvergence().converged);
  checkBothNewtonSolveSites(executor);
  BOOST_CHECK_EQUAL(std::string(std::getenv(backendEnvironment)), "petsc");

  const gridpack::powerflow::PFSolveMetrics metrics =
    powerflow.app.getSolveMetrics();
  BOOST_CHECK_EQUAL(metrics.linearSolveCalls, executor.calls());
  BOOST_CHECK_GE(metrics.completedNewtonUpdates, 1);
}

BOOST_AUTO_TEST_CASE(ExceptionFromSecondNewtonSolvePropagates)
{
  ScopedEnvironment environment(backendEnvironment);
  PowerflowCase powerflow;
  BOOST_REQUIRE_EQUAL(powerflow.world.size(), 1);
  RecordingExecutor executor(RecordingExecutor::ThrowOnSecondCall);
  powerflow.app.setLinearSystemExecutor(&executor);

  BOOST_CHECK_EXCEPTION(powerflow.app.solve(), std::runtime_error,
                        hasExecutorMessage);
  BOOST_REQUIRE_EQUAL(executor.calls(), 2);
  BOOST_REQUIRE_EQUAL(executor.newtonIterations().size(), 2U);
  BOOST_CHECK_EQUAL(executor.newtonIterations()[0], 1);
  BOOST_CHECK_EQUAL(executor.newtonIterations()[1], 2);
  BOOST_CHECK_EQUAL(powerflow.app.getSolveMetrics().linearSolveCalls, 2);
  BOOST_CHECK_EQUAL(powerflow.app.getSolveMetrics().completedNewtonUpdates, 1);
}

BOOST_AUTO_TEST_SUITE_END()

bool init_function(void)
{
  return true;
}

int main(int argc, char **argv)
{
  gridpack::Environment environment(argc, argv);
  gridpack::parallel::Communicator world;

  gridpack::utility::Configuration *configuration =
    gridpack::utility::Configuration::configuration();
  const bool opened = configuration->open(
    "pf_linear_system_executor_test.xml", world);
  if (!opened) {
    if (world.rank() == 0) {
      std::cout << "failure detected" << std::endl;
    }
    return 1;
  }

  int localResult = ::boost::unit_test::unit_test_main(
    &init_function, argc, argv);
  localResult = localResult == boost::exit_success ? 0 : 1;
  int globalResult = 0;
  boost::mpi::all_reduce(world, localResult, globalResult,
                         std::plus<int>());
  if (world.rank() == 0) {
    if (globalResult == 0) {
      std::cout << "No errors detected" << std::endl;
    } else {
      std::cout << "failure detected" << std::endl;
    }
  }
  return globalResult;
}
