/*
 * Copyright (c) 2026 Battelle Memorial Institute
 * Licensed under the modified BSD License.
 */

#include <set>
#include <string>
#include <vector>

#include "ca_pattern_scheduler.hpp"
#include "gridpack/utilities/exception.hpp"

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API
#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CAPatternSchedulerTest)

BOOST_AUTO_TEST_CASE(GroupsPatternsWithoutChangingTaskIdentity)
{
  std::vector<std::string> keys;
  keys.push_back("v1:layout:base");
  keys.push_back("v1:layout:pq:7");
  keys.push_back("v1:layout:base");
  keys.push_back("v1:layout:base");
  keys.push_back("v1:layout:pq:7");
  keys.push_back("v1:no-solve:island");

  const std::vector<gridpack::contingency_analysis::PatternScheduleEpoch>
    epochs = gridpack::contingency_analysis::buildPatternSchedule(keys, 2);
  BOOST_REQUIRE_EQUAL(epochs.size(), 4U);
  BOOST_REQUIRE_EQUAL(epochs[0].taskIds.size(), 2U);
  BOOST_CHECK_EQUAL(epochs[0].taskIds[0], 0);
  BOOST_CHECK_EQUAL(epochs[0].taskIds[1], 2);
  BOOST_REQUIRE_EQUAL(epochs[1].taskIds.size(), 1U);
  BOOST_CHECK_EQUAL(epochs[1].taskIds[0], 3);
  BOOST_REQUIRE_EQUAL(epochs[2].taskIds.size(), 2U);
  BOOST_CHECK_EQUAL(epochs[2].taskIds[0], 1);
  BOOST_CHECK_EQUAL(epochs[2].taskIds[1], 4);
  BOOST_REQUIRE_EQUAL(epochs[3].taskIds.size(), 1U);
  BOOST_CHECK_EQUAL(epochs[3].taskIds[0], 5);

  std::set<int> scheduled;
  for (std::size_t epoch = 0; epoch < epochs.size(); ++epoch) {
    BOOST_CHECK_LE(epochs[epoch].taskIds.size(), 2U);
    for (std::size_t task = 0; task < epochs[epoch].taskIds.size(); ++task) {
      const int taskId = epochs[epoch].taskIds[task];
      BOOST_CHECK_EQUAL(epochs[epoch].expectedPatternClass, keys[taskId]);
      BOOST_CHECK(scheduled.insert(taskId).second);
    }
  }
  BOOST_CHECK_EQUAL(scheduled.size(), keys.size());
}

BOOST_AUTO_TEST_CASE(SplitsWorkerPoolIntoBatchAlignedEpochs)
{
  const std::vector<std::string> keys(19, "v1:layout:base");
  const std::vector<gridpack::contingency_analysis::PatternScheduleEpoch>
    epochs = gridpack::contingency_analysis::buildPatternSchedule(keys, 16);
  BOOST_REQUIRE_EQUAL(epochs.size(), 2U);
  BOOST_CHECK_EQUAL(epochs[0].taskIds.size(), 16U);
  BOOST_CHECK_EQUAL(epochs[1].taskIds.size(), 3U);
  for (int task = 0; task < 19; ++task) {
    const std::size_t epoch = task < 16 ? 0 : 1;
    const std::size_t offset = task < 16 ? task : task - 16;
    BOOST_CHECK_EQUAL(epochs[epoch].taskIds[offset], task);
  }
}

BOOST_AUTO_TEST_CASE(RejectsZeroEpochSize)
{
  std::vector<std::string> keys(1, "v1:layout:base");
  BOOST_CHECK_THROW(
      gridpack::contingency_analysis::buildPatternSchedule(keys, 0),
      gridpack::Exception);
}

BOOST_AUTO_TEST_SUITE_END()

bool init_function(void)
{
  return true;
}

int main(int argc, char **argv)
{
  return boost::unit_test::unit_test_main(&init_function, argc, argv);
}
