/*
 * Copyright (c) 2026 Battelle Memorial Institute
 * Licensed under the modified BSD License.
 */

#ifndef GRIDPACK_CA_PATTERN_SCHEDULER_HPP_
#define GRIDPACK_CA_PATTERN_SCHEDULER_HPP_

#include <cstddef>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "gridpack/applications/modules/powerflow/pf_app_module.hpp"

namespace gridpack {
namespace contingency_analysis {

struct PatternScheduleEpoch
{
  std::string expectedPatternClass;
  std::vector<int> taskIds;
};

class ExpectedJacobianPatternClassifier
{
  public:
    explicit ExpectedJacobianPatternClassifier(
        const boost::shared_ptr<gridpack::powerflow::PFNetwork>& network);

    std::string classify(
        const gridpack::powerflow::Contingency& contingency) const;

  private:
    struct Impl;
    boost::shared_ptr<Impl> p_impl;
};

/**
 * Predict the initial Jacobian layout affected by a contingency. The
 * classifier derives its result from the loaded network rather than case
 * names, bus-number ranges, or a particular network size.
 */
std::string expectedJacobianPatternClass(
    const boost::shared_ptr<gridpack::powerflow::PFNetwork>& network,
    const gridpack::powerflow::Contingency& contingency);

/**
 * Stably group task IDs by predicted pattern and split each group into
 * synchronization epochs. No epoch contains more than epochSize tasks or
 * more than one predicted pattern class.
 */
std::vector<PatternScheduleEpoch> buildPatternSchedule(
    const std::vector<std::string>& expectedPatternClasses,
    std::size_t epochSize);

} // namespace contingency_analysis
} // namespace gridpack

#endif
