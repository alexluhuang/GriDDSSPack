/*
 * Copyright (c) 2026 Battelle Memorial Institute
 * Licensed under the modified BSD License.
 */

#ifndef GRIDPACK_CSR_SOLVER_BENCHMARK_CSR_SYSTEM_HPP_
#define GRIDPACK_CSR_SOLVER_BENCHMARK_CSR_SYSTEM_HPP_

#include <cstdint>
#include <string>
#include <vector>

namespace gridpack {
namespace benchmark {

struct CsrSystem {
  std::string name;
  std::string contingency;
  std::int64_t newton_iteration = 0;
  std::int64_t controller_iteration = 0;
  std::string convergence_status;
  std::string state;
  std::int64_t nrows = 0;
  std::int64_t ncols = 0;
  std::int64_t nnz = 0;
  std::int64_t rhs_count = 0;
  std::vector<std::int64_t> row_offsets;
  std::vector<std::int64_t> column_indices;
  std::vector<double> values;
  std::vector<double> rhs;
};

struct CscMatrix {
  std::vector<std::int64_t> column_offsets;
  std::vector<std::int64_t> row_indices;
  std::vector<double> values;
};

struct SolutionDifference {
  double max_relative_l2 = 0.0;
  double max_absolute = 0.0;
};

CsrSystem readCsrSystem(const std::string &path);
CscMatrix convertCsrToCsc(const CsrSystem &system);
double scaledResidual(const CsrSystem &system,
                      const std::vector<double> &solution);
SolutionDifference compareSolutions(const CsrSystem &system,
                                    const std::vector<double> &reference,
                                    const std::vector<double> &candidate);

}  // namespace benchmark
}  // namespace gridpack

#endif
