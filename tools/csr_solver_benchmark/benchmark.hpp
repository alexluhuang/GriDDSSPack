/*
 * Copyright (c) 2026 Battelle Memorial Institute
 * Licensed under the modified BSD License.
 */

#ifndef GRIDPACK_CSR_SOLVER_BENCHMARK_BENCHMARK_HPP_
#define GRIDPACK_CSR_SOLVER_BENCHMARK_BENCHMARK_HPP_

#include "csr_system.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace gridpack {
namespace benchmark {

enum class ScenarioSelection {
  All,
  Cold,
  Warm,
  Repeated
};

struct BenchmarkOptions {
  std::int64_t iterations = 10;
  std::int64_t warmups = 1;
  int cuda_device = 0;
  ScenarioSelection scenario = ScenarioSelection::All;
};

struct PhaseSamples {
  std::vector<double> rhs_ms;
  std::vector<double> state_ms;
  std::vector<double> analysis_ms;
  std::vector<double> factor_ms;
  std::vector<double> solve_ms;
  std::vector<double> solution_download_ms;
};

struct ScenarioResult {
  std::string name;
  PhaseSamples samples;
  double one_time_state_ms = 0.0;
  double one_time_analysis_ms = 0.0;
  double one_time_factor_ms = 0.0;
  std::vector<double> solution;
  double scaled_residual = 0.0;
};

struct BackendResult {
  std::string name;
  double setup_ms = 0.0;
  double format_conversion_ms = 0.0;
  std::vector<ScenarioResult> scenarios;
};

BackendResult benchmarkKlu(const CsrSystem &system,
                           const BenchmarkOptions &options);

enum class CudssMode {
  Device,
  Hybrid
};

double initializeCudssRuntime(int cuda_device);
BackendResult benchmarkCudss(const CsrSystem &system,
                             const BenchmarkOptions &options,
                             CudssMode mode);

}  // namespace benchmark
}  // namespace gridpack

#endif
