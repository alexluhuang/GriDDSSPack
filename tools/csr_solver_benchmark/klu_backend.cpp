/*
 * Copyright (c) 2026 Battelle Memorial Institute
 * Licensed under the modified BSD License.
 */

#include "benchmark.hpp"

#include <klu.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gridpack {
namespace benchmark {
namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point begin, Clock::time_point end)
{
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

class KluCommon {
 public:
  KluCommon()
  {
    if (klu_l_defaults(&value_) == 0) {
      throw std::runtime_error("klu_l_defaults failed");
    }
  }

  klu_l_common *get() noexcept { return &value_; }

 private:
  klu_l_common value_{};
};

class KluSymbolic {
 public:
  KluSymbolic(klu_l_symbolic *value, klu_l_common *common)
      : value_(value), common_(common)
  {
    if (value_ == nullptr) {
      throw std::runtime_error("klu_l_analyze failed with status " +
                               std::to_string(common_->status));
    }
  }

  ~KluSymbolic() noexcept
  {
    if (value_ != nullptr) {
      klu_l_free_symbolic(&value_, common_);
    }
  }

  KluSymbolic(const KluSymbolic &) = delete;
  KluSymbolic &operator=(const KluSymbolic &) = delete;

  klu_l_symbolic *get() const noexcept { return value_; }

 private:
  klu_l_symbolic *value_ = nullptr;
  klu_l_common *common_ = nullptr;
};

class KluNumeric {
 public:
  KluNumeric(klu_l_numeric *value, klu_l_common *common)
      : value_(value), common_(common)
  {
    if (value_ == nullptr) {
      throw std::runtime_error("klu_l_factor failed with status " +
                               std::to_string(common_->status));
    }
  }

  ~KluNumeric() noexcept
  {
    if (value_ != nullptr) {
      klu_l_free_numeric(&value_, common_);
    }
  }

  KluNumeric(const KluNumeric &) = delete;
  KluNumeric &operator=(const KluNumeric &) = delete;

  klu_l_numeric *get() const noexcept { return value_; }

 private:
  klu_l_numeric *value_ = nullptr;
  klu_l_common *common_ = nullptr;
};

std::unique_ptr<KluSymbolic> analyze(const CsrSystem &system,
                                     CscMatrix &matrix, KluCommon &common)
{
  return std::make_unique<KluSymbolic>(
      klu_l_analyze(system.ncols, matrix.column_offsets.data(),
                    matrix.row_indices.data(), common.get()),
      common.get());
}

std::unique_ptr<KluNumeric> factor(CscMatrix &matrix,
                                  const KluSymbolic &symbolic,
                                  KluCommon &common)
{
  return std::make_unique<KluNumeric>(
      klu_l_factor(matrix.column_offsets.data(), matrix.row_indices.data(),
                   matrix.values.data(), symbolic.get(), common.get()),
      common.get());
}

void solve(const CsrSystem &system, const KluSymbolic &symbolic,
           const KluNumeric &numeric, std::vector<double> &solution,
           KluCommon &common)
{
  if (klu_l_solve(symbolic.get(), numeric.get(), system.nrows,
                  system.rhs_count, solution.data(), common.get()) == 0) {
    throw std::runtime_error("klu_l_solve failed with status " +
                             std::to_string(common.get()->status));
  }
}

double refreshRightHandSide(const CsrSystem &system,
                            std::vector<double> &solution)
{
  const auto begin = Clock::now();
  std::copy(system.rhs.begin(), system.rhs.end(), solution.begin());
  return milliseconds(begin, Clock::now());
}

bool isMeasured(std::int64_t iteration, const BenchmarkOptions &options)
{
  return iteration >= options.warmups;
}

ScenarioResult benchmarkCold(const CsrSystem &system, CscMatrix &matrix,
                             const BenchmarkOptions &options)
{
  ScenarioResult result;
  result.name = "cold_analysis_factor_solve";
  std::vector<double> solution(system.rhs.size());
  KluCommon common;

  const std::int64_t count = options.warmups + options.iterations;
  for (std::int64_t iteration = 0; iteration < count; ++iteration) {
    const double rhs_ms = refreshRightHandSide(system, solution);

    const auto analysis_begin = Clock::now();
    std::unique_ptr<KluSymbolic> symbolic =
        analyze(system, matrix, common);
    const double analysis_ms =
        milliseconds(analysis_begin, Clock::now());

    const auto factor_begin = Clock::now();
    std::unique_ptr<KluNumeric> numeric =
        factor(matrix, *symbolic, common);
    const double factor_ms = milliseconds(factor_begin, Clock::now());

    const auto solve_begin = Clock::now();
    solve(system, *symbolic, *numeric, solution, common);
    const double solve_ms = milliseconds(solve_begin, Clock::now());

    const auto destroy_begin = Clock::now();
    numeric.reset();
    symbolic.reset();
    const double state_ms = milliseconds(destroy_begin, Clock::now());

    if (isMeasured(iteration, options)) {
      result.samples.rhs_ms.push_back(rhs_ms);
      result.samples.state_ms.push_back(state_ms);
      result.samples.analysis_ms.push_back(analysis_ms);
      result.samples.factor_ms.push_back(factor_ms);
      result.samples.solve_ms.push_back(solve_ms);
      result.solution = solution;
    }
  }
  result.scaled_residual = scaledResidual(system, result.solution);
  return result;
}

ScenarioResult benchmarkWarm(const CsrSystem &system, CscMatrix &matrix,
                             const BenchmarkOptions &options)
{
  ScenarioResult result;
  result.name = "warm_factor_solve";
  std::vector<double> solution(system.rhs.size());
  KluCommon common;

  const auto analysis_begin = Clock::now();
  std::unique_ptr<KluSymbolic> symbolic =
      analyze(system, matrix, common);
  result.one_time_analysis_ms =
      milliseconds(analysis_begin, Clock::now());

  const std::int64_t count = options.warmups + options.iterations;
  for (std::int64_t iteration = 0; iteration < count; ++iteration) {
    const double rhs_ms = refreshRightHandSide(system, solution);

    const auto factor_begin = Clock::now();
    std::unique_ptr<KluNumeric> numeric =
        factor(matrix, *symbolic, common);
    const double factor_ms = milliseconds(factor_begin, Clock::now());

    const auto solve_begin = Clock::now();
    solve(system, *symbolic, *numeric, solution, common);
    const double solve_ms = milliseconds(solve_begin, Clock::now());

    const auto destroy_begin = Clock::now();
    numeric.reset();
    const double state_ms = milliseconds(destroy_begin, Clock::now());

    if (isMeasured(iteration, options)) {
      result.samples.rhs_ms.push_back(rhs_ms);
      result.samples.state_ms.push_back(state_ms);
      result.samples.factor_ms.push_back(factor_ms);
      result.samples.solve_ms.push_back(solve_ms);
      result.solution = solution;
    }
  }
  const auto destroy_begin = Clock::now();
  symbolic.reset();
  result.one_time_state_ms =
      milliseconds(destroy_begin, Clock::now());
  result.scaled_residual = scaledResidual(system, result.solution);
  return result;
}

ScenarioResult benchmarkRepeatedSolve(const CsrSystem &system,
                                      CscMatrix &matrix,
                                      const BenchmarkOptions &options)
{
  ScenarioResult result;
  result.name = "repeated_rhs_solve";
  std::vector<double> solution(system.rhs.size());
  KluCommon common;

  const auto analysis_begin = Clock::now();
  std::unique_ptr<KluSymbolic> symbolic =
      analyze(system, matrix, common);
  result.one_time_analysis_ms =
      milliseconds(analysis_begin, Clock::now());

  const auto factor_begin = Clock::now();
  std::unique_ptr<KluNumeric> numeric =
      factor(matrix, *symbolic, common);
  result.one_time_factor_ms = milliseconds(factor_begin, Clock::now());

  const std::int64_t count = options.warmups + options.iterations;
  for (std::int64_t iteration = 0; iteration < count; ++iteration) {
    const double rhs_ms = refreshRightHandSide(system, solution);

    const auto solve_begin = Clock::now();
    solve(system, *symbolic, *numeric, solution, common);
    const double solve_ms = milliseconds(solve_begin, Clock::now());

    if (isMeasured(iteration, options)) {
      result.samples.rhs_ms.push_back(rhs_ms);
      result.samples.solve_ms.push_back(solve_ms);
      result.solution = solution;
    }
  }
  const auto destroy_begin = Clock::now();
  numeric.reset();
  symbolic.reset();
  result.one_time_state_ms =
      milliseconds(destroy_begin, Clock::now());
  result.scaled_residual = scaledResidual(system, result.solution);
  return result;
}

}  // namespace

BackendResult benchmarkKlu(const CsrSystem &system,
                           const BenchmarkOptions &options)
{
  BackendResult result;
  result.name = "klu";

  const auto conversion_begin = Clock::now();
  CscMatrix matrix = convertCsrToCsc(system);
  result.format_conversion_ms =
      milliseconds(conversion_begin, Clock::now());

  if (options.scenario == ScenarioSelection::All ||
      options.scenario == ScenarioSelection::Cold) {
    result.scenarios.push_back(benchmarkCold(system, matrix, options));
  }
  if (options.scenario == ScenarioSelection::All ||
      options.scenario == ScenarioSelection::Warm) {
    result.scenarios.push_back(benchmarkWarm(system, matrix, options));
  }
  if (options.scenario == ScenarioSelection::All ||
      options.scenario == ScenarioSelection::Repeated) {
    result.scenarios.push_back(
        benchmarkRepeatedSolve(system, matrix, options));
  }
  return result;
}

}  // namespace benchmark
}  // namespace gridpack
