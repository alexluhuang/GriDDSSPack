/*
 * Copyright (c) 2026 Battelle Memorial Institute
 * Licensed under the modified BSD License.
 */

#include "benchmark.hpp"
#include "csr_system.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using gridpack::benchmark::BackendResult;
using gridpack::benchmark::BenchmarkOptions;
using gridpack::benchmark::CsrSystem;
using gridpack::benchmark::CudssMode;
using gridpack::benchmark::PhaseSamples;
using gridpack::benchmark::ScenarioSelection;
using gridpack::benchmark::ScenarioResult;

enum class BackendSelection {
  All,
  Klu,
  Cudss
};

enum class CudssSelection {
  Both,
  Device,
  Hybrid
};

struct CommandLine {
  BenchmarkOptions benchmark;
  BackendSelection backend = BackendSelection::All;
  CudssSelection cudss = CudssSelection::Both;
  double residual_tolerance = 1.0e-10;
  double agreement_tolerance = 1.0e-9;
  std::string input_path;
};

struct Summary {
  std::size_t count = 0;
  double minimum = 0.0;
  double median = 0.0;
  double mean = 0.0;
  double maximum = 0.0;
};

void usage(const char *program)
{
  std::cout
      << "Usage: " << program << " [options] SYSTEM.gpcsr\n"
      << "Options:\n"
      << "  --backend all|klu|cudss       Backends to run (default: all)\n"
      << "  --cudss-mode both|device|hybrid\n"
      << "                                  cuDSS execution mode (default: both)\n"
      << "  --iterations N                  Measured repetitions (default: 10)\n"
      << "  --warmups N                     Unmeasured repetitions (default: 1)\n"
      << "  --scenario all|cold|warm|repeated\n"
      << "                                  Solver lifecycle (default: all)\n"
      << "  --device N                      CUDA device index (default: 0)\n"
      << "  --residual-tol VALUE            Scaled residual limit (default: 1e-10)\n"
      << "  --agreement-tol VALUE           KLU/cuDSS solution limit (default: 1e-9)\n"
      << "  --help                          Show this message\n";
}

std::string requireValue(int argc, char **argv, int &index,
                         const std::string &option)
{
  if (index + 1 >= argc) {
    throw std::runtime_error("Missing value for " + option);
  }
  ++index;
  return argv[index];
}

std::int64_t parseInteger(const std::string &text, const std::string &option,
                          bool allow_zero)
{
  std::size_t consumed = 0;
  const long long parsed = std::stoll(text, &consumed);
  if (consumed != text.size()) {
    throw std::runtime_error("Invalid integer for " + option + ": " + text);
  }
  if (parsed < 0 || (!allow_zero && parsed == 0)) {
    throw std::runtime_error(option +
                             (allow_zero ? " cannot be negative"
                                         : " must be positive"));
  }
  return static_cast<std::int64_t>(parsed);
}

double parsePositiveDouble(const std::string &text, const std::string &option)
{
  std::size_t consumed = 0;
  const double parsed = std::stod(text, &consumed);
  if (consumed != text.size() || !std::isfinite(parsed) || parsed <= 0.0) {
    throw std::runtime_error("Invalid positive value for " + option + ": " +
                             text);
  }
  return parsed;
}

CommandLine parseCommandLine(int argc, char **argv)
{
  CommandLine result;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help") {
      usage(argv[0]);
      std::exit(0);
    }
    if (argument == "--backend") {
      const std::string value =
          requireValue(argc, argv, index, argument);
      if (value == "all") {
        result.backend = BackendSelection::All;
      } else if (value == "klu") {
        result.backend = BackendSelection::Klu;
      } else if (value == "cudss") {
        result.backend = BackendSelection::Cudss;
      } else {
        throw std::runtime_error("Unknown backend selection '" + value + "'");
      }
    } else if (argument == "--cudss-mode") {
      const std::string value =
          requireValue(argc, argv, index, argument);
      if (value == "both") {
        result.cudss = CudssSelection::Both;
      } else if (value == "device") {
        result.cudss = CudssSelection::Device;
      } else if (value == "hybrid") {
        result.cudss = CudssSelection::Hybrid;
      } else {
        throw std::runtime_error("Unknown cuDSS mode '" + value + "'");
      }
    } else if (argument == "--iterations") {
      result.benchmark.iterations =
          parseInteger(requireValue(argc, argv, index, argument), argument,
                       false);
    } else if (argument == "--warmups") {
      result.benchmark.warmups =
          parseInteger(requireValue(argc, argv, index, argument), argument,
                       true);
    } else if (argument == "--scenario") {
      const std::string value =
          requireValue(argc, argv, index, argument);
      if (value == "all") {
        result.benchmark.scenario = ScenarioSelection::All;
      } else if (value == "cold") {
        result.benchmark.scenario = ScenarioSelection::Cold;
      } else if (value == "warm") {
        result.benchmark.scenario = ScenarioSelection::Warm;
      } else if (value == "repeated") {
        result.benchmark.scenario = ScenarioSelection::Repeated;
      } else {
        throw std::runtime_error("Unknown scenario selection '" + value +
                                 "'");
      }
    } else if (argument == "--device") {
      const std::int64_t device =
          parseInteger(requireValue(argc, argv, index, argument), argument,
                       true);
      if (device > std::numeric_limits<int>::max()) {
        throw std::runtime_error("--device exceeds the range of int");
      }
      result.benchmark.cuda_device = static_cast<int>(device);
    } else if (argument == "--residual-tol") {
      result.residual_tolerance =
          parsePositiveDouble(requireValue(argc, argv, index, argument),
                              argument);
    } else if (argument == "--agreement-tol") {
      result.agreement_tolerance =
          parsePositiveDouble(requireValue(argc, argv, index, argument),
                              argument);
    } else if (!argument.empty() && argument.front() == '-') {
      throw std::runtime_error("Unknown option '" + argument + "'");
    } else if (result.input_path.empty()) {
      result.input_path = argument;
    } else {
      throw std::runtime_error("Only one CSR system file may be specified");
    }
  }
  if (result.input_path.empty()) {
    throw std::runtime_error("A CSR system file is required");
  }
  if (result.benchmark.warmups >
      std::numeric_limits<std::int64_t>::max() -
          result.benchmark.iterations) {
    throw std::runtime_error("Warmup and iteration counts overflow int64");
  }
  return result;
}

Summary summarize(const std::vector<double> &samples)
{
  if (samples.empty()) {
    return {};
  }
  std::vector<double> sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  const std::size_t middle = sorted.size() / 2;
  const double median =
      sorted.size() % 2 == 0
          ? (sorted[middle - 1] + sorted[middle]) / 2.0
          : sorted[middle];
  return {samples.size(),
          sorted.front(),
          median,
          std::accumulate(samples.begin(), samples.end(), 0.0) /
              static_cast<double>(samples.size()),
          sorted.back()};
}

std::vector<double> addPhases(const PhaseSamples &samples, bool include_rhs)
{
  const std::size_t count = std::max(
      {samples.rhs_ms.size(), samples.state_ms.size(),
       samples.analysis_ms.size(),
       samples.factor_ms.size(), samples.solve_ms.size(),
       samples.solution_download_ms.size()});
  std::vector<double> result(count, 0.0);
  const auto add = [&result, count](const std::vector<double> &phase) {
    if (!phase.empty() && phase.size() != count) {
      throw std::runtime_error("Timing phase sample counts do not match");
    }
    for (std::size_t index = 0; index < phase.size(); ++index) {
      result[index] += phase[index];
    }
  };
  if (include_rhs) {
    add(samples.rhs_ms);
  }
  add(samples.state_ms);
  add(samples.analysis_ms);
  add(samples.factor_ms);
  add(samples.solve_ms);
  if (include_rhs) {
    add(samples.solution_download_ms);
  }
  return result;
}

void printTiming(const std::string &backend, const std::string &scenario,
                 const std::string &phase,
                 const std::vector<double> &samples)
{
  if (samples.empty()) {
    return;
  }
  const Summary summary = summarize(samples);
  std::cout << "TIMING"
            << " backend=" << backend
            << " scenario=" << scenario
            << " phase=" << phase
            << " count=" << summary.count
            << " min_ms=" << summary.minimum
            << " median_ms=" << summary.median
            << " mean_ms=" << summary.mean
            << " max_ms=" << summary.maximum << '\n';
}

void printBackend(const BackendResult &backend, double residual_tolerance,
                  bool &passed)
{
  std::cout << "BACKEND"
            << " name=" << backend.name
            << " setup_ms=" << backend.setup_ms
            << " format_conversion_ms=" << backend.format_conversion_ms
            << '\n';
  for (const ScenarioResult &scenario : backend.scenarios) {
    std::cout << "SCENARIO_SETUP"
              << " backend=" << backend.name
              << " scenario=" << scenario.name
              << " one_time_state_ms=" << scenario.one_time_state_ms
              << " one_time_analysis_ms=" << scenario.one_time_analysis_ms
              << " one_time_factor_ms=" << scenario.one_time_factor_ms << '\n';
    printTiming(backend.name, scenario.name, "rhs_refresh",
                scenario.samples.rhs_ms);
    printTiming(backend.name, scenario.name, "state_lifecycle",
                scenario.samples.state_ms);
    printTiming(backend.name, scenario.name, "analysis",
                scenario.samples.analysis_ms);
    printTiming(backend.name, scenario.name, "factor",
                scenario.samples.factor_ms);
    printTiming(backend.name, scenario.name, "solve",
                scenario.samples.solve_ms);
    printTiming(backend.name, scenario.name, "solution_download",
                scenario.samples.solution_download_ms);
    printTiming(backend.name, scenario.name, "solver_total",
                addPhases(scenario.samples, false));
    printTiming(backend.name, scenario.name, "end_to_end",
                addPhases(scenario.samples, true));

    const bool residual_passed =
        std::isfinite(scenario.scaled_residual) &&
        scenario.scaled_residual <= residual_tolerance;
    std::cout << "VALIDATION"
              << " backend=" << backend.name
              << " scenario=" << scenario.name
              << " scaled_residual=" << scenario.scaled_residual
              << " tolerance=" << residual_tolerance
              << " passed=" << (residual_passed ? "true" : "false") << '\n';
    passed = passed && residual_passed;
  }
}

const BackendResult *findBackend(const std::vector<BackendResult> &results,
                                 const std::string &name)
{
  const auto found =
      std::find_if(results.begin(), results.end(),
                   [&name](const BackendResult &result) {
                     return result.name == name;
                   });
  return found == results.end() ? nullptr : &*found;
}

const ScenarioResult *findScenario(const BackendResult &backend,
                                   const std::string &name)
{
  const auto found =
      std::find_if(backend.scenarios.begin(), backend.scenarios.end(),
                   [&name](const ScenarioResult &scenario) {
                     return scenario.name == name;
                   });
  return found == backend.scenarios.end() ? nullptr : &*found;
}

void compareWithKlu(const CsrSystem &system,
                    const std::vector<BackendResult> &results,
                    double tolerance, bool &passed)
{
  const BackendResult *klu = findBackend(results, "klu");
  if (klu == nullptr) {
    std::cout << "AGREEMENT skipped=true reason=no_klu_reference\n";
    return;
  }

  for (const BackendResult &candidate : results) {
    if (candidate.name == "klu") {
      continue;
    }
    for (const ScenarioResult &candidate_scenario : candidate.scenarios) {
      const ScenarioResult *reference =
          findScenario(*klu, candidate_scenario.name);
      if (reference == nullptr) {
        throw std::runtime_error("KLU reference scenario is missing");
      }
      const auto difference = gridpack::benchmark::compareSolutions(
          system, reference->solution, candidate_scenario.solution);
      const bool agreement_passed =
          std::isfinite(difference.max_relative_l2) &&
          std::isfinite(difference.max_absolute) &&
          (difference.max_relative_l2 <= tolerance ||
           difference.max_absolute <= tolerance);
      std::cout << "AGREEMENT"
                << " reference=klu"
                << " candidate=" << candidate.name
                << " scenario=" << candidate_scenario.name
                << " max_relative_l2=" << difference.max_relative_l2
                << " max_absolute=" << difference.max_absolute
                << " tolerance=" << tolerance
                << " passed=" << (agreement_passed ? "true" : "false")
                << '\n';
      passed = passed && agreement_passed;
    }
  }
}

bool wantsKlu(BackendSelection selection)
{
  return selection == BackendSelection::All ||
         selection == BackendSelection::Klu;
}

bool wantsCudss(BackendSelection selection)
{
  return selection == BackendSelection::All ||
         selection == BackendSelection::Cudss;
}

}  // namespace

int main(int argc, char **argv)
{
  try {
    const CommandLine command = parseCommandLine(argc, argv);
    const CsrSystem system =
        gridpack::benchmark::readCsrSystem(command.input_path);

    std::cout << std::scientific << std::setprecision(9);
    std::cout << "SYSTEM"
              << " name=" << std::quoted(system.name)
              << " contingency=" << std::quoted(system.contingency)
              << " newton_iteration=" << system.newton_iteration
              << " controller_iteration=" << system.controller_iteration
              << " convergence_status="
              << std::quoted(system.convergence_status)
              << " state=" << std::quoted(system.state)
              << " n=" << system.nrows
              << " nnz=" << system.nnz
              << " rhs_count=" << system.rhs_count
              << " iterations=" << command.benchmark.iterations
              << " warmups=" << command.benchmark.warmups << '\n';

    std::vector<BackendResult> results;
    if (wantsKlu(command.backend)) {
#ifdef GRIDPACK_CSR_BENCHMARK_HAVE_KLU
      results.push_back(
          gridpack::benchmark::benchmarkKlu(system, command.benchmark));
#else
      if (command.backend == BackendSelection::Klu) {
        throw std::runtime_error("KLU was requested but was not built");
      }
#endif
    }
    if (wantsCudss(command.backend)) {
#ifdef GRIDPACK_CSR_BENCHMARK_HAVE_CUDSS
      const double runtime_setup_ms =
          gridpack::benchmark::initializeCudssRuntime(
              command.benchmark.cuda_device);
      std::cout << "CUDSS_RUNTIME"
                << " device=" << command.benchmark.cuda_device
                << " setup_ms=" << runtime_setup_ms << '\n';
      if (command.cudss == CudssSelection::Both ||
          command.cudss == CudssSelection::Device) {
        results.push_back(gridpack::benchmark::benchmarkCudss(
            system, command.benchmark, CudssMode::Device));
      }
      if (command.cudss == CudssSelection::Both ||
          command.cudss == CudssSelection::Hybrid) {
        results.push_back(gridpack::benchmark::benchmarkCudss(
            system, command.benchmark, CudssMode::Hybrid));
      }
#else
      if (command.backend == BackendSelection::Cudss) {
        throw std::runtime_error("cuDSS was requested but was not built");
      }
#endif
    }

    bool passed = true;
    for (const BackendResult &result : results) {
      printBackend(result, command.residual_tolerance, passed);
    }
    compareWithKlu(system, results, command.agreement_tolerance, passed);
    std::cout << "OVERALL passed=" << (passed ? "true" : "false") << '\n';
    return passed ? 0 : 3;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
