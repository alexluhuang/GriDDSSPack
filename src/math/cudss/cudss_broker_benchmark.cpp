/*
 * Copyright (c) 2026 Battelle Memorial Institute
 * Licensed under the modified BSD License.
 */

#include "gridpack/math/cudss/cudss_mpi_broker.hpp"
#include "gridpack/math/petsc/petsc_csr_exporter.hpp"

#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

const double kResidualTolerance = 1.0e-10;
const std::uint64_t kBatchWaitMicroseconds = 60000000ULL;

struct Arguments
{
  std::uint64_t repetitions;
  double minimumThroughput;
  std::vector<std::string> captures;
};

bool parseUnsigned(const char *text, std::uint64_t& value)
{
  if (text == NULL || *text == '\0' || *text == '-') return false;
  errno = 0;
  char *end = NULL;
  const unsigned long long parsed = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed == 0) {
    return false;
  }
  value = static_cast<std::uint64_t>(parsed);
  return true;
}

bool parseThroughput(const char *text, double& value)
{
  if (text == NULL || *text == '\0') return false;
  errno = 0;
  char *end = NULL;
  const double parsed = std::strtod(text, &end);
  if (errno != 0 || end == text || *end != '\0' ||
      !std::isfinite(parsed) || parsed <= 0.0) {
    return false;
  }
  value = parsed;
  return true;
}

bool parseArguments(int argc, char **argv, Arguments& result,
                    std::string& error)
{
  if (argc < 4) {
    error = "expected REPETITIONS MINIMUM_SYSTEMS_PER_SECOND "
      "CAPTURE.gpcsr [CAPTURE.gpcsr ...]";
    return false;
  }
  if (!parseUnsigned(argv[1], result.repetitions)) {
    error = "REPETITIONS must be a positive integer";
    return false;
  }
  if (!parseThroughput(argv[2], result.minimumThroughput)) {
    error = "MINIMUM_SYSTEMS_PER_SECOND must be finite and positive";
    return false;
  }
  for (int index = 3; index < argc; ++index) {
    result.captures.push_back(argv[index]);
  }
  return true;
}

bool samePattern(const gridpack::math::RealCsrSystem& left,
                 const gridpack::math::RealCsrSystem& right)
{
  return left.rows == right.rows && left.columns == right.columns &&
    left.nonzeros == right.nonzeros &&
    left.rowOffsets == right.rowOffsets &&
    left.columnIndices == right.columnIndices;
}

double scaledResidual(const gridpack::math::RealCsrSystem& system,
                      const std::vector<double>& solution)
{
  if (solution.size() != system.rows) {
    return std::numeric_limits<double>::infinity();
  }
  double residualSquared = 0.0;
  double matrixSquared = 0.0;
  double solutionSquared = 0.0;
  double rhsSquared = 0.0;
  for (std::size_t row = 0; row < system.rows; ++row) {
    double product = 0.0;
    for (std::uint32_t entry = system.rowOffsets[row];
         entry < system.rowOffsets[row + 1]; ++entry) {
      const double value = system.values[entry];
      product += value * solution[system.columnIndices[entry]];
      matrixSquared += value * value;
    }
    const double difference = product - system.rightHandSides[row];
    residualSquared += difference * difference;
    rhsSquared += system.rightHandSides[row] * system.rightHandSides[row];
  }
  for (std::size_t index = 0; index < solution.size(); ++index) {
    solutionSquared += solution[index] * solution[index];
  }
  const double denominator =
    std::sqrt(matrixSquared) * std::sqrt(solutionSquared) +
    std::sqrt(rhsSquared);
  const double result = denominator == 0.0
    ? std::sqrt(residualSquared)
    : std::sqrt(residualSquared) / denominator;
  return std::isfinite(result)
    ? result : std::numeric_limits<double>::infinity();
}

void printRankError(int rank, const std::string& message)
{
  std::cerr << "cudss_broker_benchmark rank " << rank << ": "
            << message << std::endl;
}

void abortBenchmark(int rank, const std::string& message)
{
  printRankError(rank, message);
  MPI_Abort(MPI_COMM_WORLD, 2);
}

void checkPetsc(PetscErrorCode error, const char *operation)
{
  if (error != 0) {
    std::ostringstream message;
    message << operation << " failed with PETSc error " << error;
    throw std::runtime_error(message.str());
  }
}

class PetscFixture
{
  public:
    explicit PetscFixture(const gridpack::math::RealCsrSystem& system)
      : p_rowOffsets(system.rowOffsets.begin(), system.rowOffsets.end()),
        p_columnIndices(system.columnIndices.begin(),
                        system.columnIndices.end()),
        p_values(system.values.begin(), system.values.end()),
        p_rightHandSide(system.rightHandSides.begin(),
                        system.rightHandSides.begin() + system.rows),
        p_matrix(NULL), p_vector(NULL)
    {
      try {
        checkPetsc(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,
            static_cast<PetscInt>(system.rows),
            static_cast<PetscInt>(system.columns), p_rowOffsets.data(),
            p_columnIndices.data(), p_values.data(), &p_matrix),
            "MatCreateSeqAIJWithArrays");
        checkPetsc(VecCreateSeqWithArray(PETSC_COMM_SELF, 1,
            static_cast<PetscInt>(system.rows), p_rightHandSide.data(),
            &p_vector), "VecCreateSeqWithArray");
      } catch (...) {
        cleanup();
        throw;
      }
    }

    ~PetscFixture(void)
    {
      cleanup();
    }

    Mat matrix(void) const { return p_matrix; }
    Vec vector(void) const { return p_vector; }

  private:
    std::vector<PetscInt> p_rowOffsets;
    std::vector<PetscInt> p_columnIndices;
    std::vector<PetscScalar> p_values;
    std::vector<PetscScalar> p_rightHandSide;
    Mat p_matrix;
    Vec p_vector;

    void cleanup(void)
    {
      if (p_vector != NULL) VecDestroy(&p_vector);
      if (p_matrix != NULL) MatDestroy(&p_matrix);
    }

    PetscFixture(const PetscFixture&);
    PetscFixture& operator=(const PetscFixture&);
};

} // anonymous namespace

int main(int argc, char **argv)
{
  if (PetscInitialize(&argc, &argv, NULL, NULL) != 0) return 2;
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  int rank = -1;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const int owner = size - 1;
  const int workers = size - 1;

  Arguments arguments;
  std::string argumentError;
  int localPreflightFailure =
    parseArguments(argc, argv, arguments, argumentError) ? 0 : 1;
  if (size < 2) {
    localPreflightFailure = 1;
    argumentError = "requires at least two MPI ranks (workers plus owner)";
  }
  if (!localPreflightFailure &&
      arguments.repetitions >
        std::numeric_limits<std::uint64_t>::max() /
          static_cast<std::uint64_t>(workers) - 1) {
    localPreflightFailure = 1;
    argumentError =
      "REPETITIONS plus the warmup wave overflows the system count";
  }
  if (!localPreflightFailure &&
      static_cast<std::uint64_t>(arguments.captures.size()) >
        arguments.repetitions * static_cast<std::uint64_t>(workers)) {
    localPreflightFailure = 1;
    argumentError =
      "REPETITIONS times worker count must cover every capture";
  }

  std::vector<gridpack::math::RealCsrSystem> preloaded;
  std::vector<std::unique_ptr<PetscFixture> > fixtures;
  if (!localPreflightFailure && rank != owner) {
    try {
      preloaded.reserve(arguments.captures.size());
      preloaded.push_back(
          gridpack::math::readRealCsrSystem(arguments.captures.front()));
      for (std::size_t index = 1; index < arguments.captures.size();
           ++index) {
        gridpack::math::RealCsrSystem candidate =
          gridpack::math::readRealCsrSystem(arguments.captures[index]);
        if (!samePattern(preloaded.front(), candidate)) {
          std::ostringstream message;
          message << "capture does not share the exact CSR pattern: "
                  << arguments.captures[index];
          throw std::runtime_error(message.str());
        }
        preloaded.push_back(std::move(candidate));
      }
      fixtures.reserve(preloaded.size());
      for (std::size_t index = 0; index < preloaded.size(); ++index) {
        fixtures.push_back(std::unique_ptr<PetscFixture>(
            new PetscFixture(preloaded[index])));
      }
    } catch (const std::exception& error) {
      localPreflightFailure = 1;
      argumentError = error.what();
    }
  }

  int preflightFailure = 0;
  MPI_Allreduce(&localPreflightFailure, &preflightFailure, 1, MPI_INT,
                MPI_MAX, MPI_COMM_WORLD);
  if (localPreflightFailure) printRankError(rank, argumentError);
  if (preflightFailure) {
    if (rank == 0) {
      std::cout << "CUDSS_BROKER_BENCHMARK status=FAIL phase=preflight"
                << std::endl;
    }
    fixtures.clear();
    PetscFinalize();
    return 2;
  }

  MPI_Comm workerCommunicator = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, rank == owner ? MPI_UNDEFINED : 0, rank,
                 &workerCommunicator);
  if (workerCommunicator != MPI_COMM_NULL) {
    MPI_Comm_set_errhandler(workerCommunicator, MPI_ERRORS_RETURN);
  }

  gridpack::math::CUDSSBrokerOptions options;
  options.ownerRank = owner;
  options.batchSize = static_cast<std::size_t>(workers);
  options.minimumGpuBatchSize = options.batchSize;
  options.batchWaitMicroseconds = kBatchWaitMicroseconds;
  options.maximumRegisteredPatterns = 1;
  options.maximumDevicePatterns = 1;
  options.validateResiduals = false;
  options.residualTolerance = kResidualTolerance;
  options.strict = true;

  double localElapsed = 0.0;
  double localMaximumResidual = 0.0;
  std::uint64_t localCompleted = 0;
  std::uint64_t localFallbacks = 0;
  std::uint64_t localErrors = 0;
  gridpack::math::CUDSSBrokerStatistics serverStatistics;

  if (rank == owner) {
    try {
      gridpack::math::CUDSSBrokerServer server(MPI_COMM_WORLD, options);
      server.run();
      serverStatistics = server.statistics();
    } catch (const std::exception& error) {
      ++localErrors;
      abortBenchmark(rank, error.what());
    }
  } else {
    try {
      gridpack::math::CUDSSBrokerClient client(MPI_COMM_WORLD, options);

      MPI_Barrier(workerCommunicator);
      int localWarmupFailure = 0;
      try {
        const std::size_t capture =
          static_cast<std::size_t>(rank) % preloaded.size();
        gridpack::math::RealCsrSystem warmup =
          gridpack::math::extractPetscRealCsrSystem(
              fixtures[capture]->matrix(), fixtures[capture]->vector());
        std::vector<double> solution;
        const bool solved = client.solve(
            warmup, static_cast<std::uint64_t>(rank), solution);
        if (!solved) {
          ++localFallbacks;
          localWarmupFailure = 1;
        } else {
          localMaximumResidual = scaledResidual(warmup, solution);
          if (localMaximumResidual > kResidualTolerance) {
            localWarmupFailure = 1;
          }
        }
      } catch (const std::exception& error) {
        ++localErrors;
        localWarmupFailure = 1;
        printRankError(rank, error.what());
      }

      int warmupFailure = 0;
      MPI_Allreduce(&localWarmupFailure, &warmupFailure, 1, MPI_INT,
                    MPI_MAX, workerCommunicator);
      if (!warmupFailure) {
        std::vector<std::vector<double> > validationSolutions(
            preloaded.size());
        MPI_Barrier(workerCommunicator);
        const double begin = MPI_Wtime();
        for (std::uint64_t repetition = 0;
             repetition < arguments.repetitions; ++repetition) {
          try {
            const std::size_t capture = static_cast<std::size_t>(
              (repetition * static_cast<std::uint64_t>(workers) +
               static_cast<std::uint64_t>(rank)) %
              static_cast<std::uint64_t>(preloaded.size()));
            // Exercise the exact PETSc view, validation, and CSR-copy path
            // used by the production contingency callback.
            gridpack::math::RealCsrSystem packed =
              gridpack::math::extractPetscRealCsrSystem(
                  fixtures[capture]->matrix(),
                  fixtures[capture]->vector());
            std::vector<double> solution;
            const std::uint64_t task =
              repetition * static_cast<std::uint64_t>(workers) +
              static_cast<std::uint64_t>(rank);
            if (!client.solve(packed, task, solution)) {
              ++localFallbacks;
              break;
            }
            ++localCompleted;
            if (validationSolutions[capture].empty()) {
              validationSolutions[capture].swap(solution);
            }
          } catch (const std::exception& error) {
            ++localErrors;
            printRankError(rank, error.what());
            break;
          }
        }
        localElapsed = MPI_Wtime() - begin;
        for (std::size_t capture = 0; capture < preloaded.size(); ++capture) {
          if (!validationSolutions[capture].empty()) {
            localMaximumResidual = std::max(localMaximumResidual,
              scaledResidual(preloaded[capture],
                             validationSolutions[capture]));
          }
        }
      }
      client.done();
    } catch (const std::exception& error) {
      ++localErrors;
      // Fail the whole benchmark if a worker cannot enter or leave the
      // coordinated protocol. Continuing could strand peers in a worker
      // collective or leave the owner waiting forever for DONE.
      abortBenchmark(rank, error.what());
    }
  }

  // The server has drained every explicit DONE before joining this barrier.
  MPI_Barrier(MPI_COMM_WORLD);

  double elapsed = 0.0;
  double maximumResidual = 0.0;
  std::uint64_t completed = 0;
  std::uint64_t fallbacks = 0;
  std::uint64_t errors = 0;
  MPI_Allreduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(&localMaximumResidual, &maximumResidual, 1, MPI_DOUBLE,
                MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&localCompleted, &completed, 1, MPI_UINT64_T, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&localFallbacks, &fallbacks, 1, MPI_UINT64_T, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&localErrors, &errors, 1, MPI_UINT64_T, MPI_SUM,
                MPI_COMM_WORLD);

  int exitCode = 1;
  if (rank == owner) {
    const std::uint64_t expectedTimed =
      arguments.repetitions * static_cast<std::uint64_t>(workers);
    const std::uint64_t expectedAll = expectedTimed +
      static_cast<std::uint64_t>(workers);
    const double throughput = elapsed > 0.0
      ? static_cast<double>(completed) / elapsed : 0.0;
    const bool passed =
      completed == expectedTimed &&
      fallbacks == 0 && errors == 0 &&
      maximumResidual <= kResidualTolerance &&
      throughput >= arguments.minimumThroughput &&
      serverStatistics.registrations == 1 &&
      serverStatistics.solveRequests == expectedAll &&
      serverStatistics.retryResponses == 0 &&
      serverStatistics.fallbackResponses == 0 &&
      serverStatistics.errorResponses == 0 &&
      serverStatistics.fullBatches == arguments.repetitions + 1 &&
      serverStatistics.partialBatches == 0 &&
      serverStatistics.completedWorkers ==
        static_cast<std::uint64_t>(workers) &&
      serverStatistics.batch.submittedSystems == expectedAll &&
      serverStatistics.batch.completedSystems == expectedAll &&
      serverStatistics.batch.batchExecutions == arguments.repetitions + 1 &&
      serverStatistics.batch.analyses == 1 &&
      serverStatistics.batch.factorizations == 1 &&
      serverStatistics.batch.refactorizations == arguments.repetitions &&
      serverStatistics.batch.solves == arguments.repetitions + 1 &&
      serverStatistics.batch.cacheHits == arguments.repetitions &&
      serverStatistics.batch.cacheMisses == 1 &&
      serverStatistics.batch.cacheEvictions == 0;
    exitCode = passed ? 0 : 1;

    std::cout << std::setprecision(12)
              << "CUDSS_BROKER_BENCHMARK"
              << " status=" << (passed ? "PASS" : "FAIL")
              << " workers=" << workers
              << " repetitions=" << arguments.repetitions
              << " captures=" << arguments.captures.size()
              << " timed_systems=" << completed
              << " expected_timed_systems=" << expectedTimed
              << " elapsed_seconds=" << elapsed
              << " systems_per_second=" << throughput
              << " minimum_systems_per_second="
              << arguments.minimumThroughput
              << " maximum_scaled_residual=" << maximumResidual
              << " residual_tolerance=" << kResidualTolerance
              << " client_fallbacks=" << fallbacks
              << " client_errors=" << errors
              << " registrations=" << serverStatistics.registrations
              << " retries=" << serverStatistics.retryResponses
              << " solve_requests=" << serverStatistics.solveRequests
              << " broker_fallbacks="
              << serverStatistics.fallbackResponses
              << " broker_errors=" << serverStatistics.errorResponses
              << " full_batches=" << serverStatistics.fullBatches
              << " partial_batches=" << serverStatistics.partialBatches
              << " analyses=" << serverStatistics.batch.analyses
              << " factorizations="
              << serverStatistics.batch.factorizations
              << " refactorizations="
              << serverStatistics.batch.refactorizations
              << " cache_hits=" << serverStatistics.batch.cacheHits
              << " cache_misses=" << serverStatistics.batch.cacheMisses
              << " cache_evictions="
              << serverStatistics.batch.cacheEvictions
              << " batch_executions="
              << serverStatistics.batch.batchExecutions
              << " submitted_systems="
              << serverStatistics.batch.submittedSystems
              << std::endl;
  }
  MPI_Bcast(&exitCode, 1, MPI_INT, owner, MPI_COMM_WORLD);

  if (workerCommunicator != MPI_COMM_NULL) {
    MPI_Comm_free(&workerCommunicator);
  }
  fixtures.clear();
  PetscFinalize();
  return exitCode;
}
