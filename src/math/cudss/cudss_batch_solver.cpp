/*
 * Copyright (c) 2026 Battelle Memorial Institute
 * Licensed under the modified BSD License.
 */

#include "cudss/cudss_batch_solver.hpp"

#include "gridpack/utilities/exception.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <sstream>

#if defined(GRIDPACK_HAVE_CUDSS)
#include <cuda_runtime_api.h>
#include <cudss.h>
#endif

namespace gridpack {
namespace math {
namespace {

void fail(const std::string& message)
{
  throw gridpack::Exception("cuDSS batch solver: " + message);
}

std::size_t checkedProduct(std::size_t left, std::size_t right,
                           const std::string& label)
{
  if (right != 0 &&
      left > std::numeric_limits<std::size_t>::max() / right) {
    fail(label + " size overflows size_t");
  }
  return left * right;
}

void validatePattern(const RealCsrSystem& system)
{
  if (system.rows == 0 || system.columns != system.rows ||
      system.nonzeros == 0 || system.rightHandSideCount != 1 ||
      system.rows > static_cast<std::uint32_t>(
          std::numeric_limits<int>::max()) ||
      system.nonzeros > static_cast<std::uint32_t>(
          std::numeric_limits<int>::max())) {
    fail("systems must be nonempty square matrices with exactly one RHS");
  }
  if (system.rowOffsets.size() !=
        static_cast<std::size_t>(system.rows) + 1 ||
      system.columnIndices.size() != system.nonzeros) {
    fail("CSR structure lengths are inconsistent with system dimensions");
  }
  if (system.rowOffsets.front() != 0 ||
      system.rowOffsets.back() != system.nonzeros) {
    fail("CSR row offsets are inconsistent with nonzero count");
  }
  for (std::size_t row = 0; row < system.rows; ++row) {
    const std::uint32_t begin = system.rowOffsets[row];
    const std::uint32_t end = system.rowOffsets[row + 1];
    if (begin > end || end > system.nonzeros) {
      fail("CSR row offsets must be monotonic and bounded");
    }
    for (std::uint32_t entry = begin; entry < end; ++entry) {
      if (system.columnIndices[entry] >= system.columns ||
          (entry > begin && system.columnIndices[entry - 1] >=
                            system.columnIndices[entry])) {
        fail("CSR columns must be bounded and strictly increasing by row");
      }
    }
  }
}

void validateNumeric(const RealCsrSystem& pattern,
                     const CUDSSBatchSystemView& system)
{
  if (system.values == NULL || system.rightHandSide == NULL ||
      system.values->size() != pattern.nonzeros ||
      system.rightHandSide->size() != pattern.rows) {
    fail("numeric array lengths are inconsistent with the CSR pattern");
  }
}

void validateSystem(const RealCsrSystem& system)
{
  validatePattern(system);
  validateNumeric(system, CUDSSBatchSystemView(
      system.values, system.rightHandSides));
}

bool samePattern(const RealCsrSystem& left, const RealCsrSystem& right)
{
  return left.rows == right.rows && left.columns == right.columns &&
    left.nonzeros == right.nonzeros &&
    left.rowOffsets == right.rowOffsets &&
    left.columnIndices == right.columnIndices;
}

double scaledResidual(const RealCsrSystem& pattern,
                      const CUDSSBatchSystemView& system,
                      const std::vector<double>& solution)
{
  double residualSquared = 0.0;
  double matrixSquared = 0.0;
  double solutionSquared = 0.0;
  double rhsSquared = 0.0;
  for (std::size_t row = 0; row < pattern.rows; ++row) {
    double product = 0.0;
    for (std::uint32_t entry = pattern.rowOffsets[row];
         entry < pattern.rowOffsets[row + 1]; ++entry) {
      const double value = (*system.values)[entry];
      product += value * solution[pattern.columnIndices[entry]];
      matrixSquared += value * value;
    }
    const double difference = product - (*system.rightHandSide)[row];
    residualSquared += difference * difference;
    rhsSquared += (*system.rightHandSide)[row] * (*system.rightHandSide)[row];
  }
  for (std::size_t index = 0; index < solution.size(); ++index) {
    solutionSquared += solution[index] * solution[index];
  }
  const double denominator =
    std::sqrt(matrixSquared) * std::sqrt(solutionSquared) +
    std::sqrt(rhsSquared);
  return denominator == 0.0
    ? std::sqrt(residualSquared)
    : std::sqrt(residualSquared) / denominator;
}

#if defined(GRIDPACK_HAVE_CUDSS)

const char *statusName(cudssStatus_t status)
{
  switch (status) {
    case CUDSS_STATUS_SUCCESS: return "success";
    case CUDSS_STATUS_NOT_INITIALIZED: return "not initialized";
    case CUDSS_STATUS_ALLOC_FAILED: return "allocation failed";
    case CUDSS_STATUS_INVALID_VALUE: return "invalid value";
    case CUDSS_STATUS_NOT_SUPPORTED: return "not supported";
    case CUDSS_STATUS_EXECUTION_FAILED: return "execution failed";
    case CUDSS_STATUS_INTERNAL_ERROR: return "internal error";
    case CUDSS_STATUS_IR_FAILED: return "iterative refinement failed";
  }
  return "unknown status";
}

void checkCUDSS(cudssStatus_t status, const std::string& operation)
{
  if (status != CUDSS_STATUS_SUCCESS) {
    std::ostringstream message;
    message << operation << " failed: " << statusName(status)
            << " (" << static_cast<int>(status) << ")";
    fail(message.str());
  }
}

void checkCUDA(cudaError_t status, const std::string& operation)
{
  if (status != cudaSuccess) {
    fail(operation + " failed: " + cudaGetErrorString(status));
  }
}

template <typename T>
class DeviceBuffer
{
  public:
    explicit DeviceBuffer(std::size_t count) : p_data(NULL), p_count(count)
    {
      if (p_count != 0) {
        void *data = NULL;
        checkCUDA(cudaMalloc(&data,
            checkedProduct(p_count, sizeof(T), "device allocation")),
            "cudaMalloc");
        p_data = static_cast<T*>(data);
      }
    }
    ~DeviceBuffer(void) { if (p_data != NULL) cudaFree(p_data); }
    T *get(void) { return p_data; }
    std::size_t size(void) const { return p_count; }
  private:
    T *p_data;
    std::size_t p_count;
    DeviceBuffer(const DeviceBuffer&);
    DeviceBuffer& operator=(const DeviceBuffer&);
};

template <typename T>
class PinnedBuffer
{
  public:
    explicit PinnedBuffer(std::size_t count) : p_data(NULL), p_count(count)
    {
      if (p_count != 0) {
        void *data = NULL;
        checkCUDA(cudaHostAlloc(&data,
            checkedProduct(p_count, sizeof(T), "pinned allocation"),
            cudaHostAllocPortable), "cudaHostAlloc");
        p_data = static_cast<T*>(data);
      }
    }
    ~PinnedBuffer(void) { if (p_data != NULL) cudaFreeHost(p_data); }
    T *get(void) { return p_data; }
    std::size_t size(void) const { return p_count; }
  private:
    T *p_data;
    std::size_t p_count;
    PinnedBuffer(const PinnedBuffer&);
    PinnedBuffer& operator=(const PinnedBuffer&);
};

class MatrixHandle
{
  public:
    MatrixHandle(void) : p_value(NULL) {}
    ~MatrixHandle(void)
    {
      if (p_value != NULL) cudssMatrixDestroy(p_value);
    }
    cudssMatrix_t *address(void) { return &p_value; }
    cudssMatrix_t get(void) const { return p_value; }
  private:
    cudssMatrix_t p_value;
    MatrixHandle(const MatrixHandle&);
    MatrixHandle& operator=(const MatrixHandle&);
};

class DataHandle
{
  public:
    explicit DataHandle(cudssHandle_t handle)
      : p_handle(handle), p_value(NULL)
    {
      checkCUDSS(cudssDataCreate(p_handle, &p_value), "cudssDataCreate");
    }
    ~DataHandle(void)
    {
      if (p_value != NULL) cudssDataDestroy(p_handle, p_value);
    }
    cudssData_t get(void) const { return p_value; }
  private:
    cudssHandle_t p_handle;
    cudssData_t p_value;
    DataHandle(const DataHandle&);
    DataHandle& operator=(const DataHandle&);
};

class Runtime
{
  public:
    Runtime(int device, std::size_t batchSize)
      : p_device(device), p_stream(NULL), p_handle(NULL), p_config(NULL)
    {
      int deviceCount = 0;
      checkCUDA(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
      if (p_device < 0 || p_device >= deviceCount) {
        fail("configured CUDA device is unavailable");
      }
      if (batchSize == 0 ||
          batchSize > static_cast<std::size_t>(
              std::numeric_limits<int>::max())) {
        fail("uniform batch size must fit the cuDSS int parameter");
      }
      checkCUDA(cudaSetDevice(p_device), "cudaSetDevice");
      checkCUDA(cudaFree(NULL), "cudaFree(NULL)");
      try {
        checkCUDA(cudaStreamCreate(&p_stream), "cudaStreamCreate");
        checkCUDSS(cudssCreate(&p_handle), "cudssCreate");
        checkCUDSS(cudssSetStream(p_handle, p_stream), "cudssSetStream");
        checkCUDSS(cudssConfigCreate(&p_config), "cudssConfigCreate");
        const int size = static_cast<int>(batchSize);
        checkCUDSS(cudssConfigSet(p_config, CUDSS_CONFIG_UBATCH_SIZE,
            &size, sizeof(size)),
            "cudssConfigSet(CUDSS_CONFIG_UBATCH_SIZE)");
      } catch (...) {
        cleanup();
        throw;
      }
    }
    ~Runtime(void)
    {
      cudaSetDevice(p_device);
      cleanup();
    }
    int device(void) const { return p_device; }
    cudaStream_t stream(void) const { return p_stream; }
    cudssHandle_t handle(void) const { return p_handle; }
    cudssConfig_t config(void) const { return p_config; }
    void synchronize(void) const
    {
      checkCUDA(cudaStreamSynchronize(p_stream), "cudaStreamSynchronize");
      checkCUDA(cudaGetLastError(), "cudaGetLastError");
    }
    void drainNoThrow(void) const
    {
      // Descriptors and buffers referenced by queued work must remain alive
      // until the stream has reached a terminal state, including error paths.
      cudaStreamSynchronize(p_stream);
      cudaGetLastError();
    }
    void execute(int phase, cudssData_t data, cudssMatrix_t matrix,
                 cudssMatrix_t solution, cudssMatrix_t rhs,
                 const std::string& name) const
    {
      checkCUDSS(cudssExecute(p_handle, phase, p_config, data, matrix,
          solution, rhs), "cudssExecute(" + name + ")");
    }
  private:
    int p_device;
    cudaStream_t p_stream;
    cudssHandle_t p_handle;
    cudssConfig_t p_config;
    void cleanup(void)
    {
      if (p_config != NULL) {
        cudssConfigDestroy(p_config);
        p_config = NULL;
      }
      if (p_handle != NULL) {
        cudssDestroy(p_handle);
        p_handle = NULL;
      }
      if (p_stream != NULL) {
        cudaStreamDestroy(p_stream);
        p_stream = NULL;
      }
    }
    Runtime(const Runtime&);
    Runtime& operator=(const Runtime&);
};

class PatternState
{
  public:
    PatternState(const RealCsrSystem& pattern, std::size_t batchSize,
                 Runtime& runtime)
      : p_rows(pattern.rows), p_columns(pattern.columns),
        p_nonzeros(pattern.nonzeros), p_rowOffsets(pattern.rowOffsets),
        p_columnIndices(pattern.columnIndices), p_batchSize(batchSize),
        p_lastUse(0), p_factorized(false),
        p_hostValues(checkedProduct(pattern.nonzeros, batchSize,
                                   "batch values")),
        p_hostRhs(checkedProduct(pattern.rows, batchSize, "batch RHS")),
        p_hostSolution(checkedProduct(pattern.rows, batchSize,
                                     "batch solution")),
        p_deviceRowOffsets(pattern.rowOffsets.size()),
        p_deviceColumnIndices(pattern.columnIndices.size()),
        p_deviceValues(p_hostValues.size()), p_deviceRhs(p_hostRhs.size()),
        p_deviceSolution(p_hostSolution.size()), p_data(runtime.handle())
    {
      // Structure upload happens once per cached exact pattern. Keep these
      // copies synchronous so a failed second copy cannot leave queued work
      // referring to members of a partially constructed PatternState.
      checkCUDA(cudaMemcpy(p_deviceRowOffsets.get(),
          p_rowOffsets.data(),
          checkedProduct(p_rowOffsets.size(), sizeof(std::uint32_t),
                         "row-offset upload"),
          cudaMemcpyHostToDevice), "cudaMemcpy(row offsets)");
      checkCUDA(cudaMemcpy(p_deviceColumnIndices.get(),
          p_columnIndices.data(),
          checkedProduct(p_columnIndices.size(), sizeof(std::uint32_t),
                         "column-index upload"),
          cudaMemcpyHostToDevice), "cudaMemcpy(column indices)");
      checkCUDSS(cudssMatrixCreateCsr(p_matrix.address(), p_rows, p_columns,
          p_nonzeros, p_deviceRowOffsets.get(), NULL,
          p_deviceColumnIndices.get(), p_deviceValues.get(), CUDSS_R_32I,
          CUDSS_R_32I, CUDSS_R_64F, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL,
          CUDSS_BASE_ZERO), "cudssMatrixCreateCsr(uniform batch)");
      checkCUDSS(cudssMatrixCreateDn(p_rhs.address(), p_rows, 1, p_rows,
          p_deviceRhs.get(), CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR),
          "cudssMatrixCreateDn(batch RHS)");
      checkCUDSS(cudssMatrixCreateDn(p_solution.address(), p_columns, 1,
          p_columns, p_deviceSolution.get(), CUDSS_R_64F,
          CUDSS_LAYOUT_COL_MAJOR),
          "cudssMatrixCreateDn(batch solution)");
    }

    bool matches(const RealCsrSystem& system) const
    {
      return p_rows == system.rows && p_columns == system.columns &&
        p_nonzeros == system.nonzeros &&
        p_rowOffsets == system.rowOffsets &&
        p_columnIndices == system.columnIndices;
    }
    std::uint64_t lastUse(void) const { return p_lastUse; }
    void touch(std::uint64_t value) { p_lastUse = value; }
    bool factorized(void) const { return p_factorized; }
    void factorized(bool value) { p_factorized = value; }
    std::uint64_t structureBytes(void) const
    {
      return static_cast<std::uint64_t>(checkedProduct(
          p_rowOffsets.size() + p_columnIndices.size(),
          sizeof(std::uint32_t), "structure bytes"));
    }

    std::uint64_t upload(const std::vector<CUDSSBatchSystemView>& systems,
                         Runtime& runtime)
    {
      // cuDSS 0.8's selective UBATCH_MASK path rejects general matrices on
      // this platform. Pad partial calls by repeating valid inputs and always
      // execute the configured batch instead. Broker traffic and returned
      // results still contain only the requested systems.
      const std::size_t slots = p_batchSize;
      for (std::size_t slot = 0; slot < slots; ++slot) {
        const CUDSSBatchSystemView& system =
          systems[slot % systems.size()];
        std::copy(system.values->begin(), system.values->end(),
                  p_hostValues.get() + slot * p_nonzeros);
        std::copy(system.rightHandSide->begin(), system.rightHandSide->end(),
                  p_hostRhs.get() + slot * p_rows);
      }
      const std::size_t valueBytes = checkedProduct(
          checkedProduct(slots, p_nonzeros, "value upload"), sizeof(double),
          "value upload");
      const std::size_t rhsBytes = checkedProduct(
          checkedProduct(slots, p_rows, "RHS upload"), sizeof(double),
          "RHS upload");
      checkCUDA(cudaMemcpyAsync(p_deviceValues.get(), p_hostValues.get(),
          valueBytes, cudaMemcpyHostToDevice, runtime.stream()),
          "cudaMemcpyAsync(batch values)");
      checkCUDA(cudaMemcpyAsync(p_deviceRhs.get(), p_hostRhs.get(), rhsBytes,
          cudaMemcpyHostToDevice, runtime.stream()),
          "cudaMemcpyAsync(batch RHS)");
      checkCUDA(cudaMemsetAsync(p_deviceSolution.get(), 0,
          checkedProduct(slots * p_rows, sizeof(double), "solution reset"),
          runtime.stream()), "cudaMemsetAsync(batch solution)");
      return static_cast<std::uint64_t>(valueBytes + rhsBytes);
    }

    std::uint64_t download(std::size_t count, Runtime& runtime,
                           std::vector<std::vector<double> >& solutions)
    {
      const std::size_t bytes = checkedProduct(
          checkedProduct(count, p_rows, "solution download"), sizeof(double),
          "solution download");
      checkCUDA(cudaMemcpyAsync(p_hostSolution.get(), p_deviceSolution.get(),
          bytes, cudaMemcpyDeviceToHost, runtime.stream()),
          "cudaMemcpyAsync(batch solution)");
      runtime.synchronize();
      solutions.assign(count, std::vector<double>(p_rows));
      for (std::size_t slot = 0; slot < count; ++slot) {
        std::copy(p_hostSolution.get() + slot * p_rows,
                  p_hostSolution.get() + (slot + 1) * p_rows,
                  solutions[slot].begin());
      }
      return static_cast<std::uint64_t>(bytes);
    }

    cudssData_t data(void) const { return p_data.get(); }
    cudssMatrix_t matrix(void) const { return p_matrix.get(); }
    cudssMatrix_t rhs(void) const { return p_rhs.get(); }
    cudssMatrix_t solution(void) const { return p_solution.get(); }

  private:
    std::uint32_t p_rows;
    std::uint32_t p_columns;
    std::uint32_t p_nonzeros;
    std::vector<std::uint32_t> p_rowOffsets;
    std::vector<std::uint32_t> p_columnIndices;
    std::size_t p_batchSize;
    std::uint64_t p_lastUse;
    bool p_factorized;
    PinnedBuffer<double> p_hostValues;
    PinnedBuffer<double> p_hostRhs;
    PinnedBuffer<double> p_hostSolution;
    DeviceBuffer<std::uint32_t> p_deviceRowOffsets;
    DeviceBuffer<std::uint32_t> p_deviceColumnIndices;
    DeviceBuffer<double> p_deviceValues;
    DeviceBuffer<double> p_deviceRhs;
    DeviceBuffer<double> p_deviceSolution;
    MatrixHandle p_matrix;
    MatrixHandle p_rhs;
    MatrixHandle p_solution;
    DataHandle p_data;
    PatternState(const PatternState&);
    PatternState& operator=(const PatternState&);
};

#endif

} // anonymous namespace

class CUDSSBatchSolver::Impl
{
  public:
    explicit Impl(const CUDSSBatchOptions& options)
      : p_options(options), p_statistics(), p_available(false),
        p_reason("this GridPACK build does not include cuDSS support")
#if defined(GRIDPACK_HAVE_CUDSS)
        , p_runtime(), p_patterns(), p_clock(0)
#endif
    {
      if (p_options.batchSize == 0 ||
          p_options.batchSize > static_cast<std::size_t>(
              std::numeric_limits<int>::max())) {
        fail("batch size must fit the cuDSS int parameter");
      }
      if (p_options.maximumCachedPatterns == 0) {
        fail("maximum cached patterns must be positive");
      }
      if (!std::isfinite(p_options.residualTolerance) ||
          p_options.residualTolerance <= 0.0) {
        fail("residual tolerance must be finite and positive");
      }
#if defined(GRIDPACK_HAVE_CUDSS)
      try {
        p_runtime.reset(new Runtime(p_options.device, p_options.batchSize));
        p_available = true;
        p_reason.clear();
      } catch (const std::exception& error) {
        p_reason = error.what();
      }
#endif
    }

    bool available(void) const { return p_available; }
    std::string reason(void) const { return p_reason; }
    CUDSSBatchStatistics statistics(void) const
    {
      std::lock_guard<std::mutex> lock(p_mutex);
      return p_statistics;
    }

    void solve(const std::vector<RealCsrSystem>& systems,
               std::vector<std::vector<double> >& solutions)
    {
      if (systems.empty()) fail("cannot execute an empty batch");
      validateSystem(systems.front());
      std::vector<CUDSSBatchSystemView> views;
      views.reserve(systems.size());
      views.push_back(CUDSSBatchSystemView(
          systems.front().values, systems.front().rightHandSides));
      for (std::size_t index = 1; index < systems.size(); ++index) {
        validateSystem(systems[index]);
        if (!samePattern(systems.front(), systems[index])) {
          fail("uniform batch contains different CSR patterns");
        }
        views.push_back(CUDSSBatchSystemView(
            systems[index].values, systems[index].rightHandSides));
      }
      solve(systems.front(), views, solutions);
    }

    void solve(const RealCsrSystem& pattern,
               const std::vector<CUDSSBatchSystemView>& systems,
               std::vector<std::vector<double> >& solutions)
    {
      if (!p_available) fail(p_reason);
      if (systems.empty()) fail("cannot execute an empty batch");
      if (systems.size() > p_options.batchSize) {
        fail("submitted systems exceed configured uniform batch size");
      }
      validatePattern(pattern);
      for (std::size_t index = 0; index < systems.size(); ++index) {
        validateNumeric(pattern, systems[index]);
      }

      std::lock_guard<std::mutex> lock(p_mutex);
      p_statistics.submittedSystems += systems.size();

#if defined(GRIDPACK_HAVE_CUDSS)
      checkCUDA(cudaSetDevice(p_runtime->device()), "cudaSetDevice");
      PatternState *state = NULL;
      try {
        state = find(pattern);
        if (state == NULL) {
          ++p_statistics.cacheMisses;
          evictIfNeeded();
          std::unique_ptr<PatternState> created(new PatternState(
              pattern, p_options.batchSize, *p_runtime));
          state = created.get();
          p_statistics.structureUploadBytes += state->structureBytes();
          p_patterns.push_back(std::move(created));
        } else {
          ++p_statistics.cacheHits;
        }
        state->touch(++p_clock);
        const bool first = !state->factorized();
        p_statistics.numericUploadBytes +=
          state->upload(systems, *p_runtime);
        if (first) {
          p_runtime->execute(CUDSS_PHASE_ANALYSIS, state->data(),
              state->matrix(), state->solution(), state->rhs(), "analysis");
          ++p_statistics.analyses;
          p_runtime->execute(CUDSS_PHASE_FACTORIZATION, state->data(),
              state->matrix(), state->solution(), state->rhs(),
              "factorization");
          ++p_statistics.factorizations;
          state->factorized(true);
        } else {
          p_runtime->execute(CUDSS_PHASE_REFACTORIZATION, state->data(),
              state->matrix(), state->solution(), state->rhs(),
              "refactorization");
          ++p_statistics.refactorizations;
        }
        p_runtime->execute(CUDSS_PHASE_SOLVE, state->data(), state->matrix(),
            state->solution(), state->rhs(), "solve");
        ++p_statistics.solves;
        ++p_statistics.batchExecutions;
        p_statistics.solutionDownloadBytes +=
          state->download(systems.size(), *p_runtime, solutions);

        if (p_options.validateResiduals) {
          for (std::size_t index = 0; index < systems.size(); ++index) {
            const double residual = scaledResidual(
                pattern, systems[index], solutions[index]);
            if (!std::isfinite(residual) ||
                residual > p_options.residualTolerance) {
              std::ostringstream message;
              message << "system " << index << " scaled residual "
                      << residual << " exceeds tolerance "
                      << p_options.residualTolerance;
              fail(message.str());
            }
          }
        }
      } catch (...) {
        if (p_runtime) p_runtime->drainNoThrow();
        erase(state);
        throw;
      }
#endif
      p_statistics.completedSystems += systems.size();
    }

  private:
    CUDSSBatchOptions p_options;
    CUDSSBatchStatistics p_statistics;
    bool p_available;
    std::string p_reason;
    mutable std::mutex p_mutex;
#if defined(GRIDPACK_HAVE_CUDSS)
    std::unique_ptr<Runtime> p_runtime;
    std::vector<std::unique_ptr<PatternState> > p_patterns;
    std::uint64_t p_clock;

    PatternState *find(const RealCsrSystem& system)
    {
      for (std::vector<std::unique_ptr<PatternState> >::iterator iter =
             p_patterns.begin(); iter != p_patterns.end(); ++iter) {
        if ((*iter)->matches(system)) return iter->get();
      }
      return NULL;
    }

    void erase(PatternState *state)
    {
      if (state == NULL) return;
      for (std::vector<std::unique_ptr<PatternState> >::iterator iter =
             p_patterns.begin(); iter != p_patterns.end(); ++iter) {
        if (iter->get() == state) {
          p_patterns.erase(iter);
          return;
        }
      }
    }

    void evictIfNeeded(void)
    {
      if (p_patterns.size() < p_options.maximumCachedPatterns) return;
      std::vector<std::unique_ptr<PatternState> >::iterator victim =
        p_patterns.begin();
      for (std::vector<std::unique_ptr<PatternState> >::iterator iter =
             p_patterns.begin(); iter != p_patterns.end(); ++iter) {
        if ((*iter)->lastUse() < (*victim)->lastUse()) victim = iter;
      }
      p_patterns.erase(victim);
      ++p_statistics.cacheEvictions;
    }
#endif
};

CUDSSBatchSolver::CUDSSBatchSolver(const CUDSSBatchOptions& options)
  : p_impl(new Impl(options))
{}

CUDSSBatchSolver::~CUDSSBatchSolver(void)
{}

bool CUDSSBatchSolver::available(void) const
{
  return p_impl->available();
}

std::string CUDSSBatchSolver::unavailableReason(void) const
{
  return p_impl->reason();
}

void CUDSSBatchSolver::solve(
    const std::vector<RealCsrSystem>& systems,
    std::vector<std::vector<double> >& solutions)
{
  p_impl->solve(systems, solutions);
}

void CUDSSBatchSolver::solve(
    const RealCsrSystem& pattern,
    const std::vector<CUDSSBatchSystemView>& systems,
    std::vector<std::vector<double> >& solutions)
{
  p_impl->solve(pattern, systems, solutions);
}

CUDSSBatchStatistics CUDSSBatchSolver::statistics(void) const
{
  return p_impl->statistics();
}

} // namespace math
} // namespace gridpack
