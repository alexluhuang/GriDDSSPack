/*
 * Copyright (c) 2026 Battelle Memorial Institute
 * Licensed under the modified BSD License.
 */

#include "benchmark.hpp"

#include <cuda_runtime_api.h>
#include <cudss.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
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

const char *statusName(cudssStatus_t status) noexcept
{
  switch (status) {
    case CUDSS_STATUS_SUCCESS:
      return "success";
    case CUDSS_STATUS_NOT_INITIALIZED:
      return "not initialized";
    case CUDSS_STATUS_ALLOC_FAILED:
      return "allocation failed";
    case CUDSS_STATUS_INVALID_VALUE:
      return "invalid value";
    case CUDSS_STATUS_NOT_SUPPORTED:
      return "not supported";
    case CUDSS_STATUS_EXECUTION_FAILED:
      return "execution failed";
    case CUDSS_STATUS_INTERNAL_ERROR:
      return "internal error";
    case CUDSS_STATUS_IR_FAILED:
      return "iterative refinement failed";
  }
  return "unknown status";
}

void checkCudss(cudssStatus_t status, const std::string &operation)
{
  if (status != CUDSS_STATUS_SUCCESS) {
    throw std::runtime_error(operation + " failed: " + statusName(status) +
                             " (" +
                             std::to_string(static_cast<int>(status)) + ")");
  }
}

void checkCuda(cudaError_t status, const std::string &operation)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(operation + " failed: " +
                             cudaGetErrorString(status));
  }
}

std::size_t checkedBytes(std::size_t count, std::size_t element_size)
{
  if (element_size != 0 &&
      count > std::numeric_limits<std::size_t>::max() / element_size) {
    throw std::runtime_error("CUDA allocation size overflows size_t");
  }
  return count * element_size;
}

class CudaStream {
 public:
  CudaStream()
  {
    checkCuda(cudaStreamCreate(&value_), "cudaStreamCreate");
  }

  ~CudaStream() noexcept
  {
    if (value_ != nullptr) {
      cudaStreamDestroy(value_);
    }
  }

  CudaStream(const CudaStream &) = delete;
  CudaStream &operator=(const CudaStream &) = delete;

  cudaStream_t get() const noexcept { return value_; }

  void synchronize() const
  {
    checkCuda(cudaStreamSynchronize(value_), "cudaStreamSynchronize");
    checkCuda(cudaGetLastError(), "cudaGetLastError");
  }

 private:
  cudaStream_t value_ = nullptr;
};

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t size) : size_(size)
  {
    if (size_ != 0) {
      void *allocation = nullptr;
      checkCuda(cudaMalloc(&allocation, checkedBytes(size_, sizeof(T))),
                "cudaMalloc");
      value_ = static_cast<T *>(allocation);
    }
  }

  ~DeviceBuffer() noexcept
  {
    if (value_ != nullptr) {
      cudaFree(value_);
    }
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  T *get() noexcept { return value_; }
  const T *get() const noexcept { return value_; }

  void copyFrom(const T *source, std::size_t size,
                cudaStream_t stream) const
  {
    if (size != size_) {
      throw std::runtime_error("CUDA upload size does not match allocation");
    }
    checkCuda(cudaMemcpyAsync(value_, source,
                              checkedBytes(size_, sizeof(T)),
                              cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync host to device");
  }

  void copyTo(T *destination, std::size_t size,
              cudaStream_t stream) const
  {
    if (size != size_) {
      throw std::runtime_error("CUDA download size does not match allocation");
    }
    checkCuda(cudaMemcpyAsync(destination, value_,
                              checkedBytes(size_, sizeof(T)),
                              cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync device to host");
  }

  void zero(cudaStream_t stream) const
  {
    checkCuda(cudaMemsetAsync(value_, 0, checkedBytes(size_, sizeof(T)),
                              stream),
              "cudaMemsetAsync");
  }

 private:
  T *value_ = nullptr;
  std::size_t size_ = 0;
};

class CudssHandle {
 public:
  CudssHandle()
  {
    checkCudss(cudssCreate(&value_), "cudssCreate");
  }

  ~CudssHandle() noexcept
  {
    if (value_ != nullptr) {
      cudssDestroy(value_);
    }
  }

  CudssHandle(const CudssHandle &) = delete;
  CudssHandle &operator=(const CudssHandle &) = delete;

  cudssHandle_t get() const noexcept { return value_; }

  void setStream(cudaStream_t stream)
  {
    checkCudss(cudssSetStream(value_, stream), "cudssSetStream");
  }

 private:
  cudssHandle_t value_ = nullptr;
};

class CudssConfig {
 public:
  CudssConfig()
  {
    checkCudss(cudssConfigCreate(&value_), "cudssConfigCreate");
  }

  ~CudssConfig() noexcept
  {
    if (value_ != nullptr) {
      cudssConfigDestroy(value_);
    }
  }

  CudssConfig(const CudssConfig &) = delete;
  CudssConfig &operator=(const CudssConfig &) = delete;

  cudssConfig_t get() const noexcept { return value_; }

  void enableHybridExecution()
  {
    const int enabled = 1;
    checkCudss(cudssConfigSet(value_, CUDSS_CONFIG_HYBRID_EXECUTE_MODE,
                              &enabled, sizeof(enabled)),
               "cudssConfigSet(CUDSS_CONFIG_HYBRID_EXECUTE_MODE)");
  }

 private:
  cudssConfig_t value_ = nullptr;
};

class CudssData {
 public:
  explicit CudssData(cudssHandle_t handle) : handle_(handle)
  {
    checkCudss(cudssDataCreate(handle_, &value_), "cudssDataCreate");
  }

  ~CudssData() noexcept
  {
    if (value_ != nullptr) {
      cudssDataDestroy(handle_, value_);
    }
  }

  CudssData(const CudssData &) = delete;
  CudssData &operator=(const CudssData &) = delete;

  cudssData_t get() const noexcept { return value_; }

 private:
  cudssHandle_t handle_ = nullptr;
  cudssData_t value_ = nullptr;
};

class CudssMatrix {
 public:
  CudssMatrix() = default;

  ~CudssMatrix() noexcept
  {
    if (value_ != nullptr) {
      cudssMatrixDestroy(value_);
    }
  }

  CudssMatrix(const CudssMatrix &) = delete;
  CudssMatrix &operator=(const CudssMatrix &) = delete;

  cudssMatrix_t *address() noexcept { return &value_; }
  cudssMatrix_t get() const noexcept { return value_; }

 private:
  cudssMatrix_t value_ = nullptr;
};

class CudssSession {
 public:
  CudssSession(const CsrSystem &system, CudssMode mode)
      : system_(system),
        mode_(mode),
        device_offsets_(isDevice() ? system.row_offsets.size() : 0),
        device_columns_(isDevice() ? system.column_indices.size() : 0),
        device_values_(isDevice() ? system.values.size() : 0),
        device_rhs_(isDevice() ? system.rhs.size() : 0),
        device_solution_(isDevice() ? system.rhs.size() : 0),
        host_rhs_(system.rhs.size()),
        host_solution_(system.rhs.size())
  {
    handle_.setStream(stream_.get());
    if (mode_ == CudssMode::Hybrid) {
      config_.enableHybridExecution();
    } else {
      device_offsets_.copyFrom(system_.row_offsets.data(),
                               system_.row_offsets.size(), stream_.get());
      device_columns_.copyFrom(system_.column_indices.data(),
                               system_.column_indices.size(), stream_.get());
      device_values_.copyFrom(system_.values.data(), system_.values.size(),
                              stream_.get());
      stream_.synchronize();
    }

    const void *offsets = nullptr;
    const void *columns = nullptr;
    const void *values = nullptr;
    const void *rhs = nullptr;
    const void *solution = nullptr;
    if (isDevice()) {
      offsets = device_offsets_.get();
      columns = device_columns_.get();
      values = device_values_.get();
      rhs = device_rhs_.get();
      solution = device_solution_.get();
    } else {
      offsets = system_.row_offsets.data();
      columns = system_.column_indices.data();
      values = system_.values.data();
      rhs = host_rhs_.data();
      solution = host_solution_.data();
    }

    checkCudss(
        cudssMatrixCreateCsr(
            matrix_.address(), system_.nrows, system_.ncols, system_.nnz,
            offsets, nullptr, columns, values, CUDSS_R_64I, CUDSS_R_64I,
            CUDSS_R_64F, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL,
            CUDSS_BASE_ZERO),
        "cudssMatrixCreateCsr");
    checkCudss(cudssMatrixCreateDn(
                   rhs_.address(), system_.nrows, system_.rhs_count,
                   system_.nrows, rhs, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR),
               "cudssMatrixCreateDn(rhs)");
    checkCudss(
        cudssMatrixCreateDn(
            solution_.address(), system_.ncols, system_.rhs_count,
            system_.ncols, solution, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR),
        "cudssMatrixCreateDn(solution)");
  }

  CudssSession(const CudssSession &) = delete;
  CudssSession &operator=(const CudssSession &) = delete;

  std::unique_ptr<CudssData> createData() const
  {
    return std::make_unique<CudssData>(handle_.get());
  }

  double refreshRightHandSide()
  {
    const auto begin = Clock::now();
    if (isDevice()) {
      device_rhs_.copyFrom(system_.rhs.data(), system_.rhs.size(),
                           stream_.get());
      device_solution_.zero(stream_.get());
      stream_.synchronize();
    } else {
      std::copy(system_.rhs.begin(), system_.rhs.end(), host_rhs_.begin());
      std::fill(host_solution_.begin(), host_solution_.end(), 0.0);
    }
    return milliseconds(begin, Clock::now());
  }

  double execute(int phase, const CudssData &data,
                 const std::string &phase_name)
  {
    const auto begin = Clock::now();
    checkCudss(cudssExecute(handle_.get(), phase, config_.get(), data.get(),
                            matrix_.get(), solution_.get(), rhs_.get()),
               "cudssExecute(" + phase_name + ")");
    stream_.synchronize();
    return milliseconds(begin, Clock::now());
  }

  double captureSolution(std::vector<double> &destination)
  {
    destination.resize(host_solution_.size());
    if (isDevice()) {
      const auto begin = Clock::now();
      device_solution_.copyTo(destination.data(), destination.size(),
                              stream_.get());
      stream_.synchronize();
      return milliseconds(begin, Clock::now());
    }
    std::copy(host_solution_.begin(), host_solution_.end(),
              destination.begin());
    return 0.0;
  }

 private:
  bool isDevice() const noexcept { return mode_ == CudssMode::Device; }

  const CsrSystem &system_;
  CudssMode mode_;
  CudaStream stream_;
  CudssHandle handle_;
  CudssConfig config_;
  DeviceBuffer<std::int64_t> device_offsets_;
  DeviceBuffer<std::int64_t> device_columns_;
  DeviceBuffer<double> device_values_;
  DeviceBuffer<double> device_rhs_;
  DeviceBuffer<double> device_solution_;
  std::vector<double> host_rhs_;
  std::vector<double> host_solution_;
  CudssMatrix matrix_;
  CudssMatrix rhs_;
  CudssMatrix solution_;
};

bool isMeasured(std::int64_t iteration, const BenchmarkOptions &options)
{
  return iteration >= options.warmups;
}

ScenarioResult benchmarkCold(const CsrSystem &system, CudssSession &session,
                             const BenchmarkOptions &options)
{
  ScenarioResult result;
  result.name = "cold_analysis_factor_solve";
  const std::int64_t count = options.warmups + options.iterations;
  for (std::int64_t iteration = 0; iteration < count; ++iteration) {
    const double rhs_ms = session.refreshRightHandSide();
    const auto create_begin = Clock::now();
    std::unique_ptr<CudssData> data = session.createData();
    double state_ms = milliseconds(create_begin, Clock::now());
    const double analysis_ms =
        session.execute(CUDSS_PHASE_ANALYSIS, *data, "analysis");
    const double factor_ms =
        session.execute(CUDSS_PHASE_FACTORIZATION, *data, "factorization");
    const double solve_ms =
        session.execute(CUDSS_PHASE_SOLVE, *data, "solve");
    const auto destroy_begin = Clock::now();
    data.reset();
    state_ms += milliseconds(destroy_begin, Clock::now());
    const double solution_download_ms =
        session.captureSolution(result.solution);

    if (isMeasured(iteration, options)) {
      result.samples.rhs_ms.push_back(rhs_ms);
      result.samples.state_ms.push_back(state_ms);
      result.samples.analysis_ms.push_back(analysis_ms);
      result.samples.factor_ms.push_back(factor_ms);
      result.samples.solve_ms.push_back(solve_ms);
      result.samples.solution_download_ms.push_back(solution_download_ms);
    }
  }
  result.scaled_residual = scaledResidual(system, result.solution);
  return result;
}

ScenarioResult benchmarkWarm(const CsrSystem &system, CudssSession &session,
                             const BenchmarkOptions &options)
{
  ScenarioResult result;
  result.name = "warm_factor_solve";
  const auto create_begin = Clock::now();
  std::unique_ptr<CudssData> data = session.createData();
  result.one_time_state_ms = milliseconds(create_begin, Clock::now());
  result.one_time_analysis_ms =
      session.execute(CUDSS_PHASE_ANALYSIS, *data, "analysis");

  const std::int64_t count = options.warmups + options.iterations;
  for (std::int64_t iteration = 0; iteration < count; ++iteration) {
    const double rhs_ms = session.refreshRightHandSide();
    const double factor_ms =
        session.execute(CUDSS_PHASE_FACTORIZATION, *data, "factorization");
    const double solve_ms =
        session.execute(CUDSS_PHASE_SOLVE, *data, "solve");
    const double solution_download_ms =
        session.captureSolution(result.solution);

    if (isMeasured(iteration, options)) {
      result.samples.rhs_ms.push_back(rhs_ms);
      result.samples.factor_ms.push_back(factor_ms);
      result.samples.solve_ms.push_back(solve_ms);
      result.samples.solution_download_ms.push_back(solution_download_ms);
    }
  }
  const auto destroy_begin = Clock::now();
  data.reset();
  result.one_time_state_ms += milliseconds(destroy_begin, Clock::now());
  result.scaled_residual = scaledResidual(system, result.solution);
  return result;
}

ScenarioResult benchmarkRepeatedSolve(const CsrSystem &system,
                                      CudssSession &session,
                                      const BenchmarkOptions &options)
{
  ScenarioResult result;
  result.name = "repeated_rhs_solve";
  const auto create_begin = Clock::now();
  std::unique_ptr<CudssData> data = session.createData();
  result.one_time_state_ms = milliseconds(create_begin, Clock::now());
  result.one_time_analysis_ms =
      session.execute(CUDSS_PHASE_ANALYSIS, *data, "analysis");
  result.one_time_factor_ms =
      session.execute(CUDSS_PHASE_FACTORIZATION, *data, "factorization");

  const std::int64_t count = options.warmups + options.iterations;
  for (std::int64_t iteration = 0; iteration < count; ++iteration) {
    const double rhs_ms = session.refreshRightHandSide();
    const double solve_ms =
        session.execute(CUDSS_PHASE_SOLVE, *data, "solve");
    const double solution_download_ms =
        session.captureSolution(result.solution);

    if (isMeasured(iteration, options)) {
      result.samples.rhs_ms.push_back(rhs_ms);
      result.samples.solve_ms.push_back(solve_ms);
      result.samples.solution_download_ms.push_back(solution_download_ms);
    }
  }
  const auto destroy_begin = Clock::now();
  data.reset();
  result.one_time_state_ms += milliseconds(destroy_begin, Clock::now());
  result.scaled_residual = scaledResidual(system, result.solution);
  return result;
}

}  // namespace

double initializeCudssRuntime(int cuda_device)
{
  const auto begin = Clock::now();
  int device_count = 0;
  checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (cuda_device < 0 || cuda_device >= device_count) {
    throw std::runtime_error("Requested CUDA device is not available");
  }
  checkCuda(cudaSetDevice(cuda_device), "cudaSetDevice");
  checkCuda(cudaFree(nullptr), "cudaFree(nullptr)");
  {
    CudaStream stream;
    CudssHandle handle;
    handle.setStream(stream.get());
    CudssConfig config;
    CudssData data(handle.get());
  }
  checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  return milliseconds(begin, Clock::now());
}

BackendResult benchmarkCudss(const CsrSystem &system,
                             const BenchmarkOptions &options,
                             CudssMode mode)
{
  if (system.rhs_count != 1) {
    throw std::runtime_error(
        "The scalar cuDSS benchmark requires exactly one RHS");
  }

  BackendResult result;
  result.name = mode == CudssMode::Device ? "cudss-device" : "cudss-hybrid";

  const auto setup_begin = Clock::now();
  CudssSession session(system, mode);
  result.setup_ms = milliseconds(setup_begin, Clock::now());

  if (options.scenario == ScenarioSelection::All ||
      options.scenario == ScenarioSelection::Cold) {
    result.scenarios.push_back(benchmarkCold(system, session, options));
  }
  if (options.scenario == ScenarioSelection::All ||
      options.scenario == ScenarioSelection::Warm) {
    result.scenarios.push_back(benchmarkWarm(system, session, options));
  }
  if (options.scenario == ScenarioSelection::All ||
      options.scenario == ScenarioSelection::Repeated) {
    result.scenarios.push_back(
        benchmarkRepeatedSolve(system, session, options));
  }
  return result;
}

}  // namespace benchmark
}  // namespace gridpack
