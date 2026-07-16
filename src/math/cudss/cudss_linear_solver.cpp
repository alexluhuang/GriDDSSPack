/*
 * Copyright (c) 2026 Battelle Memorial Institute
 * Licensed under the modified BSD License.
 */

#include "cudss/cudss_linear_solver.hpp"

#include "petsc/petsc_exception.hpp"
#include "petsc/petsc_linear_solver_implementation.hpp"
#include "petsc/petsc_matrix_extractor.hpp"
#include "petsc/petsc_vector_extractor.hpp"
#include "gridpack/utilities/exception.hpp"

#include <petscmat.h>
#include <petscvec.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#if defined(GRIDPACK_HAVE_CUDSS)
#include <cuda_runtime_api.h>
#include <cudss.h>

#include <mutex>
#endif

namespace gridpack {
namespace math {
namespace {

enum CUDSSExecutionMode
{
  CUDSS_DEVICE_MODE,
  CUDSS_HYBRID_MODE
};

std::string lowercase(const std::string& input)
{
  std::string result(input);
  for (std::string::iterator position = result.begin();
       position != result.end(); ++position) {
    *position = static_cast<char>(
      std::tolower(static_cast<unsigned char>(*position)));
  }
  return result;
}

void fail(const std::string& message);

bool environmentBoolean(const char *name, bool defaultValue)
{
  const char *value = std::getenv(name);
  if (value == NULL || *value == '\0') {
    return defaultValue;
  }
  const std::string normalized = lowercase(value);
  if (normalized == "1" || normalized == "true" ||
      normalized == "yes" || normalized == "on") {
    return true;
  }
  if (normalized == "0" || normalized == "false" ||
      normalized == "no" || normalized == "off") {
    return false;
  }
  fail(std::string(name) + " must be a boolean value");
  return defaultValue;
}

int environmentInteger(const char *name, int defaultValue)
{
  const char *value = std::getenv(name);
  if (value == NULL || *value == '\0') {
    return defaultValue;
  }
  errno = 0;
  char *end = NULL;
  const long parsed = std::strtol(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' ||
      parsed < std::numeric_limits<int>::min() ||
      parsed > std::numeric_limits<int>::max()) {
    fail(std::string(name) + " must be an integer");
  }
  return static_cast<int>(parsed);
}

double environmentDouble(const char *name, double defaultValue)
{
  const char *value = std::getenv(name);
  if (value == NULL || *value == '\0') {
    return defaultValue;
  }
  errno = 0;
  char *end = NULL;
  const double parsed = std::strtod(value, &end);
  if (errno != 0 || end == value || *end != '\0') {
    fail(std::string(name) + " must be a floating-point value");
  }
  return parsed;
}

void fail(const std::string& message)
{
  throw gridpack::Exception("cuDSS linear solver: " + message);
}

class CUDSSLinearSolverBase
  : public LinearSolverImplementation<RealType, int>
{
public:
  typedef LinearSolverImplementation<RealType, int> BaseType;
  typedef BaseType::MatrixType MatrixType;
  typedef BaseType::VectorType VectorType;

  CUDSSLinearSolverBase(
    MatrixType& matrix,
    utility::Configuration::CursorPtr parentConfiguration,
    bool cudssCompiled)
    : BaseType(matrix),
      p_parentConfiguration(parentConfiguration),
      p_fallback(),
      p_fallbackHasSolved(false),
      p_strict(false),
      p_diagnostics(false),
      p_mode(CUDSS_DEVICE_MODE),
      p_device(0),
      p_residualTolerance(1.0e-10),
      p_lastFallbackReason(),
      p_statistics()
  {
    p_statistics.backend = "cudss";
    p_statistics.cudssCompiled = cudssCompiled;
  }

  virtual ~CUDSSLinearSolverBase(void)
  {
    if (p_diagnostics && this->communicator().worldRank() == 0) {
      std::ostringstream message;
      message << "GridPACK cuDSS diagnostics:"
              << " mode="
              << (p_mode == CUDSS_DEVICE_MODE ? "device" : "hybrid")
              << " strict=" << (p_strict ? "true" : "false")
              << " compiled="
              << (p_statistics.cudssCompiled ? "true" : "false")
              << " owner_world_rank=" << p_statistics.ownerWorldRank
              << " device=" << p_statistics.device
              << " analyses=" << p_statistics.analyses
              << " factorizations=" << p_statistics.factorizations
              << " refactorizations=" << p_statistics.refactorizations
              << " solves=" << p_statistics.solves
              << " cache_hits=" << p_statistics.cacheHits
              << " cache_misses=" << p_statistics.cacheMisses
              << " fallbacks=" << p_statistics.fallbacks;
      if (std::isfinite(p_statistics.lastScaledResidual)) {
        message << " last_scaled_residual="
                << p_statistics.lastScaledResidual;
      }
      if (!p_lastFallbackReason.empty()) {
        message << " last_fallback=\"" << p_lastFallbackReason << "\"";
      }
      std::cerr << message.str() << std::endl;
    }
  }

  LinearSolverStatistics statistics(void) const
  {
    return p_statistics;
  }

protected:
  utility::Configuration::CursorPtr p_parentConfiguration;
  mutable boost::scoped_ptr<
    PETScLinearSolverImplementation<RealType, int> > p_fallback;
  mutable bool p_fallbackHasSolved;
  bool p_strict;
  bool p_diagnostics;
  CUDSSExecutionMode p_mode;
  int p_device;
  double p_residualTolerance;
  mutable std::string p_lastFallbackReason;
  mutable LinearSolverStatistics p_statistics;

  void p_configure(utility::Configuration::CursorPtr props)
  {
    BaseType::p_configure(props);

    if (props) {
      p_strict = props->get("CUDSSStrict", p_strict);
      p_diagnostics = props->get("CUDSSDiagnostics", p_diagnostics);
      p_device = props->get("CUDSSDevice", p_device);
      p_residualTolerance =
        props->get("CUDSSResidualTolerance", p_residualTolerance);

      const std::string configuredMode =
        lowercase(props->get("CUDSSMode", "device"));
      if (configuredMode == "device") {
        p_mode = CUDSS_DEVICE_MODE;
      } else if (configuredMode == "hybrid") {
        p_mode = CUDSS_HYBRID_MODE;
      } else {
        fail("CUDSSMode must be \"device\" or \"hybrid\"");
      }
    }

    p_strict = environmentBoolean("GRIDPACK_CUDSS_STRICT", p_strict);
    p_diagnostics =
      environmentBoolean("GRIDPACK_CUDSS_DIAGNOSTICS", p_diagnostics);
    p_device = environmentInteger("GRIDPACK_CUDSS_DEVICE", p_device);
    p_residualTolerance = environmentDouble(
      "GRIDPACK_CUDSS_RESIDUAL_TOLERANCE", p_residualTolerance);
    const char *environmentMode = std::getenv("GRIDPACK_CUDSS_MODE");
    const std::string configuredMode =
      lowercase(environmentMode != NULL && *environmentMode != '\0'
                  ? environmentMode
                  : (p_mode == CUDSS_DEVICE_MODE ? "device" : "hybrid"));
    if (configuredMode == "device") {
      p_mode = CUDSS_DEVICE_MODE;
    } else if (configuredMode == "hybrid") {
      p_mode = CUDSS_HYBRID_MODE;
    } else {
      fail("CUDSSMode/GRIDPACK_CUDSS_MODE must be \"device\" or \"hybrid\"");
    }

    if (p_device < 0) {
      fail("CUDSSDevice must be nonnegative");
    }
    if (!std::isfinite(p_residualTolerance) ||
        p_residualTolerance <= 0.0) {
      fail("CUDSSResidualTolerance must be finite and positive");
    }

    this->p_configureBackend();
  }

  virtual void p_configureBackend(void) = 0;

  void p_unavailable(const std::string& reason)
  {
    p_lastFallbackReason = reason;
    if (p_strict) {
      fail(reason);
    }
  }

  void p_fallbackSolve(const VectorType& b, VectorType& x,
                       bool resolve, const std::string& reason) const
  {
    if (p_strict) {
      fail(reason);
    }

    p_lastFallbackReason = reason;
    ++p_statistics.fallbacks;

    if (!p_fallback) {
      p_fallback.reset(
        new PETScLinearSolverImplementation<RealType, int>(this->p_matrix));
      p_fallback->configurationKey(this->configurationKey());
      p_fallback->configure(p_parentConfiguration);
    }

    if (resolve && p_fallbackHasSolved) {
      p_fallback->resolve(b, x);
    } else {
      p_fallback->solve(b, x);
      p_fallbackHasSolved = true;
    }
  }
};

#if !defined(GRIDPACK_HAVE_CUDSS)

class UnavailableCUDSSLinearSolver
  : public CUDSSLinearSolverBase
{
public:
  UnavailableCUDSSLinearSolver(
    MatrixType& matrix,
    utility::Configuration::CursorPtr parentConfiguration)
    : CUDSSLinearSolverBase(matrix, parentConfiguration, false)
  {}

protected:
  void p_configureBackend(void)
  {
    p_unavailable(
      "this GridPACK build does not include cuDSS support");
  }

  void p_solveImpl(MatrixType&, const VectorType& b, VectorType& x) const
  {
    p_fallbackSolve(
      b, x, false, "this GridPACK build does not include cuDSS support");
  }

  void p_resolveImpl(const VectorType& b, VectorType& x) const
  {
    p_fallbackSolve(
      b, x, true, "this GridPACK build does not include cuDSS support");
  }
};

#else

const char *cudssStatusName(cudssStatus_t status)
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

void checkCUDSS(cudssStatus_t status, const std::string& operation)
{
  if (status != CUDSS_STATUS_SUCCESS) {
    std::ostringstream message;
    message << operation << " failed: " << cudssStatusName(status)
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

void checkPETSc(PetscErrorCode status)
{
  if (status != 0) {
    throw PETScException(status);
  }
}

std::size_t checkedBytes(std::size_t count, std::size_t elementSize)
{
  if (elementSize != 0 &&
      count > std::numeric_limits<std::size_t>::max() / elementSize) {
    fail("CUDA allocation size exceeds size_t");
  }
  return count * elementSize;
}

class CudaStream
{
public:
  CudaStream(void)
    : p_stream(NULL)
  {
    checkCUDA(cudaStreamCreate(&p_stream), "cudaStreamCreate");
  }

  ~CudaStream(void)
  {
    if (p_stream != NULL) {
      cudaStreamDestroy(p_stream);
    }
  }

  cudaStream_t get(void) const
  {
    return p_stream;
  }

  void synchronize(void) const
  {
    checkCUDA(cudaStreamSynchronize(p_stream), "cudaStreamSynchronize");
    checkCUDA(cudaGetLastError(), "cudaGetLastError");
  }

private:
  CudaStream(const CudaStream&);
  CudaStream& operator=(const CudaStream&);

  cudaStream_t p_stream;
};

template <typename T>
class DeviceBuffer
{
public:
  explicit DeviceBuffer(std::size_t size)
    : p_value(NULL),
      p_size(size)
  {
    if (p_size != 0) {
      void *allocation = NULL;
      checkCUDA(
        cudaMalloc(&allocation, checkedBytes(p_size, sizeof(T))),
        "cudaMalloc");
      p_value = static_cast<T *>(allocation);
    }
  }

  ~DeviceBuffer(void)
  {
    if (p_value != NULL) {
      cudaFree(p_value);
    }
  }

  T *get(void)
  {
    return p_value;
  }

  const T *get(void) const
  {
    return p_value;
  }

  void copyFrom(const T *source, std::size_t size,
                cudaStream_t stream) const
  {
    if (size != p_size) {
      fail("CUDA upload size does not match allocation");
    }
    checkCUDA(
      cudaMemcpyAsync(p_value, source, checkedBytes(p_size, sizeof(T)),
                      cudaMemcpyHostToDevice, stream),
      "cudaMemcpyAsync host to device");
  }

  void copyTo(T *destination, std::size_t size,
              cudaStream_t stream) const
  {
    if (size != p_size) {
      fail("CUDA download size does not match allocation");
    }
    checkCUDA(
      cudaMemcpyAsync(destination, p_value,
                      checkedBytes(p_size, sizeof(T)),
                      cudaMemcpyDeviceToHost, stream),
      "cudaMemcpyAsync device to host");
  }

  void zero(cudaStream_t stream) const
  {
    checkCUDA(
      cudaMemsetAsync(p_value, 0, checkedBytes(p_size, sizeof(T)), stream),
      "cudaMemsetAsync");
  }

private:
  DeviceBuffer(const DeviceBuffer&);
  DeviceBuffer& operator=(const DeviceBuffer&);

  T *p_value;
  std::size_t p_size;
};

class CudssHandle
{
public:
  CudssHandle(void)
    : p_handle(NULL)
  {
    checkCUDSS(cudssCreate(&p_handle), "cudssCreate");
  }

  ~CudssHandle(void)
  {
    if (p_handle != NULL) {
      cudssDestroy(p_handle);
    }
  }

  cudssHandle_t get(void) const
  {
    return p_handle;
  }

  void setStream(cudaStream_t stream)
  {
    checkCUDSS(cudssSetStream(p_handle, stream), "cudssSetStream");
  }

private:
  CudssHandle(const CudssHandle&);
  CudssHandle& operator=(const CudssHandle&);

  cudssHandle_t p_handle;
};

class CudssConfig
{
public:
  CudssConfig(void)
    : p_config(NULL)
  {
    checkCUDSS(cudssConfigCreate(&p_config), "cudssConfigCreate");
  }

  ~CudssConfig(void)
  {
    if (p_config != NULL) {
      cudssConfigDestroy(p_config);
    }
  }

  cudssConfig_t get(void) const
  {
    return p_config;
  }

  void enableHybridExecution(void)
  {
    const int enabled = 1;
    checkCUDSS(
      cudssConfigSet(p_config, CUDSS_CONFIG_HYBRID_EXECUTE_MODE,
                     &enabled, sizeof(enabled)),
      "cudssConfigSet(CUDSS_CONFIG_HYBRID_EXECUTE_MODE)");
  }

private:
  CudssConfig(const CudssConfig&);
  CudssConfig& operator=(const CudssConfig&);

  cudssConfig_t p_config;
};

class CudssData
{
public:
  explicit CudssData(cudssHandle_t handle)
    : p_handle(handle),
      p_data(NULL)
  {
    checkCUDSS(cudssDataCreate(p_handle, &p_data), "cudssDataCreate");
  }

  ~CudssData(void)
  {
    if (p_data != NULL) {
      cudssDataDestroy(p_handle, p_data);
    }
  }

  cudssData_t get(void) const
  {
    return p_data;
  }

private:
  CudssData(const CudssData&);
  CudssData& operator=(const CudssData&);

  cudssHandle_t p_handle;
  cudssData_t p_data;
};

class CudssMatrix
{
public:
  CudssMatrix(void)
    : p_matrix(NULL)
  {}

  ~CudssMatrix(void)
  {
    if (p_matrix != NULL) {
      cudssMatrixDestroy(p_matrix);
    }
  }

  cudssMatrix_t *address(void)
  {
    return &p_matrix;
  }

  cudssMatrix_t get(void) const
  {
    return p_matrix;
  }

private:
  CudssMatrix(const CudssMatrix&);
  CudssMatrix& operator=(const CudssMatrix&);

  cudssMatrix_t p_matrix;
};

class CudssRuntime
{
public:
  explicit CudssRuntime(CUDSSExecutionMode mode)
    : p_stream(),
      p_handle(),
      p_config()
  {
    p_handle.setStream(p_stream.get());
    if (mode == CUDSS_HYBRID_MODE) {
      p_config.enableHybridExecution();
    }
  }

  cudssHandle_t handle(void) const
  {
    return p_handle.get();
  }

  cudaStream_t stream(void) const
  {
    return p_stream.get();
  }

  void synchronize(void) const
  {
    p_stream.synchronize();
  }

  void execute(int phase, const CudssData& data,
               cudssMatrix_t matrix, cudssMatrix_t solution,
               cudssMatrix_t rhs, const std::string& phaseName) const
  {
    checkCUDSS(
      cudssExecute(p_handle.get(), phase, p_config.get(), data.get(),
                   matrix, solution, rhs),
      "cudssExecute(" + phaseName + ")");
    p_stream.synchronize();
  }

private:
  CudssRuntime(const CudssRuntime&);
  CudssRuntime& operator=(const CudssRuntime&);

  CudaStream p_stream;
  CudssHandle p_handle;
  CudssConfig p_config;
};

struct HostSystem
{
  std::int64_t rows;
  std::vector<std::int32_t> rowOffsets;
  std::vector<std::int32_t> columnIndices;
  std::vector<double> values;
  std::vector<double> rhs;
};

class MatCsrReadView
{
public:
  explicit MatCsrReadView(Mat matrix)
    : p_matrix(matrix),
      p_rows(0),
      p_rowOffsets(NULL),
      p_columnIndices(NULL),
      p_values(NULL),
      p_rowViewActive(false),
      p_valuesActive(false)
  {
    PetscBool done = PETSC_FALSE;
    checkPETSc(
      MatGetRowIJ(p_matrix, 0, PETSC_FALSE, PETSC_FALSE, &p_rows,
                  &p_rowOffsets, &p_columnIndices, &done));
    if (done != PETSC_TRUE) {
      fail("MatGetRowIJ did not expose a CSR representation");
    }
    p_rowViewActive = true;

    try {
      checkPETSc(MatSeqAIJGetArrayRead(p_matrix, &p_values));
      p_valuesActive = true;
    } catch (...) {
      closeNoThrow();
      throw;
    }
  }

  ~MatCsrReadView(void)
  {
    closeNoThrow();
  }

  PetscInt rows(void) const
  {
    return p_rows;
  }

  const PetscInt *rowOffsets(void) const
  {
    return p_rowOffsets;
  }

  const PetscInt *columnIndices(void) const
  {
    return p_columnIndices;
  }

  const PetscScalar *values(void) const
  {
    return p_values;
  }

  void close(void)
  {
    PetscErrorCode firstError = 0;
    if (p_valuesActive) {
      firstError = MatSeqAIJRestoreArrayRead(p_matrix, &p_values);
      p_valuesActive = false;
    }
    if (p_rowViewActive) {
      PetscBool done = PETSC_FALSE;
      const PetscErrorCode rowError =
        MatRestoreRowIJ(p_matrix, 0, PETSC_FALSE, PETSC_FALSE, &p_rows,
                        &p_rowOffsets, &p_columnIndices, &done);
      p_rowViewActive = false;
      if (firstError == 0) {
        firstError = rowError;
      }
    }
    checkPETSc(firstError);
  }

private:
  MatCsrReadView(const MatCsrReadView&);
  MatCsrReadView& operator=(const MatCsrReadView&);

  void closeNoThrow(void)
  {
    try {
      close();
    } catch (...) {
    }
  }

  Mat p_matrix;
  PetscInt p_rows;
  const PetscInt *p_rowOffsets;
  const PetscInt *p_columnIndices;
  const PetscScalar *p_values;
  bool p_rowViewActive;
  bool p_valuesActive;
};

class VecReadView
{
public:
  explicit VecReadView(Vec vector)
    : p_vector(vector),
      p_values(NULL),
      p_active(false)
  {
    checkPETSc(VecGetArrayRead(p_vector, &p_values));
    p_active = true;
  }

  ~VecReadView(void)
  {
    closeNoThrow();
  }

  const PetscScalar *values(void) const
  {
    return p_values;
  }

  void close(void)
  {
    if (p_active) {
      const PetscErrorCode status =
        VecRestoreArrayRead(p_vector, &p_values);
      p_active = false;
      checkPETSc(status);
    }
  }

private:
  VecReadView(const VecReadView&);
  VecReadView& operator=(const VecReadView&);

  void closeNoThrow(void)
  {
    try {
      close();
    } catch (...) {
    }
  }

  Vec p_vector;
  const PetscScalar *p_values;
  bool p_active;
};

class VecWriteView
{
public:
  explicit VecWriteView(Vec vector)
    : p_vector(vector),
      p_values(NULL),
      p_active(false)
  {
    checkPETSc(VecGetArray(p_vector, &p_values));
    p_active = true;
  }

  ~VecWriteView(void)
  {
    closeNoThrow();
  }

  PetscScalar *values(void)
  {
    return p_values;
  }

  void close(void)
  {
    if (p_active) {
      const PetscErrorCode status = VecRestoreArray(p_vector, &p_values);
      p_active = false;
      checkPETSc(status);
    }
  }

private:
  VecWriteView(const VecWriteView&);
  VecWriteView& operator=(const VecWriteView&);

  void closeNoThrow(void)
  {
    try {
      close();
    } catch (...) {
    }
  }

  Vec p_vector;
  PetscScalar *p_values;
  bool p_active;
};

bool isExactObjectType(PetscObject object, const char *type)
{
  PetscBool matches = PETSC_FALSE;
  checkPETSc(PetscObjectTypeCompare(object, type, &matches));
  return matches == PETSC_TRUE;
}

MPI_Comm objectCommunicator(PetscObject object)
{
  MPI_Comm communicator = MPI_COMM_NULL;
  checkPETSc(PetscObjectGetComm(object, &communicator));
  return communicator;
}

void requireCongruentCommunicators(
  PetscObject left, PetscObject right, const std::string& label)
{
  int comparison = MPI_UNEQUAL;
  const int status = MPI_Comm_compare(
    objectCommunicator(left), objectCommunicator(right), &comparison);
  if (status != MPI_SUCCESS ||
      (comparison != MPI_IDENT && comparison != MPI_CONGRUENT)) {
    fail(label + " PETSc communicators must be congruent");
  }
}

HostSystem extractHostSystem(RealMatrix& matrix, const RealVector& rhs)
{
  Mat petscMatrix = *PETScMatrix(matrix);
  Vec petscRhs = *PETScVector(rhs);

  requireCongruentCommunicators(
    reinterpret_cast<PetscObject>(petscMatrix),
    reinterpret_cast<PetscObject>(petscRhs), "matrix and right-hand side");

  if (!isExactObjectType(
        reinterpret_cast<PetscObject>(petscMatrix), MATSEQAIJ)) {
    fail("matrix must have exact PETSc type MATSEQAIJ");
  }
  if (!isExactObjectType(
        reinterpret_cast<PetscObject>(petscRhs), VECSEQ)) {
    fail("right-hand side must have exact PETSc type VECSEQ");
  }

  PetscBool assembled = PETSC_FALSE;
  checkPETSc(MatAssembled(petscMatrix, &assembled));
  if (assembled != PETSC_TRUE) {
    fail("matrix must be assembled before solve");
  }

  PetscInt rows = 0;
  PetscInt columns = 0;
  PetscInt localRows = 0;
  PetscInt localColumns = 0;
  PetscInt rhsSize = 0;
  PetscInt rhsLocalSize = 0;
  checkPETSc(MatGetSize(petscMatrix, &rows, &columns));
  checkPETSc(MatGetLocalSize(petscMatrix, &localRows, &localColumns));
  checkPETSc(VecGetSize(petscRhs, &rhsSize));
  checkPETSc(VecGetLocalSize(petscRhs, &rhsLocalSize));

  if (rows <= 0 || rows != columns || localRows != rows ||
      localColumns != columns) {
    fail("cuDSS requires a nonempty square sequential matrix");
  }
  if (rhsSize != rows || rhsLocalSize != rows) {
    fail("right-hand side size must equal the matrix row count");
  }
  if (rows > std::numeric_limits<std::int32_t>::max()) {
    fail("matrix row count exceeds the int32 cuDSS backend limit");
  }

  MatCsrReadView matrixView(petscMatrix);
  if (matrixView.rows() != rows) {
    fail("PETSc CSR row count is inconsistent with MatGetSize");
  }

  const PetscInt nonzeros = matrixView.rowOffsets()[rows];
  if (nonzeros <= 0 ||
      nonzeros > std::numeric_limits<std::int32_t>::max()) {
    fail("matrix nonzero count is outside the int32 cuDSS backend limit");
  }

  HostSystem result;
  result.rows = static_cast<std::int64_t>(rows);
  result.rowOffsets.resize(static_cast<std::size_t>(rows) + 1);
  result.columnIndices.resize(static_cast<std::size_t>(nonzeros));
  result.values.resize(static_cast<std::size_t>(nonzeros));
  result.rhs.resize(static_cast<std::size_t>(rows));

  for (PetscInt row = 0; row <= rows; ++row) {
    const PetscInt offset = matrixView.rowOffsets()[row];
    if (offset < 0 ||
        offset > std::numeric_limits<std::int32_t>::max()) {
      fail("CSR row offset is outside the int32 range");
    }
    result.rowOffsets[static_cast<std::size_t>(row)] =
      static_cast<std::int32_t>(offset);
  }
  if (result.rowOffsets.front() != 0) {
    fail("CSR row offsets must begin at zero");
  }

  for (PetscInt row = 0; row < rows; ++row) {
    const std::int32_t begin =
      result.rowOffsets[static_cast<std::size_t>(row)];
    const std::int32_t end =
      result.rowOffsets[static_cast<std::size_t>(row) + 1];
    if (begin > end) {
      fail("CSR row offsets must be monotonic");
    }
    std::int32_t previousColumn = -1;
    for (std::int32_t entry = begin; entry < end; ++entry) {
      const PetscInt column = matrixView.columnIndices()[entry];
      if (column < 0 || column >= columns ||
          column > std::numeric_limits<std::int32_t>::max()) {
        fail("CSR column index is outside matrix bounds");
      }
      const std::int32_t narrowedColumn =
        static_cast<std::int32_t>(column);
      if (narrowedColumn <= previousColumn) {
        fail("CSR columns must be strictly increasing within each row");
      }
      previousColumn = narrowedColumn;
      result.columnIndices[static_cast<std::size_t>(entry)] =
        narrowedColumn;

      const double value =
        static_cast<double>(PetscRealPart(matrixView.values()[entry]));
      if (!std::isfinite(value)) {
        fail("matrix contains a non-finite value");
      }
      result.values[static_cast<std::size_t>(entry)] = value;
    }
  }

  matrixView.close();

  VecReadView rhsView(petscRhs);
  for (PetscInt row = 0; row < rows; ++row) {
    const double value =
      static_cast<double>(PetscRealPart(rhsView.values()[row]));
    if (!std::isfinite(value)) {
      fail("right-hand side contains a non-finite value");
    }
    result.rhs[static_cast<std::size_t>(row)] = value;
  }
  rhsView.close();

  return result;
}

void writeSolution(const std::vector<double>& solution,
                   RealMatrix& matrix, RealVector& x)
{
  Mat petscMatrix = *PETScMatrix(matrix);
  Vec petscSolution = *PETScVector(x);
  requireCongruentCommunicators(
    reinterpret_cast<PetscObject>(petscMatrix),
    reinterpret_cast<PetscObject>(petscSolution), "matrix and solution");
  if (!isExactObjectType(
        reinterpret_cast<PetscObject>(petscSolution), VECSEQ)) {
    fail("solution must have exact PETSc type VECSEQ");
  }

  PetscInt size = 0;
  PetscInt localSize = 0;
  checkPETSc(VecGetSize(petscSolution, &size));
  checkPETSc(VecGetLocalSize(petscSolution, &localSize));
  if (size != localSize ||
      static_cast<std::size_t>(size) != solution.size()) {
    fail("solution vector size is inconsistent with the cuDSS result");
  }

  VecWriteView solutionView(petscSolution);
  for (PetscInt row = 0; row < size; ++row) {
    solutionView.values()[row] =
      static_cast<PetscScalar>(solution[static_cast<std::size_t>(row)]);
  }
  solutionView.close();
}

double scaledResidual(const HostSystem& system,
                      const std::vector<double>& solution)
{
  if (solution.size() != static_cast<std::size_t>(system.rows)) {
    fail("solution length is inconsistent during residual validation");
  }

  long double residualSquared = 0.0L;
  long double matrixSquared = 0.0L;
  long double solutionSquared = 0.0L;
  long double rhsSquared = 0.0L;

  for (std::int64_t row = 0; row < system.rows; ++row) {
    long double product = 0.0L;
    const std::int32_t begin =
      system.rowOffsets[static_cast<std::size_t>(row)];
    const std::int32_t end =
      system.rowOffsets[static_cast<std::size_t>(row) + 1];
    for (std::int32_t entry = begin; entry < end; ++entry) {
      const double value = system.values[static_cast<std::size_t>(entry)];
      const std::int32_t column =
        system.columnIndices[static_cast<std::size_t>(entry)];
      product += static_cast<long double>(value) *
                 static_cast<long double>(
                   solution[static_cast<std::size_t>(column)]);
      matrixSquared +=
        static_cast<long double>(value) * static_cast<long double>(value);
    }
    const long double rhs =
      static_cast<long double>(system.rhs[static_cast<std::size_t>(row)]);
    const long double residual = product - rhs;
    residualSquared += residual * residual;
    rhsSquared += rhs * rhs;

    const long double x =
      static_cast<long double>(solution[static_cast<std::size_t>(row)]);
    solutionSquared += x * x;
  }

  const long double numerator = std::sqrt(residualSquared);
  const long double denominator =
    std::sqrt(matrixSquared) * std::sqrt(solutionSquared) +
    std::sqrt(rhsSquared);
  if (denominator == 0.0L) {
    return numerator == 0.0L
      ? 0.0
      : std::numeric_limits<double>::infinity();
  }
  return static_cast<double>(numerator / denominator);
}

class PatternState
{
public:
  PatternState(const HostSystem& system, CUDSSExecutionMode mode,
               CudssRuntime& runtime)
    : p_rows(system.rows),
      p_mode(mode),
      p_rowOffsets(system.rowOffsets),
      p_columnIndices(system.columnIndices),
      p_values(system.values),
      p_rhs(system.rhs),
      p_solution(static_cast<std::size_t>(system.rows), 0.0),
      p_deviceRowOffsets(),
      p_deviceColumnIndices(),
      p_deviceValues(),
      p_deviceRhs(),
      p_deviceSolution(),
      p_matrix(),
      p_rhsMatrix(),
      p_solutionMatrix(),
      p_data(runtime.handle())
  {
    if (p_mode == CUDSS_DEVICE_MODE) {
      p_deviceRowOffsets.reset(
        new DeviceBuffer<std::int32_t>(p_rowOffsets.size()));
      p_deviceColumnIndices.reset(
        new DeviceBuffer<std::int32_t>(p_columnIndices.size()));
      p_deviceValues.reset(new DeviceBuffer<double>(p_values.size()));
      p_deviceRhs.reset(new DeviceBuffer<double>(p_rhs.size()));
      p_deviceSolution.reset(new DeviceBuffer<double>(p_solution.size()));

      p_deviceRowOffsets->copyFrom(
        &p_rowOffsets[0], p_rowOffsets.size(), runtime.stream());
      p_deviceColumnIndices->copyFrom(
        &p_columnIndices[0], p_columnIndices.size(), runtime.stream());
      p_deviceValues->copyFrom(
        &p_values[0], p_values.size(), runtime.stream());
      runtime.synchronize();
    }

    const void *rowOffsets =
      p_mode == CUDSS_DEVICE_MODE
        ? static_cast<const void *>(p_deviceRowOffsets->get())
        : static_cast<const void *>(&p_rowOffsets[0]);
    const void *columnIndices =
      p_mode == CUDSS_DEVICE_MODE
        ? static_cast<const void *>(p_deviceColumnIndices->get())
        : static_cast<const void *>(&p_columnIndices[0]);
    const void *values =
      p_mode == CUDSS_DEVICE_MODE
        ? static_cast<const void *>(p_deviceValues->get())
        : static_cast<const void *>(&p_values[0]);
    const void *rhs =
      p_mode == CUDSS_DEVICE_MODE
        ? static_cast<const void *>(p_deviceRhs->get())
        : static_cast<const void *>(&p_rhs[0]);
    const void *solution =
      p_mode == CUDSS_DEVICE_MODE
        ? static_cast<const void *>(p_deviceSolution->get())
        : static_cast<const void *>(&p_solution[0]);

    checkCUDSS(
      cudssMatrixCreateCsr(
        p_matrix.address(), p_rows, p_rows,
        static_cast<std::int64_t>(p_values.size()), rowOffsets, NULL,
        columnIndices, values, CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
        CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO),
      "cudssMatrixCreateCsr");
    checkCUDSS(
      cudssMatrixCreateDn(
        p_rhsMatrix.address(), p_rows, 1, p_rows, rhs, CUDSS_R_64F,
        CUDSS_LAYOUT_COL_MAJOR),
      "cudssMatrixCreateDn(rhs)");
    checkCUDSS(
      cudssMatrixCreateDn(
        p_solutionMatrix.address(), p_rows, 1, p_rows, solution,
        CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR),
      "cudssMatrixCreateDn(solution)");
  }

  bool matchesPattern(const HostSystem& system) const
  {
    return p_rows == system.rows &&
           p_rowOffsets == system.rowOffsets &&
           p_columnIndices == system.columnIndices;
  }

  bool matchesValues(const HostSystem& system) const
  {
    return p_values == system.values;
  }

  void updateValues(const HostSystem& system, CudssRuntime& runtime)
  {
    if (!matchesPattern(system)) {
      fail("attempted to update values for a different CSR pattern");
    }
    std::copy(system.values.begin(), system.values.end(), p_values.begin());
    if (p_mode == CUDSS_DEVICE_MODE) {
      p_deviceValues->copyFrom(
        &p_values[0], p_values.size(), runtime.stream());
      runtime.synchronize();
    }
  }

  void prepareRightHandSide(const HostSystem& system, CudssRuntime& runtime)
  {
    std::copy(system.rhs.begin(), system.rhs.end(), p_rhs.begin());
    std::fill(p_solution.begin(), p_solution.end(), 0.0);
    if (p_mode == CUDSS_DEVICE_MODE) {
      p_deviceRhs->copyFrom(&p_rhs[0], p_rhs.size(), runtime.stream());
      p_deviceSolution->zero(runtime.stream());
      runtime.synchronize();
    }
  }

  std::vector<double> captureSolution(CudssRuntime& runtime)
  {
    if (p_mode == CUDSS_DEVICE_MODE) {
      p_deviceSolution->copyTo(
        &p_solution[0], p_solution.size(), runtime.stream());
      runtime.synchronize();
    }
    return p_solution;
  }

  const CudssData& data(void) const
  {
    return p_data;
  }

  cudssMatrix_t matrix(void) const
  {
    return p_matrix.get();
  }

  cudssMatrix_t rhsMatrix(void) const
  {
    return p_rhsMatrix.get();
  }

  cudssMatrix_t solutionMatrix(void) const
  {
    return p_solutionMatrix.get();
  }

private:
  PatternState(const PatternState&);
  PatternState& operator=(const PatternState&);

  std::int64_t p_rows;
  CUDSSExecutionMode p_mode;
  std::vector<std::int32_t> p_rowOffsets;
  std::vector<std::int32_t> p_columnIndices;
  std::vector<double> p_values;
  std::vector<double> p_rhs;
  std::vector<double> p_solution;
  std::unique_ptr<DeviceBuffer<std::int32_t> > p_deviceRowOffsets;
  std::unique_ptr<DeviceBuffer<std::int32_t> > p_deviceColumnIndices;
  std::unique_ptr<DeviceBuffer<double> > p_deviceValues;
  std::unique_ptr<DeviceBuffer<double> > p_deviceRhs;
  std::unique_ptr<DeviceBuffer<double> > p_deviceSolution;
  CudssMatrix p_matrix;
  CudssMatrix p_rhsMatrix;
  CudssMatrix p_solutionMatrix;
  CudssData p_data;
};

std::mutex& executionMutex(void)
{
  static std::mutex mutex;
  return mutex;
}

class CUDSSLinearSolver
  : public CUDSSLinearSolverBase
{
public:
  CUDSSLinearSolver(
    MatrixType& matrix,
    utility::Configuration::CursorPtr parentConfiguration)
    : CUDSSLinearSolverBase(matrix, parentConfiguration, true),
      p_available(false),
      p_unavailableReason("cuDSS was not configured"),
      p_runtime(),
      p_patterns()
  {}

protected:
  mutable bool p_available;
  mutable std::string p_unavailableReason;
  mutable std::unique_ptr<CudssRuntime> p_runtime;
  mutable std::vector<std::unique_ptr<PatternState> > p_patterns;

  void p_configureBackend(void)
  {
    if (sizeof(RealType) != sizeof(double)) {
      p_disable("GridPACK RealType must be double");
      return;
    }
    if (sizeof(PetscInt) != sizeof(std::int32_t)) {
      p_disable("PETSc must use 32-bit indices");
      return;
    }
#if defined(PETSC_USE_COMPLEX)
    p_disable("PETSc must use real scalar storage");
    return;
#endif
    if (this->processor_size() != 1) {
      p_disable("matrix communicator must contain exactly one process");
      return;
    }
    int worldSize = 0;
    int worldRank = -1;
    if (MPI_Comm_size(MPI_COMM_WORLD, &worldSize) != MPI_SUCCESS ||
        MPI_Comm_rank(MPI_COMM_WORLD, &worldRank) != MPI_SUCCESS) {
      p_disable("failed to inspect MPI_COMM_WORLD topology");
      return;
    }
    if (worldSize != 1 || worldRank != 0 ||
        this->communicator().worldRank() != 0) {
      p_disable(
        "scalar cuDSS requires MPI_COMM_WORLD size 1 and world rank 0");
      return;
    }

    try {
      int deviceCount = 0;
      checkCUDA(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
      if (p_device >= deviceCount) {
        std::ostringstream message;
        message << "CUDSSDevice " << p_device
                << " is unavailable; detected " << deviceCount
                << " CUDA device(s)";
        p_disable(message.str());
        return;
      }
      checkCUDA(cudaSetDevice(p_device), "cudaSetDevice");
      checkCUDA(cudaFree(NULL), "cudaFree(NULL)");

      p_runtime.reset(new CudssRuntime(p_mode));
      p_available = true;
      p_unavailableReason.clear();
      p_statistics.deviceOwner = true;
      p_statistics.ownerWorldRank = 0;
      p_statistics.device = p_device;
    } catch (const std::exception& error) {
      p_disable(error.what());
    }
  }

  void p_solveImpl(MatrixType& matrix, const VectorType& b,
                   VectorType& x) const
  {
    p_solveCUDSS(matrix, b, x, false);
  }

  void p_resolveImpl(const VectorType& b, VectorType& x) const
  {
    p_solveCUDSS(this->p_matrix, b, x, true);
  }

private:
  void p_disable(const std::string& reason)
  {
    p_available = false;
    p_unavailableReason = reason;
    p_patterns.clear();
    p_runtime.reset();
    p_statistics.deviceOwner = false;
    p_statistics.ownerWorldRank = -1;
    p_statistics.device = -1;
    p_unavailable(reason);
  }

  PatternState *p_findPattern(const HostSystem& system) const
  {
    for (std::vector<std::unique_ptr<PatternState> >::iterator state =
           p_patterns.begin(); state != p_patterns.end(); ++state) {
      if ((*state)->matchesPattern(system)) {
        return state->get();
      }
    }
    return NULL;
  }

  void p_solveCUDSS(MatrixType& matrix, const VectorType& b,
                    VectorType& x, bool resolve) const
  {
    if (!p_available) {
      p_fallbackSolve(b, x, resolve, p_unavailableReason);
      return;
    }

    try {
      std::lock_guard<std::mutex> lock(executionMutex());
      checkCUDA(cudaSetDevice(p_device), "cudaSetDevice");

      const HostSystem system = extractHostSystem(matrix, b);
      PatternState *state = p_findPattern(system);
      if (state == NULL) {
        ++p_statistics.cacheMisses;
        std::unique_ptr<PatternState> newState(
          new PatternState(system, p_mode, *p_runtime));
        state = newState.get();
        p_patterns.push_back(std::move(newState));

        p_runtime->execute(
          CUDSS_PHASE_ANALYSIS, state->data(), state->matrix(),
          state->solutionMatrix(), state->rhsMatrix(), "analysis");
        ++p_statistics.analyses;

        p_runtime->execute(
          CUDSS_PHASE_FACTORIZATION, state->data(), state->matrix(),
          state->solutionMatrix(), state->rhsMatrix(), "factorization");
        ++p_statistics.factorizations;
      } else {
        ++p_statistics.cacheHits;
        if (!state->matchesValues(system)) {
          state->updateValues(system, *p_runtime);
          p_runtime->execute(
            CUDSS_PHASE_REFACTORIZATION, state->data(), state->matrix(),
            state->solutionMatrix(), state->rhsMatrix(), "refactorization");
          ++p_statistics.refactorizations;
        }
      }

      state->prepareRightHandSide(system, *p_runtime);
      p_runtime->execute(
        CUDSS_PHASE_SOLVE, state->data(), state->matrix(),
        state->solutionMatrix(), state->rhsMatrix(), "solve");
      ++p_statistics.solves;

      const std::vector<double> solution =
        state->captureSolution(*p_runtime);
      const double residual = scaledResidual(system, solution);
      p_statistics.lastScaledResidual = residual;
      if (!std::isfinite(residual) || residual > p_residualTolerance) {
        std::ostringstream message;
        message << "scaled residual " << residual
                << " exceeds CUDSSResidualTolerance "
                << p_residualTolerance;
        fail(message.str());
      }

      writeSolution(solution, matrix, x);
    } catch (const std::exception& error) {
      p_available = false;
      p_unavailableReason = error.what();
      if (p_strict) {
        throw;
      }
      p_statistics.deviceOwner = false;
      p_fallbackSolve(b, x, resolve, error.what());
    }
  }
};

#endif

} // anonymous namespace

LinearSolverImplementation<RealType, int> *
createCUDSSLinearSolver(
  RealMatrix& matrix,
  utility::Configuration::CursorPtr parentConfiguration)
{
#if defined(GRIDPACK_HAVE_CUDSS)
  return new CUDSSLinearSolver(matrix, parentConfiguration);
#else
  return new UnavailableCUDSSLinearSolver(matrix, parentConfiguration);
#endif
}

} // namespace math
} // namespace gridpack
