/*
 * Copyright (c) 2026 Battelle Memorial Institute
 * Licensed under the modified BSD License.
 */

#include "petsc/petsc_csr_exporter.hpp"

#include "petsc/petsc_exception.hpp"
#include "petsc/petsc_matrix_extractor.hpp"
#include "petsc/petsc_vector_extractor.hpp"
#include "gridpack/utilities/exception.hpp"

#include <petscmat.h>
#include <petscvec.h>

#include <mpi.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>

namespace gridpack {
namespace math {
namespace {

const char kMagic[8] = {'G', 'P', 'C', 'S', 'R', '0', '0', '1'};
const std::uint32_t kFormatVersion = 1;
const std::uint32_t kHeaderSize = 40;
const std::uint32_t kFormatFlags = 0x00000007;
const std::uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
const std::uint64_t kFnvPrime = 1099511628211ULL;

void fail(const std::string& message)
{
  throw gridpack::Exception("CSR diagnostic: " + message);
}

void require(bool condition, const std::string& message)
{
  if (!condition) {
    fail(message);
  }
}

std::size_t checkedProduct(std::uint32_t left, std::uint32_t right,
                           const std::string& label)
{
  const std::size_t leftSize = static_cast<std::size_t>(left);
  const std::size_t rightSize = static_cast<std::size_t>(right);
  if (rightSize != 0 &&
      leftSize > std::numeric_limits<std::size_t>::max() / rightSize) {
    fail(label + " size exceeds host size_t");
  }
  return leftSize * rightSize;
}

std::uint64_t checkedAdd(std::uint64_t left, std::uint64_t right,
                         const std::string& label)
{
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    fail(label + " byte count overflows uint64");
  }
  return left + right;
}

std::uint64_t checkedMultiply(std::uint64_t left, std::uint64_t right,
                              const std::string& label)
{
  if (right != 0 &&
      left > std::numeric_limits<std::uint64_t>::max() / right) {
    fail(label + " byte count overflows uint64");
  }
  return left * right;
}

std::uint64_t serializedSize(const RealCsrSystem& system)
{
  std::uint64_t result = kHeaderSize;
  result = checkedAdd(
      result,
      checkedMultiply(static_cast<std::uint64_t>(system.rows) + 1, 4,
                      "row offsets"),
      "CSR file");
  result = checkedAdd(
      result,
      checkedMultiply(system.nonzeros, 4, "column indices"),
      "CSR file");
  result = checkedAdd(
      result,
      checkedMultiply(system.nonzeros, 8, "matrix values"),
      "CSR file");
  const std::uint64_t rightHandSideValues =
    checkedMultiply(system.rows, system.rightHandSideCount,
                    "right-hand-side values");
  result = checkedAdd(
      result,
      checkedMultiply(rightHandSideValues, 8, "right-hand-side values"),
      "CSR file");
  return result;
}

void validateSystem(const RealCsrSystem& system)
{
  require(system.rows > 0, "matrix row count must be positive");
  require(system.columns == system.rows,
          "only square matrices are supported");
  require(system.nonzeros > 0, "matrix must contain at least one nonzero");
  require(system.rightHandSideCount > 0,
          "right-hand-side count must be positive");

  const std::size_t rows = static_cast<std::size_t>(system.rows);
  const std::size_t nonzeros =
    static_cast<std::size_t>(system.nonzeros);
  const std::size_t rhsValues =
    checkedProduct(system.rows, system.rightHandSideCount,
                   "right-hand side");

  require(system.rowOffsets.size() == rows + 1,
          "CSR row-offset count does not equal rows + 1");
  require(system.columnIndices.size() == nonzeros,
          "CSR column-index count does not equal nonzeros");
  require(system.values.size() == nonzeros,
          "matrix value count does not equal nonzeros");
  require(system.rightHandSides.size() == rhsValues,
          "right-hand-side value count is inconsistent");
  require(system.rowOffsets.front() == 0,
          "CSR row offsets must start at zero");
  require(system.rowOffsets.back() == system.nonzeros,
          "final CSR row offset must equal nonzeros");

  for (std::size_t row = 0; row < rows; ++row) {
    const std::uint32_t begin = system.rowOffsets[row];
    const std::uint32_t end = system.rowOffsets[row + 1];
    require(begin <= end && end <= system.nonzeros,
            "CSR row offsets must be monotonic and bounded");

    bool havePrevious = false;
    std::uint32_t previous = 0;
    for (std::uint32_t entry = begin; entry < end; ++entry) {
      const std::uint32_t column = system.columnIndices[entry];
      require(column < system.columns,
              "CSR column index is outside matrix bounds");
      require(!havePrevious || previous < column,
              "CSR columns must be strictly increasing within each row");
      previous = column;
      havePrevious = true;
    }
  }

  for (std::size_t entry = 0; entry < system.values.size(); ++entry) {
    require(std::isfinite(system.values[entry]),
            "matrix values must be finite");
  }
  for (std::size_t entry = 0;
       entry < system.rightHandSides.size(); ++entry) {
    require(std::isfinite(system.rightHandSides[entry]),
            "right-hand-side values must be finite");
  }
}

void writeBytes(std::ostream& output, const char *bytes, std::size_t count,
                const std::string& label)
{
  output.write(bytes, static_cast<std::streamsize>(count));
  if (!output) {
    fail("failed to write " + label);
  }
}

void writeUint32(std::ostream& output, std::uint32_t value)
{
  char bytes[4];
  for (std::size_t index = 0; index < sizeof(bytes); ++index) {
    bytes[index] = static_cast<char>((value >> (8 * index)) & 0xff);
  }
  writeBytes(output, bytes, sizeof(bytes), "uint32 field");
}

void writeUint64(std::ostream& output, std::uint64_t value)
{
  char bytes[8];
  for (std::size_t index = 0; index < sizeof(bytes); ++index) {
    bytes[index] = static_cast<char>((value >> (8 * index)) & 0xff);
  }
  writeBytes(output, bytes, sizeof(bytes), "uint64 field");
}

void writeDouble(std::ostream& output, double value)
{
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  writeUint64(output, bits);
}

void readBytes(std::istream& input, char *bytes, std::size_t count,
               const std::string& label)
{
  input.read(bytes, static_cast<std::streamsize>(count));
  if (input.gcount() != static_cast<std::streamsize>(count)) {
    fail("unexpected end of file while reading " + label);
  }
}

std::uint32_t readUint32(std::istream& input)
{
  char bytes[4];
  readBytes(input, bytes, sizeof(bytes), "uint32 field");
  std::uint32_t result = 0;
  for (std::size_t index = 0; index < sizeof(bytes); ++index) {
    result |= static_cast<std::uint32_t>(
      static_cast<unsigned char>(bytes[index])) << (8 * index);
  }
  return result;
}

std::uint64_t readUint64(std::istream& input)
{
  char bytes[8];
  readBytes(input, bytes, sizeof(bytes), "uint64 field");
  std::uint64_t result = 0;
  for (std::size_t index = 0; index < sizeof(bytes); ++index) {
    result |= static_cast<std::uint64_t>(
      static_cast<unsigned char>(bytes[index])) << (8 * index);
  }
  return result;
}

double readDouble(std::istream& input)
{
  const std::uint64_t bits = readUint64(input);
  double result = 0.0;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

void checkPetsc(PetscErrorCode error)
{
  if (error != 0) {
    throw PETScException(error);
  }
}

class MatRowIJView
{
public:
  explicit MatRowIJView(Mat matrix)
    : p_matrix(matrix),
      p_rows(0),
      p_rowOffsets(NULL),
      p_columnIndices(NULL),
      p_done(PETSC_FALSE),
      p_active(false)
  {
    checkPetsc(MatGetRowIJ(p_matrix, 0, PETSC_FALSE, PETSC_FALSE,
                          &p_rows, &p_rowOffsets, &p_columnIndices, &p_done));
    p_active = true;
    require(p_done == PETSC_TRUE,
            "MatGetRowIJ did not provide scalar CSR storage");
  }

  ~MatRowIJView()
  {
    if (p_active) {
      MatRestoreRowIJ(p_matrix, 0, PETSC_FALSE, PETSC_FALSE,
                      &p_rows, &p_rowOffsets, &p_columnIndices, &p_done);
    }
  }

  MatRowIJView(const MatRowIJView&) = delete;
  MatRowIJView& operator=(const MatRowIJView&) = delete;

  PetscInt rows() const
  {
    return p_rows;
  }

  const PetscInt *rowOffsets() const
  {
    return p_rowOffsets;
  }

  const PetscInt *columnIndices() const
  {
    return p_columnIndices;
  }

  void restore()
  {
    if (p_active) {
      p_active = false;
      checkPetsc(MatRestoreRowIJ(
          p_matrix, 0, PETSC_FALSE, PETSC_FALSE,
          &p_rows, &p_rowOffsets, &p_columnIndices, &p_done));
    }
  }

private:
  Mat p_matrix;
  PetscInt p_rows;
  const PetscInt *p_rowOffsets;
  const PetscInt *p_columnIndices;
  PetscBool p_done;
  bool p_active;
};

class MatValuesView
{
public:
  explicit MatValuesView(Mat matrix)
    : p_matrix(matrix), p_values(NULL), p_active(false)
  {
    checkPetsc(MatSeqAIJGetArrayRead(p_matrix, &p_values));
    p_active = true;
  }

  ~MatValuesView()
  {
    if (p_active) {
      MatSeqAIJRestoreArrayRead(p_matrix, &p_values);
    }
  }

  MatValuesView(const MatValuesView&) = delete;
  MatValuesView& operator=(const MatValuesView&) = delete;

  const PetscScalar *values() const
  {
    return p_values;
  }

  void restore()
  {
    if (p_active) {
      p_active = false;
      checkPetsc(MatSeqAIJRestoreArrayRead(p_matrix, &p_values));
    }
  }

private:
  Mat p_matrix;
  const PetscScalar *p_values;
  bool p_active;
};

class VecValuesView
{
public:
  explicit VecValuesView(Vec vector)
    : p_vector(vector), p_values(NULL), p_active(false)
  {
    checkPetsc(VecGetArrayRead(p_vector, &p_values));
    p_active = true;
  }

  ~VecValuesView()
  {
    if (p_active) {
      VecRestoreArrayRead(p_vector, &p_values);
    }
  }

  VecValuesView(const VecValuesView&) = delete;
  VecValuesView& operator=(const VecValuesView&) = delete;

  const PetscScalar *values() const
  {
    return p_values;
  }

  void restore()
  {
    if (p_active) {
      p_active = false;
      checkPetsc(VecRestoreArrayRead(p_vector, &p_values));
    }
  }

private:
  Vec p_vector;
  const PetscScalar *p_values;
  bool p_active;
};

void hashByte(std::uint64_t& hash, unsigned char value)
{
  hash ^= value;
  hash *= kFnvPrime;
}

void hashUint32(std::uint64_t& hash, std::uint32_t value)
{
  for (std::size_t index = 0; index < sizeof(value); ++index) {
    hashByte(hash, static_cast<unsigned char>(
        (value >> (8 * index)) & 0xff));
  }
}

void hashDouble(std::uint64_t& hash, double value)
{
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  for (std::size_t index = 0; index < sizeof(bits); ++index) {
    hashByte(hash, static_cast<unsigned char>(
        (bits >> (8 * index)) & 0xff));
  }
}

std::uint64_t patternHash(const RealCsrSystem& system)
{
  std::uint64_t hash = kFnvOffsetBasis;
  hashUint32(hash, system.rows);
  hashUint32(hash, system.columns);
  hashUint32(hash, system.nonzeros);
  for (std::size_t index = 0; index < system.rowOffsets.size(); ++index) {
    hashUint32(hash, system.rowOffsets[index]);
  }
  for (std::size_t index = 0;
       index < system.columnIndices.size(); ++index) {
    hashUint32(hash, system.columnIndices[index]);
  }
  return hash;
}

std::uint64_t numericHash(const RealCsrSystem& system)
{
  std::uint64_t hash = kFnvOffsetBasis;
  for (std::size_t index = 0; index < system.values.size(); ++index) {
    hashDouble(hash, system.values[index]);
  }
  for (std::size_t index = 0;
       index < system.rightHandSides.size(); ++index) {
    hashDouble(hash, system.rightHandSides[index]);
  }
  return hash;
}

std::string hexadecimal(std::uint64_t value)
{
  std::ostringstream output;
  output << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
  return output.str();
}

void ensureDirectory(const std::string& path)
{
  require(!path.empty(), "export directory cannot be empty");

  std::string current;
  std::size_t offset = 0;
  if (path[0] == '/') {
    current = "/";
    offset = 1;
  }

  while (offset <= path.size()) {
    const std::size_t separator = path.find('/', offset);
    const std::string part =
      path.substr(offset, separator == std::string::npos
                          ? std::string::npos : separator - offset);
    if (!part.empty()) {
      if (!current.empty() && current[current.size() - 1] != '/') {
        current += "/";
      }
      current += part;

      if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
        fail("cannot create export directory '" + current + "': " +
             std::strerror(errno));
      }
      struct stat status;
      if (::stat(current.c_str(), &status) != 0 ||
          !S_ISDIR(status.st_mode)) {
        fail("export path component is not a directory: '" + current + "'");
      }
    }
    if (separator == std::string::npos) {
      break;
    }
    offset = separator + 1;
  }
}

std::string joinPath(const std::string& directory,
                     const std::string& filename)
{
  if (!directory.empty() &&
      directory[directory.size() - 1] == '/') {
    return directory + filename;
  }
  return directory + "/" + filename;
}

std::string safeFilename(std::string value)
{
  if (value.empty()) {
    value = "unnamed";
  }
  for (std::size_t index = 0; index < value.size(); ++index) {
    const unsigned char character =
      static_cast<unsigned char>(value[index]);
    if (!std::isalnum(character) && value[index] != '-' &&
        value[index] != '_' && value[index] != '.') {
      value[index] = '_';
    }
  }
  if (value.size() > 64) {
    value.resize(64);
  }
  return value;
}

std::string csvField(const std::string& value)
{
  std::string result = "\"";
  for (std::size_t index = 0; index < value.size(); ++index) {
    const char character = value[index];
    if (character == '"') {
      result += "\"\"";
    } else if (character == '\n' || character == '\r') {
      result += ' ';
    } else {
      result += character;
    }
  }
  result += '"';
  return result;
}

std::uint64_t parsePositiveLimit(const char *text)
{
  require(text != NULL && text[0] != '\0',
          "GRIDPACK_PF_CSR_EXPORT_LIMIT cannot be empty");
  for (const char *character = text; character[0] != '\0'; ++character) {
    require(std::isdigit(static_cast<unsigned char>(character[0])),
            "GRIDPACK_PF_CSR_EXPORT_LIMIT must contain only decimal digits");
  }
  errno = 0;
  char *end = NULL;
  const unsigned long long parsed = std::strtoull(text, &end, 10);
  require(errno == 0 && end != text && end[0] == '\0' && parsed > 0,
          "GRIDPACK_PF_CSR_EXPORT_LIMIT must be a positive integer");
  return static_cast<std::uint64_t>(parsed);
}

struct CaptureConfig
{
  bool enabled;
  std::string directory;
  std::string filter;
  bool hasLimit;
  std::uint64_t limit;

  CaptureConfig()
    : enabled(false), hasLimit(false), limit(0)
  {
    const char *directoryValue =
      std::getenv("GRIDPACK_PF_CSR_EXPORT_DIR");
    if (directoryValue == NULL || directoryValue[0] == '\0') {
      return;
    }

    enabled = true;
    directory = directoryValue;
    const char *filterValue =
      std::getenv("GRIDPACK_PF_CSR_EXPORT_FILTER");
    if (filterValue != NULL && filterValue[0] != '\0') {
      filter = filterValue;
    }
    const char *limitValue =
      std::getenv("GRIDPACK_PF_CSR_EXPORT_LIMIT");
    if (limitValue != NULL) {
      limit = parsePositiveLimit(limitValue);
      hasLimit = true;
    }
    require(!filter.empty() || hasLimit,
            "set GRIDPACK_PF_CSR_EXPORT_FILTER and/or "
            "GRIDPACK_PF_CSR_EXPORT_LIMIT when enabling capture");
    ensureDirectory(directory);
  }
};

class CaptureState
{
public:
  CaptureState()
    : p_captured(0), p_worldRank(0), p_processId(::getpid())
  {
    if (!p_config.enabled) {
      return;
    }
    int initialized = 0;
    require(MPI_Initialized(&initialized) == MPI_SUCCESS,
            "MPI_Initialized failed");
    if (initialized) {
      require(MPI_Comm_rank(MPI_COMM_WORLD, &p_worldRank) == MPI_SUCCESS,
              "MPI_Comm_rank failed");
    }
  }

  bool enabled() const
  {
    return p_config.enabled;
  }

  bool reserve(const std::string& caseName, std::uint64_t& sequence)
  {
    if (!p_config.filter.empty() &&
        caseName.find(p_config.filter) == std::string::npos) {
      return false;
    }

    std::lock_guard<std::mutex> lock(p_mutex);
    if (p_config.hasLimit && p_captured >= p_config.limit) {
      return false;
    }
    ++p_captured;
    sequence = p_captured;
    return true;
  }

  std::string dataFilename(const PetscCsrExportContext& context,
                           std::uint64_t sequence) const
  {
    std::ostringstream filename;
    filename << "pfcsr_r" << std::setw(6) << std::setfill('0') << p_worldRank
             << "_p" << p_processId
             << "_s" << std::setw(6) << sequence
             << "_a" << std::setw(3) << context.areaInterchangePass
             << "_c" << std::setw(3) << context.controllerPass
             << "_l" << std::setw(3) << context.linearSolveOrdinal
             << "_" << safeFilename(context.caseName) << ".gpcsr";
    return filename.str();
  }

  std::string dataPath(const std::string& filename) const
  {
    return joinPath(p_config.directory, filename);
  }

  void appendManifest(const std::string& filename,
                      std::uint64_t sequence,
                      const PetscCsrExportContext& context,
                      const RealCsrSystem& system)
  {
    std::lock_guard<std::mutex> lock(p_mutex);
    std::ostringstream manifestName;
    manifestName << "manifest_r" << std::setw(6) << std::setfill('0')
                 << p_worldRank << "_p" << p_processId << ".csv";
    const std::string path =
      joinPath(p_config.directory, manifestName.str());

    std::ifstream existing(path.c_str(), std::ios::binary | std::ios::ate);
    const bool needsHeader =
      !existing || existing.tellg() == std::streampos(0);
    existing.close();

    std::ofstream manifest(path.c_str(), std::ios::out | std::ios::app);
    require(static_cast<bool>(manifest),
            "cannot open manifest '" + path + "'");
    if (needsHeader) {
      manifest
        << "format_version,file,world_rank,process_id,capture_sequence,"
        << "case_name,case_kind,capture_stage,convergence_status,"
        << "area_pass,controller_pass,newton_iteration,"
        << "linear_solve_ordinal,rows,columns,nonzeros,pattern_hash,"
        << "numeric_hash,qlim_enabled,switched_shunt_enabled,ltc_enabled,"
        << "area_interchange_enabled\n";
    }
    manifest
      << kFormatVersion << ","
      << csvField(filename) << ","
      << p_worldRank << ","
      << p_processId << ","
      << sequence << ","
      << csvField(context.caseName) << ","
      << (context.baseCase ? "base_case" : "contingency") << ","
      << "pre_linear_solve,not_evaluated,"
      << context.areaInterchangePass << ","
      << context.controllerPass << ","
      << context.newtonIteration << ","
      << context.linearSolveOrdinal << ","
      << system.rows << ","
      << system.columns << ","
      << system.nonzeros << ","
      << hexadecimal(patternHash(system)) << ","
      << hexadecimal(numericHash(system)) << ","
      << (context.qlimEnabled ? 1 : 0) << ","
      << (context.switchedShuntEnabled ? 1 : 0) << ","
      << (context.ltcEnabled ? 1 : 0) << ","
      << (context.areaInterchangeEnabled ? 1 : 0) << "\n";
    manifest.close();
    require(static_cast<bool>(manifest),
            "failed to append manifest '" + path + "'");
  }

private:
  CaptureConfig p_config;
  std::uint64_t p_captured;
  int p_worldRank;
  pid_t p_processId;
  std::mutex p_mutex;
};

CaptureState& captureState()
{
  static CaptureState state;
  return state;
}

} // namespace

void writeRealCsrSystem(const std::string& path,
                        const RealCsrSystem& system)
{
  require(sizeof(double) == sizeof(std::uint64_t) &&
          std::numeric_limits<double>::is_iec559,
          "version 1 requires IEEE-754 binary64 doubles");
  validateSystem(system);

  std::ifstream existing(path.c_str(), std::ios::in | std::ios::binary);
  require(!existing, "refusing to overwrite output file '" + path + "'");
  existing.close();

  std::ofstream output(path.c_str(),
                       std::ios::out | std::ios::binary | std::ios::trunc);
  require(static_cast<bool>(output),
          "cannot open output file '" + path + "'");

  writeBytes(output, kMagic, sizeof(kMagic), "format magic");
  writeUint32(output, kFormatVersion);
  writeUint32(output, kHeaderSize);
  writeUint32(output, kFormatFlags);
  writeUint32(output, system.rows);
  writeUint32(output, system.columns);
  writeUint32(output, system.nonzeros);
  writeUint32(output, system.rightHandSideCount);
  writeUint32(output, 0);

  for (std::size_t index = 0; index < system.rowOffsets.size(); ++index) {
    writeUint32(output, system.rowOffsets[index]);
  }
  for (std::size_t index = 0;
       index < system.columnIndices.size(); ++index) {
    writeUint32(output, system.columnIndices[index]);
  }
  for (std::size_t index = 0; index < system.values.size(); ++index) {
    writeDouble(output, system.values[index]);
  }
  for (std::size_t index = 0;
       index < system.rightHandSides.size(); ++index) {
    writeDouble(output, system.rightHandSides[index]);
  }
  output.close();
  require(static_cast<bool>(output),
          "failed to finish output file '" + path + "'");
}

RealCsrSystem readRealCsrSystem(const std::string& path)
{
  require(sizeof(double) == sizeof(std::uint64_t) &&
          std::numeric_limits<double>::is_iec559,
          "version 1 requires IEEE-754 binary64 doubles");

  std::ifstream input(path.c_str(), std::ios::in | std::ios::binary);
  require(static_cast<bool>(input),
          "cannot open input file '" + path + "'");

  char magic[sizeof(kMagic)];
  readBytes(input, magic, sizeof(magic), "format magic");
  require(std::memcmp(magic, kMagic, sizeof(kMagic)) == 0,
          "invalid format magic");
  require(readUint32(input) == kFormatVersion,
          "unsupported CSR format version");
  require(readUint32(input) == kHeaderSize,
          "unsupported CSR header size");
  require(readUint32(input) == kFormatFlags,
          "unsupported CSR format flags");

  RealCsrSystem system;
  system.rows = readUint32(input);
  system.columns = readUint32(input);
  system.nonzeros = readUint32(input);
  system.rightHandSideCount = readUint32(input);
  require(readUint32(input) == 0, "reserved header field must be zero");

  const std::uint64_t expectedSize = serializedSize(system);
  require(expectedSize <= static_cast<std::uint64_t>(
              std::numeric_limits<std::streamoff>::max()),
          "serialized file size exceeds streamoff");
  input.seekg(0, std::ios::end);
  const std::streamoff actualSize = input.tellg();
  require(actualSize >= 0, "cannot determine input file size");
  require(static_cast<std::uint64_t>(actualSize) == expectedSize,
          "file size does not match the declared CSR payload");
  input.seekg(static_cast<std::streamoff>(kHeaderSize), std::ios::beg);
  require(static_cast<bool>(input), "cannot seek to CSR payload");

  require(static_cast<std::uint64_t>(system.rows) + 1 <=
          std::numeric_limits<std::size_t>::max(),
          "row-offset count exceeds host size_t");
  const std::size_t rowOffsetCount =
    static_cast<std::size_t>(system.rows) + 1;
  const std::size_t nonzeros =
    static_cast<std::size_t>(system.nonzeros);
  const std::size_t rhsValues =
    checkedProduct(system.rows, system.rightHandSideCount,
                   "right-hand side");

  system.rowOffsets.resize(rowOffsetCount);
  system.columnIndices.resize(nonzeros);
  system.values.resize(nonzeros);
  system.rightHandSides.resize(rhsValues);
  for (std::size_t index = 0; index < rowOffsetCount; ++index) {
    system.rowOffsets[index] = readUint32(input);
  }
  for (std::size_t index = 0; index < nonzeros; ++index) {
    system.columnIndices[index] = readUint32(input);
  }
  for (std::size_t index = 0; index < nonzeros; ++index) {
    system.values[index] = readDouble(input);
  }
  for (std::size_t index = 0; index < rhsValues; ++index) {
    system.rightHandSides[index] = readDouble(input);
  }

  require(input.peek() == std::char_traits<char>::eof(),
          "unexpected trailing data after CSR payload");
  validateSystem(system);
  return system;
}

RealCsrSystem extractPetscRealCsrSystem(
    const RealMatrix& matrix,
    const RealVector& rightHandSide)
{
#if defined(PETSC_USE_COMPLEX)
  fail("complex PETSc scalar builds are not supported");
  return RealCsrSystem();
#else
  require(sizeof(PetscScalar) == sizeof(double) &&
          std::numeric_limits<PetscScalar>::is_iec559,
          "PETSc scalar type must be IEEE-754 binary64");
  require(sizeof(PetscInt) == sizeof(std::int32_t) &&
          std::numeric_limits<PetscInt>::is_integer &&
          std::numeric_limits<PetscInt>::is_signed &&
          std::numeric_limits<PetscInt>::digits == 31,
          "PETSc must use signed 32-bit indices");

  Mat petscMatrix = *PETScMatrix(matrix);
  Vec petscVector = *PETScVector(rightHandSide);

  MatType matrixType = NULL;
  checkPetsc(MatGetType(petscMatrix, &matrixType));
  require(matrixType != NULL &&
          std::strcmp(matrixType, MATSEQAIJ) == 0,
          "matrix must have exact PETSc type MATSEQAIJ");
  VecType vectorType = NULL;
  checkPetsc(VecGetType(petscVector, &vectorType));
  require(vectorType != NULL &&
          std::strcmp(vectorType, VECSEQ) == 0,
          "right-hand side must have exact PETSc type VECSEQ");

  PetscBool assembled = PETSC_FALSE;
  checkPetsc(MatAssembled(petscMatrix, &assembled));
  require(assembled == PETSC_TRUE, "matrix must be assembled");

  MPI_Comm matrixCommunicator = MPI_COMM_NULL;
  MPI_Comm vectorCommunicator = MPI_COMM_NULL;
  checkPetsc(PetscObjectGetComm(
      reinterpret_cast<PetscObject>(petscMatrix), &matrixCommunicator));
  checkPetsc(PetscObjectGetComm(
      reinterpret_cast<PetscObject>(petscVector), &vectorCommunicator));
  int matrixProcesses = 0;
  int vectorProcesses = 0;
  require(MPI_Comm_size(matrixCommunicator, &matrixProcesses) == MPI_SUCCESS &&
          matrixProcesses == 1,
          "matrix communicator must contain exactly one process");
  require(MPI_Comm_size(vectorCommunicator, &vectorProcesses) == MPI_SUCCESS &&
          vectorProcesses == 1,
          "right-hand-side communicator must contain exactly one process");

  PetscInt rows = 0;
  PetscInt columns = 0;
  PetscInt localRows = 0;
  PetscInt localColumns = 0;
  PetscInt vectorSize = 0;
  PetscInt localVectorSize = 0;
  checkPetsc(MatGetSize(petscMatrix, &rows, &columns));
  checkPetsc(MatGetLocalSize(
      petscMatrix, &localRows, &localColumns));
  checkPetsc(VecGetSize(petscVector, &vectorSize));
  checkPetsc(VecGetLocalSize(petscVector, &localVectorSize));
  require(rows > 0 && columns == rows,
          "matrix must be nonempty and square");
  require(localRows == rows && localColumns == columns,
          "all matrix rows and columns must be local");
  require(vectorSize == rows && localVectorSize == rows,
          "right-hand-side size must equal the local matrix row count");

  MatRowIJView rowView(petscMatrix);
  require(rowView.rows() == rows,
          "MatGetRowIJ row count does not match MatGetSize");
  const PetscInt petscNonzeros = rowView.rowOffsets()[rows];
  require(petscNonzeros > 0,
          "matrix must contain at least one stored nonzero");

  MatValuesView matrixValues(petscMatrix);
  VecValuesView vectorValues(petscVector);

  RealCsrSystem system;
  system.rows = static_cast<std::uint32_t>(rows);
  system.columns = static_cast<std::uint32_t>(columns);
  system.nonzeros = static_cast<std::uint32_t>(petscNonzeros);
  system.rightHandSideCount = 1;
  system.rowOffsets.resize(static_cast<std::size_t>(rows) + 1);
  system.columnIndices.resize(
      static_cast<std::size_t>(petscNonzeros));
  system.values.resize(static_cast<std::size_t>(petscNonzeros));
  system.rightHandSides.resize(static_cast<std::size_t>(rows));

  for (PetscInt row = 0; row <= rows; ++row) {
    const PetscInt offset = rowView.rowOffsets()[row];
    require(offset >= 0 && offset <= petscNonzeros,
            "PETSc CSR row offset is outside the nonzero range");
    system.rowOffsets[static_cast<std::size_t>(row)] =
      static_cast<std::uint32_t>(offset);
  }
  for (PetscInt entry = 0; entry < petscNonzeros; ++entry) {
    const PetscInt column = rowView.columnIndices()[entry];
    require(column >= 0 && column < columns,
            "PETSc CSR column index is outside matrix bounds");
    system.columnIndices[static_cast<std::size_t>(entry)] =
      static_cast<std::uint32_t>(column);
    system.values[static_cast<std::size_t>(entry)] =
      static_cast<double>(matrixValues.values()[entry]);
  }
  for (PetscInt row = 0; row < rows; ++row) {
    system.rightHandSides[static_cast<std::size_t>(row)] =
      static_cast<double>(vectorValues.values()[row]);
  }

  vectorValues.restore();
  matrixValues.restore();
  rowView.restore();
  validateSystem(system);
  return system;
#endif
}

void exportPetscRealCsrSystemIfEnabled(
    const RealMatrix& matrix,
    const RealVector& rightHandSide,
    const PetscCsrExportContext& context)
{
  CaptureState& state = captureState();
  if (!state.enabled()) {
    return;
  }

  std::uint64_t sequence = 0;
  if (!state.reserve(context.caseName, sequence)) {
    return;
  }

  const RealCsrSystem system =
    extractPetscRealCsrSystem(matrix, rightHandSide);
  const std::string filename =
    state.dataFilename(context, sequence);
  writeRealCsrSystem(state.dataPath(filename), system);
  state.appendManifest(filename, sequence, context, system);
}

} // namespace math
} // namespace gridpack
