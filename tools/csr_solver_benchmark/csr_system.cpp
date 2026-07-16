/*
 * Copyright (c) 2026 Battelle Memorial Institute
 * Licensed under the modified BSD License.
 */

#include "csr_system.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gridpack {
namespace benchmark {
namespace {

constexpr const char *kHeader = "GRIDPACK_CSR_SYSTEM_V1";
constexpr const char *kFooter = "END_GRIDPACK_CSR_SYSTEM_V1";
constexpr std::array<char, 8> kBinaryMagic = {
    'G', 'P', 'C', 'S', 'R', '0', '0', '1'};
constexpr std::uint32_t kBinaryVersion = 1;
constexpr std::uint32_t kBinaryHeaderSize = 40;
constexpr std::uint32_t kBinaryFlags = 0x00000007U;

void readExact(std::istream &input, void *destination, std::size_t size,
               const std::string &description)
{
  if (size > static_cast<std::size_t>(
                 std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error(description + " exceeds streamsize");
  }
  input.read(static_cast<char *>(destination),
             static_cast<std::streamsize>(size));
  if (!input) {
    throw std::runtime_error("Unexpected end of file while reading " +
                             description);
  }
}

std::uint32_t readLittleU32(std::istream &input,
                            const std::string &description)
{
  std::array<unsigned char, 4> bytes{};
  readExact(input, bytes.data(), bytes.size(), description);
  return static_cast<std::uint32_t>(bytes[0]) |
         (static_cast<std::uint32_t>(bytes[1]) << 8U) |
         (static_cast<std::uint32_t>(bytes[2]) << 16U) |
         (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

std::uint64_t readLittleU64(std::istream &input,
                            const std::string &description)
{
  std::array<unsigned char, 8> bytes{};
  readExact(input, bytes.data(), bytes.size(), description);
  std::uint64_t value = 0;
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    value |= static_cast<std::uint64_t>(bytes[index])
             << static_cast<unsigned int>(8U * index);
  }
  return value;
}

double readLittleDouble(std::istream &input, const std::string &description)
{
  static_assert(sizeof(double) == sizeof(std::uint64_t),
                "The GPCSR reader requires binary64 double");
  static_assert(std::numeric_limits<double>::is_iec559,
                "The GPCSR reader requires IEEE-754 double");
  const std::uint64_t bits = readLittleU64(input, description);
  double value = 0.0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

void requireLabel(std::istream &input, const std::string &expected)
{
  std::string actual;
  if (!(input >> actual)) {
    throw std::runtime_error("Expected field '" + expected +
                             "', but the file ended");
  }
  if (actual != expected) {
    throw std::runtime_error("Expected field '" + expected + "', found '" +
                             actual + "'");
  }
}

std::string readQuoted(std::istream &input, const std::string &label)
{
  requireLabel(input, label);
  std::string value;
  if (!(input >> std::quoted(value))) {
    throw std::runtime_error("Invalid quoted string for field '" + label + "'");
  }
  return value;
}

std::int64_t readInteger(std::istream &input, const std::string &label)
{
  requireLabel(input, label);
  std::int64_t value = 0;
  if (!(input >> value)) {
    throw std::runtime_error("Invalid integer for field '" + label + "'");
  }
  return value;
}

std::size_t checkedSize(std::int64_t value, const std::string &label)
{
  if (value < 0) {
    throw std::runtime_error("Field '" + label + "' cannot be negative");
  }
  const auto as_unsigned = static_cast<std::uint64_t>(value);
  if (as_unsigned > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("Field '" + label + "' exceeds host size_t");
  }
  return static_cast<std::size_t>(as_unsigned);
}

std::size_t checkedProduct(std::int64_t left, std::int64_t right,
                           const std::string &label)
{
  const std::size_t left_size = checkedSize(left, label);
  const std::size_t right_size = checkedSize(right, label);
  if (right_size != 0 &&
      left_size > std::numeric_limits<std::size_t>::max() / right_size) {
    throw std::runtime_error("Field '" + label + "' overflows host size_t");
  }
  return left_size * right_size;
}

template <typename T>
std::vector<T> readArray(std::istream &input, const std::string &label,
                         std::size_t expected_size)
{
  requireLabel(input, label);
  std::int64_t serialized_size = 0;
  if (!(input >> serialized_size)) {
    throw std::runtime_error("Invalid element count for section '" + label +
                             "'");
  }
  if (checkedSize(serialized_size, label) != expected_size) {
    throw std::runtime_error("Section '" + label +
                             "' has an unexpected element count");
  }

  std::vector<T> values(expected_size);
  for (std::size_t index = 0; index < expected_size; ++index) {
    if (!(input >> values[index])) {
      throw std::runtime_error("Invalid element " + std::to_string(index) +
                               " in section '" + label + "'");
    }
  }
  return values;
}

void validate(const CsrSystem &system)
{
  if (system.nrows <= 0 || system.ncols <= 0) {
    throw std::runtime_error("Matrix dimensions must be positive");
  }
  if (system.nrows != system.ncols) {
    throw std::runtime_error("Only square linear systems are supported");
  }
  if (system.nnz <= 0) {
    throw std::runtime_error("The matrix must contain at least one nonzero");
  }
  if (system.rhs_count <= 0) {
    throw std::runtime_error("rhs_count must be positive");
  }

  if (system.row_offsets.front() != 0) {
    throw std::runtime_error("CSR row_offsets must start at zero");
  }
  if (system.row_offsets.back() != system.nnz) {
    throw std::runtime_error("The final CSR row offset must equal nnz");
  }

  for (std::int64_t row = 0; row < system.nrows; ++row) {
    const std::size_t row_index = checkedSize(row, "row index");
    const std::int64_t begin = system.row_offsets[row_index];
    const std::int64_t end = system.row_offsets[row_index + 1];
    if (begin < 0 || end < begin || end > system.nnz) {
      throw std::runtime_error("CSR row offsets are not monotonic and bounded");
    }

    std::int64_t previous_column = -1;
    for (std::int64_t entry = begin; entry < end; ++entry) {
      const std::size_t entry_index = checkedSize(entry, "entry index");
      const std::int64_t column = system.column_indices[entry_index];
      if (column < 0 || column >= system.ncols) {
        throw std::runtime_error("CSR column index is outside matrix bounds");
      }
      if (column <= previous_column) {
        throw std::runtime_error(
            "CSR column indices must be strictly increasing within each row");
      }
      previous_column = column;
    }
  }

  const auto finite = [](double value) { return std::isfinite(value); };
  if (!std::all_of(system.values.begin(), system.values.end(), finite)) {
    throw std::runtime_error("Matrix values must all be finite");
  }
  if (!std::all_of(system.rhs.begin(), system.rhs.end(), finite)) {
    throw std::runtime_error("Right-hand-side values must all be finite");
  }
}

long double squaredNorm(const double *values, std::size_t size)
{
  long double sum = 0.0L;
  for (std::size_t index = 0; index < size; ++index) {
    const long double value = static_cast<long double>(values[index]);
    sum += value * value;
  }
  return sum;
}

}  // namespace

CsrSystem readTextCsrSystem(const std::string &path)
{
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Unable to open CSR system file '" + path + "'");
  }

  requireLabel(input, kHeader);
  CsrSystem system;
  system.name = readQuoted(input, "name");
  system.contingency = readQuoted(input, "contingency");
  system.newton_iteration = readInteger(input, "newton_iteration");
  system.controller_iteration = readInteger(input, "controller_iteration");
  system.convergence_status = readQuoted(input, "convergence_status");
  system.state = readQuoted(input, "state");
  system.nrows = readInteger(input, "nrows");
  system.ncols = readInteger(input, "ncols");
  system.nnz = readInteger(input, "nnz");
  const std::int64_t index_base = readInteger(input, "index_base");
  if (index_base != 0) {
    throw std::runtime_error("Only zero-based CSR files are supported");
  }
  system.rhs_count = readInteger(input, "rhs_count");

  if (system.nrows == std::numeric_limits<std::int64_t>::max()) {
    throw std::runtime_error("nrows is too large to store nrows + 1 offsets");
  }
  const std::size_t offset_count =
      checkedSize(system.nrows + 1, "row_offsets");
  const std::size_t nonzero_count = checkedSize(system.nnz, "nnz");
  const std::size_t rhs_size =
      checkedProduct(system.nrows, system.rhs_count, "rhs");

  system.row_offsets =
      readArray<std::int64_t>(input, "row_offsets", offset_count);
  system.column_indices =
      readArray<std::int64_t>(input, "column_indices", nonzero_count);
  system.values = readArray<double>(input, "values", nonzero_count);
  system.rhs = readArray<double>(input, "rhs", rhs_size);
  requireLabel(input, kFooter);

  std::string trailing;
  if (input >> trailing) {
    throw std::runtime_error("Unexpected trailing token '" + trailing + "'");
  }

  validate(system);
  return system;
}

CsrSystem readBinaryCsrSystem(const std::string &path)
{
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Unable to open CSR system file '" + path + "'");
  }

  std::array<char, kBinaryMagic.size()> magic{};
  readExact(input, magic.data(), magic.size(), "GPCSR magic");
  if (magic != kBinaryMagic) {
    throw std::runtime_error("Invalid GPCSR magic");
  }

  const std::uint32_t version = readLittleU32(input, "GPCSR version");
  const std::uint32_t header_size =
      readLittleU32(input, "GPCSR header size");
  const std::uint32_t flags = readLittleU32(input, "GPCSR flags");
  const std::uint32_t rows = readLittleU32(input, "GPCSR row count");
  const std::uint32_t columns = readLittleU32(input, "GPCSR column count");
  const std::uint32_t nonzeros = readLittleU32(input, "GPCSR nonzero count");
  const std::uint32_t rhs_count = readLittleU32(input, "GPCSR RHS count");
  const std::uint32_t reserved = readLittleU32(input, "GPCSR reserved field");

  if (version != kBinaryVersion) {
    throw std::runtime_error("Unsupported GPCSR version " +
                             std::to_string(version));
  }
  if (header_size != kBinaryHeaderSize) {
    throw std::runtime_error("GPCSR v1 header size must be 40 bytes");
  }
  if (flags != kBinaryFlags) {
    throw std::runtime_error("GPCSR v1 flags must equal 0x00000007");
  }
  if (reserved != 0) {
    throw std::runtime_error("GPCSR v1 reserved field must be zero");
  }
  if (rows == 0 || columns == 0 || rows != columns) {
    throw std::runtime_error("GPCSR matrix must be square and nonempty");
  }
  if (nonzeros == 0) {
    throw std::runtime_error("GPCSR matrix must contain a nonzero");
  }
  if (rhs_count == 0) {
    throw std::runtime_error("GPCSR RHS count must be positive");
  }

  CsrSystem system;
  system.name = path;
  system.convergence_status = "not_recorded";
  system.nrows = static_cast<std::int64_t>(rows);
  system.ncols = static_cast<std::int64_t>(columns);
  system.nnz = static_cast<std::int64_t>(nonzeros);
  system.rhs_count = static_cast<std::int64_t>(rhs_count);

  const std::size_t offset_count =
      checkedSize(system.nrows + 1, "GPCSR row offsets");
  const std::size_t nonzero_count = checkedSize(system.nnz, "GPCSR nnz");
  const std::size_t rhs_size =
      checkedProduct(system.nrows, system.rhs_count, "GPCSR RHS");

  const auto checkedBytes = [](std::size_t count, std::size_t element_size,
                               const std::string &description) {
    if (count > std::numeric_limits<std::size_t>::max() / element_size) {
      throw std::runtime_error(description + " byte size overflows size_t");
    }
    return count * element_size;
  };
  const auto checkedAdd = [](std::size_t left, std::size_t right,
                             const std::string &description) {
    if (left > std::numeric_limits<std::size_t>::max() - right) {
      throw std::runtime_error(description + " byte size overflows size_t");
    }
    return left + right;
  };

  std::size_t expected_size = kBinaryHeaderSize;
  expected_size =
      checkedAdd(expected_size,
                 checkedBytes(offset_count, sizeof(std::uint32_t),
                              "GPCSR row offsets"),
                 "GPCSR file");
  expected_size =
      checkedAdd(expected_size,
                 checkedBytes(nonzero_count, sizeof(std::uint32_t),
                              "GPCSR column indices"),
                 "GPCSR file");
  expected_size =
      checkedAdd(expected_size,
                 checkedBytes(nonzero_count, sizeof(double), "GPCSR values"),
                 "GPCSR file");
  expected_size =
      checkedAdd(expected_size,
                 checkedBytes(rhs_size, sizeof(double), "GPCSR RHS"),
                 "GPCSR file");

  input.seekg(0, std::ios::end);
  const std::streampos serialized_size = input.tellg();
  if (serialized_size < 0) {
    throw std::runtime_error("Unable to determine GPCSR file size");
  }
  if (static_cast<std::uintmax_t>(serialized_size) !=
      static_cast<std::uintmax_t>(expected_size)) {
    throw std::runtime_error(
        "GPCSR file size does not match the header dimensions");
  }
  input.seekg(static_cast<std::streamoff>(kBinaryHeaderSize), std::ios::beg);
  if (!input) {
    throw std::runtime_error("Unable to seek to the GPCSR payload");
  }

  system.row_offsets.resize(offset_count);
  for (std::size_t index = 0; index < offset_count; ++index) {
    system.row_offsets[index] = static_cast<std::int64_t>(
        readLittleU32(input, "GPCSR row offset " + std::to_string(index)));
  }
  system.column_indices.resize(nonzero_count);
  for (std::size_t index = 0; index < nonzero_count; ++index) {
    system.column_indices[index] = static_cast<std::int64_t>(
        readLittleU32(input, "GPCSR column index " + std::to_string(index)));
  }
  system.values.resize(nonzero_count);
  for (std::size_t index = 0; index < nonzero_count; ++index) {
    system.values[index] =
        readLittleDouble(input, "GPCSR value " + std::to_string(index));
  }
  system.rhs.resize(rhs_size);
  for (std::size_t index = 0; index < rhs_size; ++index) {
    system.rhs[index] =
        readLittleDouble(input, "GPCSR RHS value " + std::to_string(index));
  }

  validate(system);
  return system;
}

CsrSystem readCsrSystem(const std::string &path)
{
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Unable to open CSR system file '" + path + "'");
  }
  std::array<char, kBinaryMagic.size()> prefix{};
  input.read(prefix.data(), static_cast<std::streamsize>(prefix.size()));
  if (input.gcount() == static_cast<std::streamsize>(prefix.size()) &&
      prefix == kBinaryMagic) {
    return readBinaryCsrSystem(path);
  }
  return readTextCsrSystem(path);
}

CscMatrix convertCsrToCsc(const CsrSystem &system)
{
  CscMatrix result;
  const std::size_t dimension = checkedSize(system.ncols, "ncols");
  const std::size_t nonzero_count = checkedSize(system.nnz, "nnz");
  result.column_offsets.assign(dimension + 1, 0);
  result.row_indices.resize(nonzero_count);
  result.values.resize(nonzero_count);

  for (const std::int64_t column : system.column_indices) {
    const std::size_t column_index = checkedSize(column, "column index");
    ++result.column_offsets[column_index + 1];
  }
  for (std::size_t column = 0; column < dimension; ++column) {
    result.column_offsets[column + 1] += result.column_offsets[column];
  }

  std::vector<std::int64_t> next = result.column_offsets;
  for (std::int64_t row = 0; row < system.nrows; ++row) {
    const std::size_t row_index = checkedSize(row, "row index");
    const std::int64_t begin = system.row_offsets[row_index];
    const std::int64_t end = system.row_offsets[row_index + 1];
    for (std::int64_t entry = begin; entry < end; ++entry) {
      const std::size_t entry_index = checkedSize(entry, "entry index");
      const std::int64_t column = system.column_indices[entry_index];
      const std::size_t column_index = checkedSize(column, "column index");
      const std::size_t destination =
          checkedSize(next[column_index]++, "CSC destination");
      result.row_indices[destination] = row;
      result.values[destination] = system.values[entry_index];
    }
  }
  return result;
}

double scaledResidual(const CsrSystem &system,
                      const std::vector<double> &solution)
{
  const std::size_t dimension = checkedSize(system.nrows, "nrows");
  const std::size_t expected =
      checkedProduct(system.nrows, system.rhs_count, "solution");
  if (solution.size() != expected) {
    throw std::runtime_error("Solution size does not match the CSR system");
  }
  if (!std::all_of(solution.begin(), solution.end(),
                   [](double value) { return std::isfinite(value); })) {
    return std::numeric_limits<double>::infinity();
  }

  const long double matrix_norm =
      std::sqrt(squaredNorm(system.values.data(), system.values.size()));
  double maximum = 0.0;
  for (std::int64_t rhs_index = 0; rhs_index < system.rhs_count; ++rhs_index) {
    const std::size_t rhs_offset =
        checkedProduct(rhs_index, system.nrows, "rhs offset");
    long double residual_squared = 0.0L;
    for (std::int64_t row = 0; row < system.nrows; ++row) {
      const std::size_t row_index = checkedSize(row, "row index");
      long double product = 0.0L;
      for (std::int64_t entry = system.row_offsets[row_index];
           entry < system.row_offsets[row_index + 1]; ++entry) {
        const std::size_t entry_index = checkedSize(entry, "entry index");
        const std::size_t column =
            checkedSize(system.column_indices[entry_index], "column index");
        product += static_cast<long double>(system.values[entry_index]) *
                   static_cast<long double>(solution[rhs_offset + column]);
      }
      const long double residual =
          product - static_cast<long double>(system.rhs[rhs_offset + row_index]);
      residual_squared += residual * residual;
    }

    const long double solution_norm =
        std::sqrt(squaredNorm(solution.data() + rhs_offset, dimension));
    const long double rhs_norm =
        std::sqrt(squaredNorm(system.rhs.data() + rhs_offset, dimension));
    const long double denominator = matrix_norm * solution_norm + rhs_norm;
    const long double residual_norm = std::sqrt(residual_squared);
    const double scaled =
        denominator == 0.0L
            ? (residual_norm == 0.0L
                   ? 0.0
                   : std::numeric_limits<double>::infinity())
            : static_cast<double>(residual_norm / denominator);
    maximum = std::max(maximum, scaled);
  }
  return maximum;
}

SolutionDifference compareSolutions(const CsrSystem &system,
                                    const std::vector<double> &reference,
                                    const std::vector<double> &candidate)
{
  const std::size_t dimension = checkedSize(system.nrows, "nrows");
  const std::size_t expected =
      checkedProduct(system.nrows, system.rhs_count, "solution");
  if (reference.size() != expected || candidate.size() != expected) {
    throw std::runtime_error("Cannot compare solutions with unexpected sizes");
  }
  const auto finite = [](double value) { return std::isfinite(value); };
  if (!std::all_of(reference.begin(), reference.end(), finite) ||
      !std::all_of(candidate.begin(), candidate.end(), finite)) {
    return {std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()};
  }

  SolutionDifference result;
  for (std::int64_t rhs_index = 0; rhs_index < system.rhs_count; ++rhs_index) {
    const std::size_t offset =
        checkedProduct(rhs_index, system.nrows, "rhs offset");
    long double difference_squared = 0.0L;
    for (std::size_t row = 0; row < dimension; ++row) {
      const long double difference =
          static_cast<long double>(candidate[offset + row]) -
          static_cast<long double>(reference[offset + row]);
      difference_squared += difference * difference;
      result.max_absolute =
          std::max(result.max_absolute,
                   std::abs(candidate[offset + row] - reference[offset + row]));
    }
    const long double reference_norm =
        std::sqrt(squaredNorm(reference.data() + offset, dimension));
    const long double difference_norm = std::sqrt(difference_squared);
    const long double denominator =
        std::max(reference_norm, std::numeric_limits<long double>::epsilon());
    result.max_relative_l2 =
        std::max(result.max_relative_l2,
                 static_cast<double>(difference_norm / denominator));
  }
  return result;
}

}  // namespace benchmark
}  // namespace gridpack
