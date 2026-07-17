/*
 * Copyright (c) 2026 Battelle Memorial Institute
 * Licensed under the modified BSD License.
 */

#include <cstdio>
#include <fstream>
#include <limits>
#include <string>

#include "gridpack/math/matrix.hpp"
#include "gridpack/math/vector.hpp"
#include "gridpack/math/petsc/petsc_csr_exporter.hpp"
#include "gridpack/parallel/communicator.hpp"
#include "gridpack/utilities/exception.hpp"

#include "test_main.cpp"

namespace {

class TemporaryFile
{
public:
  explicit TemporaryFile(const std::string& path)
    : p_path(path)
  {
    std::remove(p_path.c_str());
  }

  ~TemporaryFile()
  {
    std::remove(p_path.c_str());
  }

  TemporaryFile(const TemporaryFile&) = delete;
  TemporaryFile& operator=(const TemporaryFile&) = delete;

  const std::string& path() const
  {
    return p_path;
  }

private:
  std::string p_path;
};

} // namespace

BOOST_AUTO_TEST_SUITE(PetscCsrExporterTest)

BOOST_AUTO_TEST_CASE(extractAndRoundTrip)
{
  gridpack::parallel::Communicator world;
  BOOST_REQUIRE_EQUAL(world.size(), 1);

  gridpack::math::RealMatrix matrix(
      world, 3, 3, gridpack::math::Sparse);
  matrix.setElement(0, 0, 10.0);
  matrix.setElement(0, 2, 2.0);
  matrix.setElement(1, 0, 3.0);
  matrix.setElement(1, 1, 9.0);
  matrix.setElement(2, 1, 4.0);
  matrix.setElement(2, 2, 7.0);
  matrix.ready();

  gridpack::math::RealVector rightHandSide(world, 3);
  rightHandSide.setElement(0, 12.0);
  rightHandSide.setElement(1, 30.0);
  rightHandSide.setElement(2, 29.0);
  rightHandSide.ready();

  const gridpack::math::RealCsrSystem extracted =
    gridpack::math::extractPetscRealCsrSystem(
        matrix, rightHandSide);
  BOOST_CHECK_EQUAL(extracted.rows, 3);
  BOOST_CHECK_EQUAL(extracted.columns, 3);
  BOOST_CHECK_EQUAL(extracted.nonzeros, 6);
  BOOST_CHECK_EQUAL(extracted.rightHandSideCount, 1);

  const std::uint32_t expectedRows[] = {0, 2, 4, 6};
  const std::uint32_t expectedColumns[] = {0, 2, 0, 1, 1, 2};
  const double expectedValues[] = {10.0, 2.0, 3.0, 9.0, 4.0, 7.0};
  const double expectedRightHandSide[] = {12.0, 30.0, 29.0};
  BOOST_CHECK_EQUAL_COLLECTIONS(
      extracted.rowOffsets.begin(), extracted.rowOffsets.end(),
      expectedRows, expectedRows + 4);
  BOOST_CHECK_EQUAL_COLLECTIONS(
      extracted.columnIndices.begin(), extracted.columnIndices.end(),
      expectedColumns, expectedColumns + 6);
  BOOST_CHECK_EQUAL_COLLECTIONS(
      extracted.values.begin(), extracted.values.end(),
      expectedValues, expectedValues + 6);
  BOOST_CHECK_EQUAL_COLLECTIONS(
      extracted.rightHandSides.begin(), extracted.rightHandSides.end(),
      expectedRightHandSide, expectedRightHandSide + 3);

  TemporaryFile file("petsc_csr_exporter_test.gpcsr");
  gridpack::math::writeRealCsrSystem(file.path(), extracted);
  const gridpack::math::RealCsrSystem restored =
    gridpack::math::readRealCsrSystem(file.path());

  BOOST_CHECK_EQUAL(restored.rows, extracted.rows);
  BOOST_CHECK_EQUAL(restored.columns, extracted.columns);
  BOOST_CHECK_EQUAL(restored.nonzeros, extracted.nonzeros);
  BOOST_CHECK_EQUAL(restored.rightHandSideCount,
                    extracted.rightHandSideCount);
  BOOST_CHECK_EQUAL_COLLECTIONS(
      restored.rowOffsets.begin(), restored.rowOffsets.end(),
      extracted.rowOffsets.begin(), extracted.rowOffsets.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(
      restored.columnIndices.begin(), restored.columnIndices.end(),
      extracted.columnIndices.begin(), extracted.columnIndices.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(
      restored.values.begin(), restored.values.end(),
      extracted.values.begin(), extracted.values.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(
      restored.rightHandSides.begin(), restored.rightHandSides.end(),
      extracted.rightHandSides.begin(), extracted.rightHandSides.end());

  std::ifstream input(file.path().c_str(),
                      std::ios::binary | std::ios::ate);
  BOOST_REQUIRE(input);
  const std::streamoff expectedBytes =
    40 + 4 * 4 + 4 * 6 + 8 * 6 + 8 * 3;
  BOOST_CHECK_EQUAL(
      static_cast<std::streamoff>(input.tellg()), expectedBytes);

  const unsigned char expectedHeader[40] = {
    'G', 'P', 'C', 'S', 'R', '0', '0', '1',
    1, 0, 0, 0,
    40, 0, 0, 0,
    7, 0, 0, 0,
    3, 0, 0, 0,
    3, 0, 0, 0,
    6, 0, 0, 0,
    1, 0, 0, 0,
    0, 0, 0, 0
  };
  unsigned char header[40] = {};
  input.seekg(0);
  input.read(reinterpret_cast<char *>(header), sizeof(header));
  BOOST_REQUIRE_EQUAL(input.gcount(),
                      static_cast<std::streamsize>(sizeof(header)));
  BOOST_CHECK_EQUAL_COLLECTIONS(
      header, header + sizeof(header),
      expectedHeader, expectedHeader + sizeof(expectedHeader));
}

BOOST_AUTO_TEST_CASE(rejectNonFiniteMatrixValueDuringExtraction)
{
  gridpack::parallel::Communicator world;
  BOOST_REQUIRE_EQUAL(world.size(), 1);

  gridpack::math::RealMatrix matrix(
      world, 1, 1, gridpack::math::Sparse);
  matrix.setElement(
      0, 0, std::numeric_limits<double>::quiet_NaN());
  matrix.ready();

  gridpack::math::RealVector rightHandSide(world, 1);
  rightHandSide.setElement(0, 1.0);
  rightHandSide.ready();

  BOOST_CHECK_THROW(
      gridpack::math::extractPetscRealCsrSystem(
          matrix, rightHandSide),
      gridpack::Exception);
}

BOOST_AUTO_TEST_CASE(rejectNonFiniteRightHandSideDuringExtraction)
{
  gridpack::parallel::Communicator world;
  BOOST_REQUIRE_EQUAL(world.size(), 1);

  gridpack::math::RealMatrix matrix(
      world, 1, 1, gridpack::math::Sparse);
  matrix.setElement(0, 0, 1.0);
  matrix.ready();

  gridpack::math::RealVector rightHandSide(world, 1);
  rightHandSide.setElement(
      0, std::numeric_limits<double>::infinity());
  rightHandSide.ready();

  BOOST_CHECK_THROW(
      gridpack::math::extractPetscRealCsrSystem(
          matrix, rightHandSide),
      gridpack::Exception);
}

BOOST_AUTO_TEST_CASE(rejectTrailingData)
{
  gridpack::math::RealCsrSystem system;
  system.rows = 1;
  system.columns = 1;
  system.nonzeros = 1;
  system.rightHandSideCount = 1;
  system.rowOffsets.push_back(0);
  system.rowOffsets.push_back(1);
  system.columnIndices.push_back(0);
  system.values.push_back(2.0);
  system.rightHandSides.push_back(4.0);

  TemporaryFile file("petsc_csr_exporter_trailing_test.gpcsr");
  gridpack::math::writeRealCsrSystem(file.path(), system);
  std::ofstream output(file.path().c_str(),
                       std::ios::binary | std::ios::app);
  output.put('\0');
  output.close();

  BOOST_CHECK_THROW(
      gridpack::math::readRealCsrSystem(file.path()),
      gridpack::Exception);
}

BOOST_AUTO_TEST_CASE(rejectHugeHeaderWithShortPayload)
{
  TemporaryFile file("petsc_csr_exporter_huge_header_test.gpcsr");
  const unsigned char header[40] = {
    'G', 'P', 'C', 'S', 'R', '0', '0', '1',
    1, 0, 0, 0,
    40, 0, 0, 0,
    7, 0, 0, 0,
    255, 255, 255, 255,
    255, 255, 255, 255,
    255, 255, 255, 255,
    255, 255, 255, 255,
    0, 0, 0, 0
  };
  std::ofstream output(file.path().c_str(),
                       std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char *>(header), sizeof(header));
  output.close();

  BOOST_CHECK_THROW(
      gridpack::math::readRealCsrSystem(file.path()),
      gridpack::Exception);
}

BOOST_AUTO_TEST_CASE(writeSequentialRealVector)
{
  gridpack::parallel::Communicator world;
  BOOST_REQUIRE_EQUAL(world.size(), 1);

  gridpack::math::RealVector destination(world, 3);
  destination.fill(0.0);
  destination.ready();
  const std::vector<double> values = {1.25, -2.5, 3.75};
  gridpack::math::writePetscRealVector(destination, values);

  for (int index = 0; index < 3; ++index) {
    double value = 0.0;
    destination.getElement(index, value);
    BOOST_CHECK_EQUAL(value, values[static_cast<std::size_t>(index)]);
  }
  const std::vector<double> wrongSize = {1.0, 2.0};
  BOOST_CHECK_THROW(
      gridpack::math::writePetscRealVector(destination, wrongSize),
      gridpack::Exception);
}

BOOST_AUTO_TEST_SUITE_END()
