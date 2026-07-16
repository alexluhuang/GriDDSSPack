/*
 * Copyright (c) 2026 Battelle Memorial Institute
 * Licensed under the modified BSD License.
 */

#ifndef GRIDPACK_MATH_PETSC_CSR_EXPORTER_HPP_
#define GRIDPACK_MATH_PETSC_CSR_EXPORTER_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include "gridpack/math/matrix.hpp"
#include "gridpack/math/vector.hpp"

namespace gridpack {
namespace math {

/**
 * A real, zero-based CSR linear system.
 */
struct RealCsrSystem
{
  std::uint32_t rows;
  std::uint32_t columns;
  std::uint32_t nonzeros;
  std::uint32_t rightHandSideCount;
  std::vector<std::uint32_t> rowOffsets;
  std::vector<std::uint32_t> columnIndices;
  std::vector<double> values;
  std::vector<double> rightHandSides;
};

/**
 * Power-flow context stored in the diagnostic export manifest.
 */
struct PetscCsrExportContext
{
  std::string caseName;
  bool baseCase;
  int areaInterchangePass;
  int controllerPass;
  int newtonIteration;
  int linearSolveOrdinal;
  bool qlimEnabled;
  bool switchedShuntEnabled;
  bool ltcEnabled;
  bool areaInterchangeEnabled;
};

/**
 * Write a system in GridPACK's compact CSR interchange format.
 *
 * Version 1 is byte-serialized little-endian data, never a native C++ struct:
 *
 *   byte 0:  char[8] "GPCSR001"
 *   byte 8:  uint32 version (1)
 *   byte 12: uint32 header size (40)
 *   byte 16: uint32 flags (bit 0 little endian, bit 1 zero-based CSR,
 *            bit 2 IEEE-754 binary64)
 *   byte 20: uint32 rows
 *   byte 24: uint32 columns
 *   byte 28: uint32 nonzeros
 *   byte 32: uint32 right-hand-side count
 *   byte 36: uint32 reserved (0)
 *
 * The payload immediately follows the header:
 *
 *   uint32 rowOffsets[rows + 1]
 *   uint32 columnIndices[nonzeros]
 *   binary64 values[nonzeros]
 *   binary64 rightHandSides[rows * rightHandSideCount]
 *
 * Multiple right-hand sides are consecutive vectors (column-major).
 */
void writeRealCsrSystem(const std::string& path,
                        const RealCsrSystem& system);

/**
 * Read and strictly validate a version 1 GridPACK CSR system.
 */
RealCsrSystem readRealCsrSystem(const std::string& path);

/**
 * Copy a real, sequential PETSc AIJ matrix and sequential vector into CSR.
 *
 * This diagnostic deliberately rejects complex or non-double PETSc builds,
 * 64-bit PetscInt builds, non-SeqAIJ matrices, and distributed vectors.
 */
RealCsrSystem extractPetscRealCsrSystem(const RealMatrix& matrix,
                                        const RealVector& rightHandSide);

/**
 * Export a PETSc system when diagnostic capture is enabled.
 *
 * Normal execution is unchanged unless GRIDPACK_PF_CSR_EXPORT_DIR is set.
 * An enabled capture must also set at least one guard:
 *
 *   GRIDPACK_PF_CSR_EXPORT_FILTER  case-name substring to capture
 *   GRIDPACK_PF_CSR_EXPORT_LIMIT   positive per-process file limit
 *
 * Each process writes uniquely named .gpcsr files and its own manifest CSV.
 */
void exportPetscRealCsrSystemIfEnabled(
    const RealMatrix& matrix,
    const RealVector& rightHandSide,
    const PetscCsrExportContext& context);

} // namespace math
} // namespace gridpack

#endif
