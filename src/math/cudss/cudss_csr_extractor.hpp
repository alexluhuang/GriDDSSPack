// Emacs Mode Line: -*- Mode:c++;-*-
// -------------------------------------------------------------
/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
// -------------------------------------------------------------
/**
 * @file   cudss_csr_extractor.hpp
 * @brief  Expose a GridPACK matrix/vector's underlying PETSc storage as the
 *         plain CSR / contiguous arrays cuDSS consumes.
 *
 * cuDSS wants a 3-array CSR system matrix and column-major dense RHS/solution.
 * GridPACK matrices are PETSc @c Mat objects (MATAIJ) with a fixed sparsity
 * pattern; on a serial (SEQAIJ) matrix PETSc already stores exactly the CSR we
 * need, so extraction is a pointer hand-off (no reformatting).  On the
 * DGX Spark / GB10 these arrays live in unified coherent memory, so cuDSS can
 * read them without a host->device copy; on ordinary hardware the caller stages
 * them into device buffers.
 *
 * The cuDSS backend forces the base LinearSolverImplementation to solve
 * serially when running on more than one process (see the derived p_configure),
 * so the @c Mat handed to p_solveImpl() is always SEQAIJ here.
 */
// -------------------------------------------------------------

#ifndef _cudss_csr_extractor_hpp_
#define _cudss_csr_extractor_hpp_

#ifdef GRIDPACK_WITH_CUDSS

#include <petscmat.h>
#include <petscvec.h>
#include <cudss.h>
#include <library_types.h>

#include "gridpack/math/matrix.hpp"
#include "petsc/petsc_matrix_extractor.hpp"
#include "petsc/petsc_vector_extractor.hpp"
#include "cudss/cudss_exception.hpp"

namespace gridpack {
namespace math {

// -------------------------------------------------------------
//  cuDSS type mapping (keyed off the PETSc scalar/index configuration so the
//  types always match the extracted CSR arrays, regardless of the T of the
//  GridPACK MatrixT<T,I>).
// -------------------------------------------------------------

/// The cuDSS value type of the underlying PETSc scalar.
/**
 * NOTE: cuDSS (>= 0.x) uses its own @c cudssDataType_t enum for the
 * matrix-creation type arguments, distinct from CUDA's @c cudaDataType_t; the
 * two are not implicitly convertible in C++.  The CUDSS_* enumerators alias the
 * corresponding CUDA_* values, so this is a type-name change, not a value one.
 */
inline cudssDataType_t cudssPetscValueType(void)
{
#if defined(PETSC_USE_COMPLEX)
  return CUDSS_C_64F;
#else
  return CUDSS_R_64F;
#endif
}

/// The cuDSS integer type of the underlying PETSc index (offsets and indices).
inline cudssDataType_t cudssPetscIndexType(void)
{
  return (sizeof(PetscInt) == 8) ? CUDSS_R_64I : CUDSS_R_32I;
}

// -------------------------------------------------------------
//  class PetscSeqCSRView
// -------------------------------------------------------------
/// RAII read-only view of a SEQAIJ matrix as 0-based CSR host arrays.
/**
 * Acquires the PETSc CSR structure (row offsets, column indices) and value
 * array on construction and releases them on destruction, so the pointers are
 * valid only for the lifetime of the view.  The pattern arrays (rowptr/colind)
 * are stable across Newton iterations; only values() changes.
 */
template <typename T, typename I>
class PetscSeqCSRView
{
public:

  explicit PetscSeqCSRView(MatrixT<T, I>& A)
    : p_mat(PETScMatrix(A)),
      p_n(0), p_nnz(0),
      p_ia(NULL), p_ja(NULL), p_a(NULL),
      p_gotIJ(false), p_gotArr(false)
  {
    PetscBool done = PETSC_FALSE;
    // shift=0 (0-based), symmetric=FALSE, inodecompressed=FALSE -> plain CSR
    GP_PETSC_CHECK(MatGetRowIJ(*p_mat, 0, PETSC_FALSE, PETSC_FALSE,
                               &p_n, &p_ia, &p_ja, &done));
    if (!done) {
      throw gridpack::Exception(
        "cuDSS backend: MatGetRowIJ did not return CSR structure "
        "(matrix is not SEQAIJ -- is the solve serial?)");
    }
    p_gotIJ = true;
    GP_PETSC_CHECK(MatSeqAIJGetArrayRead(*p_mat, &p_a));
    p_gotArr = true;
    p_nnz = (p_n > 0) ? p_ia[p_n] : 0;
  }

  ~PetscSeqCSRView(void)
  {
    // Restore in reverse acquisition order; swallow errors in the destructor.
    try {
      if (p_gotArr) {
        MatSeqAIJRestoreArrayRead(*p_mat, &p_a);
      }
      if (p_gotIJ) {
        PetscBool done = PETSC_FALSE;
        MatRestoreRowIJ(*p_mat, 0, PETSC_FALSE, PETSC_FALSE,
                        &p_n, &p_ia, &p_ja, &done);
      }
    } catch (...) {
      // never throw from a destructor
    }
  }

  PetscInt rows(void)   const { return p_n; }
  PetscInt nnz(void)    const { return p_nnz; }
  const PetscInt   *rowptr(void) const { return p_ia; }
  const PetscInt   *colind(void) const { return p_ja; }
  const PetscScalar *values(void) const { return p_a; }

private:

  Mat *p_mat;
  PetscInt p_n;
  PetscInt p_nnz;
  const PetscInt   *p_ia;
  const PetscInt   *p_ja;
  const PetscScalar *p_a;
  bool p_gotIJ;
  bool p_gotArr;

  // non-copyable
  PetscSeqCSRView(const PetscSeqCSRView&);
  PetscSeqCSRView& operator=(const PetscSeqCSRView&);
};

// -------------------------------------------------------------
//  Vector array access (SEQ vectors -> contiguous host arrays)
// -------------------------------------------------------------

/// Read-only access to a GridPACK vector's contiguous PETSc value array.
template <typename T, typename I>
inline const PetscScalar *petscVecArrayRead(const VectorT<T, I>& v, PetscInt& n)
{
  const Vec *vv = PETScVector(v);
  const PetscScalar *arr = NULL;
  GP_PETSC_CHECK(VecGetLocalSize(*vv, &n));
  GP_PETSC_CHECK(VecGetArrayRead(*vv, &arr));
  return arr;
}

template <typename T, typename I>
inline void petscVecRestoreArrayRead(const VectorT<T, I>& v,
                                     const PetscScalar *&arr)
{
  const Vec *vv = PETScVector(v);
  VecRestoreArrayRead(*vv, &arr);
}

/// Writable access to a GridPACK vector's contiguous PETSc value array.
template <typename T, typename I>
inline PetscScalar *petscVecArray(VectorT<T, I>& v, PetscInt& n)
{
  Vec *vv = PETScVector(v);
  PetscScalar *arr = NULL;
  GP_PETSC_CHECK(VecGetLocalSize(*vv, &n));
  GP_PETSC_CHECK(VecGetArray(*vv, &arr));
  return arr;
}

template <typename T, typename I>
inline void petscVecRestoreArray(VectorT<T, I>& v, PetscScalar *&arr)
{
  Vec *vv = PETScVector(v);
  VecRestoreArray(*vv, &arr);
}

} // namespace math
} // namespace gridpack

#endif // GRIDPACK_WITH_CUDSS

#endif
