# =============================================================================
# GridPACK contingency-analysis image with the NVIDIA cuDSS GPU direct-solver
# backend compiled in (opt-in at runtime via input.xml).
#
# Target platform: NVIDIA DGX Spark / GB10 Grace Blackwell (aarch64 / sbsa,
# compute capability sm_121, 128 GB unified coherent memory).  A single ca.x is
# produced that contains BOTH the existing PETSc/KLU CPU path and the cuDSS GPU
# path; the GPU path is selected at run time from input.xml and silently falls
# back to PETSc when no CUDA device is visible.  The CLI is unchanged:
#     mpirun -n K ca.x input.xml
#
# NOTE: the CUDA/cuDSS versions below are build-time knobs.  Pin them to a
# known-good pair for your driver (cuDSS requires CUDA 12.x+ and Pascal-or-newer;
# GB10 is Blackwell/sm_121).  See GPU_CA_IMPLEMENTATION.md.
# =============================================================================

# CUDA-enabled aarch64 (sbsa) base image.  The *devel* image ships nvcc, the
# CUDA runtime/headers and the NVIDIA apt repo used to install cuDSS below.
# NOTE: GB10 (Blackwell, sm_121) requires CUDA >= 12.8; CUDA 13.x is used here to
# match the cuDSS cuda13 packages.  Pin to a tag valid for your driver.
ARG cuda_image=nvidia/cuda:13.0.0-devel-ubuntu24.04
FROM ${cuda_image}

# Configure dependency versions
ARG boost_version=1.81.0
ARG ga_version=5.9.1
ARG petsc_version=3.24.2
# NVIDIA cuDSS redistributable archive (sbsa / aarch64, cuda13).  Pinned to the
# version validated on the GB10.  Downloaded directly from the NVIDIA redist
# server so the image is self-contained and reproducible.
ARG cudss_archive=libcudss-linux-sbsa-0.8.0.10_cuda13-archive
ARG cudss_url=https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-sbsa/libcudss-linux-sbsa-0.8.0.10_cuda13-archive.tar.xz
# GPU compute architecture(s).  121 == sm_121 (GB10 Blackwell).  Real code is
# emitted for sm_121 and PTX is kept as a forward-compatible fallback.
ARG cuda_arch=121

# Setup environment variables used throughout installation
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC GNUMAKEFLAGS=--no-print-directory
ENV OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

ENV GRIDPACK_ROOT_DIR=/app GP_EXT_DEPS=/deps
ENV GRIDPACK_INSTALL_DIR=${GRIDPACK_ROOT_DIR}/src/install GRIDPACK_BUILD_DIR=${GRIDPACK_ROOT_DIR}/src/build
ENV GRIDPACK_DIR=${GRIDPACK_INSTALL_DIR}

ENV boost_dir=${GP_EXT_DEPS}/boost-${boost_version} \
    ga_dir=${GP_EXT_DEPS}/ga-${ga_version} \
    petsc_dir=${GP_EXT_DEPS}/petsc

ENV boost_gp_dir=${boost_dir}/install_for_gridpack \
    ga_gp_dir=${ga_dir}/install_for_gridpack \
    petsc_gp_dir=${petsc_dir}/install_for_gridpack

ENV PETSC_DIR=${petsc_dir} PETSC_ARCH=build-dir

# cuDSS is unpacked from the redist archive to /opt/cudss (see below).
ENV CUDSS_DIR=/opt/cudss
ENV cudss_DIR=/opt/cudss/lib/cmake/cudss

ENV LD_LIBRARY_PATH=${boost_gp_dir}/lib:${ga_gp_dir}/lib:${petsc_gp_dir}/lib:/opt/cudss/lib:/usr/local/cuda/lib64
ENV DYLD_LIBRARY_PATH=${LD_LIBRARY_PATH}

# Install required system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends cmake make wget tzdata git gfortran build-essential pkg-config \
    python3 python3-pip python3-venv python3-dev python-is-python3 \
    openmpi-bin openmpi-common openmpi-doc libopenmpi-dev && \
    apt-get clean

# -----------------------------------------------------------------------------
# Install NVIDIA cuDSS (Direct Sparse Solver) from the redistributable archive.
#
# Downloading the pinned redist tarball (rather than relying on apt package
# names, which vary by CUDA release) keeps the image self-contained and
# reproducible.  Unpacked to /opt/cudss so cudss_DIR=/opt/cudss/lib/cmake/cudss.
# -----------------------------------------------------------------------------
RUN mkdir -p /opt/cudss && cd /tmp && \
    wget -q "${cudss_url}" -O cudss.tar.xz && \
    tar -xf cudss.tar.xz && \
    cp -a "${cudss_archive}/." /opt/cudss/ && \
    rm -rf cudss.tar.xz "${cudss_archive}" && \
    test -f /opt/cudss/lib/cmake/cudss/cudss-config.cmake && \
    echo "cuDSS installed to /opt/cudss"

# Compile/Install Boost
WORKDIR ${GP_EXT_DEPS}
RUN wget "https://github.com/boostorg/boost/releases/download/boost-${boost_version}/boost-${boost_version}.tar.gz"
RUN tar -xf "boost-${boost_version}.tar.gz"
WORKDIR ${boost_dir}
RUN ./bootstrap.sh --prefix=install_for_gridpack --with-libraries=mpi,serialization,random,filesystem,system
RUN echo 'using mpi : mpicxx ; ' >> project-config.jam
RUN ./b2 -a -d+2 link="shared" stage
RUN ./b2 -a -d+2 link="shared" install

# Compile/Install GA (Global Arrays)
WORKDIR ${GP_EXT_DEPS}
RUN wget "https://github.com/GlobalArrays/ga/releases/download/v${ga_version}/ga-${ga_version}.tar.gz"
RUN tar -xf "ga-${ga_version}.tar.gz"
WORKDIR ${ga_dir}
RUN ./configure --with-mpi-ts --disable-f77 \
    --without-blas --without-lapack --without-scalapack \
    --enable-cxx --enable-i4 \
    --prefix=${ga_gp_dir} \
    CFLAGS="-Wno-implicit-function-declaration -Wno-incompatible-pointer-types -Wno-old-style-definition" \
    CXXFLAGS="-Wno-incompatible-pointer-types" \
    --enable-shared=yes --enable-static=no
RUN make -j 10 install
WORKDIR ${GP_EXT_DEPS}

# Compile/Install PETSc.
#
# PETSc stays CPU-only: it provides the fallback KLU/SuperLU_DIST/MUMPS direct
# solvers.  cuDSS is added as its own GridPACK backend, not through PETSc
# (PETSc has no native cuDSS MatSolverType), so --with-cuda is intentionally
# NOT set here.
RUN git clone https://gitlab.com/petsc/petsc.git
WORKDIR ${petsc_dir}
RUN git checkout "tags/v${petsc_version}" -b "v${petsc_version}"
RUN ./configure \
    --prefix=${PWD}/install_for_gridpack \
    --scalar-type=real  \
    --with-fortran-bindings=0 \
    --download-superlu_dist \
    --download-metis \
    --download-parmetis \
    --download-suitesparse \
    --download-f2cblaslapack \
    --download-scalapack \
    --download-mumps \
    --download-cmake=0 \
    --with-sowing=0 \
    --with-debugging=0 \
    --with-shared-libraries=1
RUN make all
RUN make install
#RUN make PETSC_DIR=${petsc_gp_dir} PETSC_ARCH="" check

# Copy in GridPACK source code from repository
COPY README.md .gitignore .gitmodules ${GRIDPACK_ROOT_DIR}/
COPY .git ${GRIDPACK_ROOT_DIR}/.git
COPY docs ${GRIDPACK_ROOT_DIR}/docs
COPY python ${GRIDPACK_ROOT_DIR}/python
COPY src ${GRIDPACK_ROOT_DIR}/src

# Build GridPACK.
#
# Key differences from the CPU-only image:
#   * CMAKE_BUILD_TYPE=Release   (was Debug -- a large, free speedup; measure
#                                 this as the new baseline BEFORE crediting GPU)
#   * GRIDPACK_WITH_CUDSS=ON     compiles the cuDSS backend into the binaries
#   * cudss_DIR                  points CMake at the cuDSS package config
#   * CMAKE_CUDA_ARCHITECTURES   sm_121 (GB10) with PTX fallback
WORKDIR ${GRIDPACK_BUILD_DIR}
RUN cmake -Wdev -D GA_DIR:STRING=${ga_gp_dir} \
    -D Boost_ROOT:STRING=${boost_gp_dir} \
    -D Boost_DIR:string=${boost_gp_dir}/lib/cmake/Boost-${boost_version} \
    -D PETSC_DIR:PATH=${petsc_gp_dir} \
    -D MPI_CXX_COMPILER:STRING='mpicxx' \
    -D MPI_C_COMPILER:STRING='mpicc' \
    -D MPIEXEC:STRING='mpiexec' \
    -D MPIEXEC_MAX_NUMPROCS:STRING=2 \
    -D GRIDPACK_TEST_TIMEOUT:STRING=120 \
    -D ENABLE_ENVIRONMENT_FROM_COMM:BOOL=YES \
    -D CMAKE_INSTALL_PREFIX:PATH=${GRIDPACK_INSTALL_DIR} \
    -D CMAKE_BUILD_TYPE:STRING=Release \
    -D GRIDPACK_WITH_CUDSS:BOOL=ON \
    -D cudss_DIR:PATH="${cudss_DIR}" \
    -D CMAKE_CUDA_ARCHITECTURES:STRING="${cuda_arch}" \
    -D BUILD_SHARED_LIBS=true \
    ..
RUN make install

# Install Python bindings
WORKDIR ${GRIDPACK_ROOT_DIR}
RUN git submodule update --init
RUN pip config --global set global.break-system-packages true
WORKDIR ${GRIDPACK_ROOT_DIR}/python
RUN pip install --upgrade --prefix=${GRIDPACK_INSTALL_DIR} .

# Configure Python module search path using .pth file (no environment variables needed)
# Python automatically reads .pth files from its site-packages directories
RUN pyvnum=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")') && \
    system_site_packages=$(python3 -c 'import site; print(site.getsitepackages()[0])') && \
    echo "${GRIDPACK_INSTALL_DIR}/lib/python${pyvnum}/site-packages" > ${system_site_packages}/gridpack.pth && \
    echo "${GRIDPACK_INSTALL_DIR}/local/lib/python${pyvnum}/dist-packages" >> ${system_site_packages}/gridpack.pth && \
    echo "Configured Python ${pyvnum} module search path via ${system_site_packages}/gridpack.pth"

WORKDIR ${GRIDPACK_ROOT_DIR}/workspace
ENV PATH=${GRIDPACK_INSTALL_DIR}/bin:${GRIDPACK_INSTALL_DIR}/local/bin:${PATH}
