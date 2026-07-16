cmake_minimum_required(VERSION 3.22)

project(gridpack_csr_solver_benchmark LANGUAGES CXX)

option(GRIDPACK_CSR_BENCHMARK_ENABLE_KLU
  "Build the SuiteSparse KLU reference backend" ON)
option(GRIDPACK_CSR_BENCHMARK_ENABLE_CUDSS
  "Build the NVIDIA cuDSS backend" ON)

if(NOT GRIDPACK_CSR_BENCHMARK_ENABLE_KLU
   AND NOT GRIDPACK_CSR_BENCHMARK_ENABLE_CUDSS)
  message(FATAL_ERROR "At least one solver backend must be enabled")
endif()

add_executable(gridpack_csr_solver_benchmark
  benchmark_main.cpp
  csr_system.cpp
)

target_compile_features(gridpack_csr_solver_benchmark PRIVATE cxx_std_17)
target_compile_options(gridpack_csr_solver_benchmark PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Wconversion
  -Wshadow
)

if(GRIDPACK_CSR_BENCHMARK_ENABLE_KLU)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(KLU REQUIRED IMPORTED_TARGET KLU)
  target_sources(gridpack_csr_solver_benchmark PRIVATE klu_backend.cpp)
  target_compile_definitions(gridpack_csr_solver_benchmark
    PRIVATE GRIDPACK_CSR_BENCHMARK_HAVE_KLU=1)
  target_link_libraries(gridpack_csr_solver_benchmark PRIVATE PkgConfig::KLU)
endif()

if(GRIDPACK_CSR_BENCHMARK_ENABLE_CUDSS)
  find_package(CUDAToolkit 13 REQUIRED)
  find_package(cudss 0.8.0 CONFIG REQUIRED)
  target_sources(gridpack_csr_solver_benchmark PRIVATE cudss_backend.cpp)
  target_compile_definitions(gridpack_csr_solver_benchmark
    PRIVATE GRIDPACK_CSR_BENCHMARK_HAVE_CUDSS=1)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(gridpack_csr_solver_benchmark
      PRIVATE
        cudss
        -Wl,--no-as-needed
        CUDA::cublas
        -Wl,--as-needed
        CUDA::cusolver
        CUDA::cudart
    )
  else()
    target_link_libraries(gridpack_csr_solver_benchmark
      PRIVATE
        cudss
        CUDA::cublas
        CUDA::cusolver
        CUDA::cudart
    )
  endif()
endif()
