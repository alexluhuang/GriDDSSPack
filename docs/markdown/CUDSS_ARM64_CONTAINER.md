# Native Arm64 CUDA/cuDSS contingency-analysis image

`Dockerfile.cudss-arm64` builds a Release `ca.x` with the optional NVIDIA
cuDSS linear-solver backend. The image is intentionally Arm64-only; build it
natively on the DGX Spark or another `aarch64` host:

```bash
./tools/build_cudss_arm64_image.sh gridpack-cudss-arm64:cuda13-cudss0.8
```

The wrapper rejects x86 hosts so the build cannot silently use QEMU. The
Dockerfile also checks `uname`, the Debian architecture, and BuildKit's target
architecture.

## Run

Place `input.xml` and every file it references in the current directory. The
entrypoint preserves the normal GridPACK interface:

```bash
docker run --rm --gpus all \
  -v "$PWD":/work \
  gridpack-cudss-arm64:cuda13-cudss0.8 input.xml
```

Select cuDSS under the power-flow `LinearSolver` configuration:

```xml
<LinearSolver>
  <Backend>cudss</Backend>
  <CUDSSMode>device</CUDSSMode>
  <CUDSSDevice>0</CUDSSDevice>
  <CUDSSStrict>false</CUDSSStrict>
  <CUDSSResidualTolerance>1e-10</CUDSSResidualTolerance>
  <CUDSSDiagnostics>false</CUDSSDiagnostics>
  <PETScPrefix>pre_</PETScPrefix>
  <PETScOptions>
    -ksp_type preonly
    -pc_type lu
    -pc_factor_mat_solver_type klu
  </PETScOptions>
</LinearSolver>
```

`CUDSSMode` accepts `device` and `hybrid`. With `CUDSSStrict` set to `false`,
an unavailable GPU or an unsupported matrix uses the configured PETSc/KLU
path. Set it to `true` when a cuDSS failure must stop the run.

The same image can therefore run without NVIDIA device injection:

```bash
docker run --rm \
  -v "$PWD":/work \
  gridpack-cudss-arm64:cuda13-cudss0.8 input.xml
```

Alternatively, omit `Backend` (the default is `petsc`) or set it explicitly
to `petsc` for a CPU-only run.

For an interactive command, override the CA entrypoint with a shell:

```bash
docker run --rm -it --entrypoint /bin/bash \
  -v "$PWD":/work \
  gridpack-cudss-arm64:cuda13-cudss0.8
```

## Inspect and smoke-test

Confirm that the image is native Arm64 and that all startup libraries resolve:

```bash
docker image inspect \
  --format '{{.Architecture}} {{index .Config.Labels "org.opencontainers.image.cuda.version"}} {{index .Config.Labels "org.opencontainers.image.cudss.version"}}' \
  gridpack-cudss-arm64:cuda13-cudss0.8

docker run --rm --entrypoint /bin/sh \
  gridpack-cudss-arm64:cuda13-cudss0.8 \
  -c 'test "$(uname -m)" = aarch64 && ! ldd /opt/gridpack/bin/ca.x | grep "not found"'
```

A bounded 14-bus smoke fixture is included in the image:

```bash
docker run --rm \
  --workdir /opt/gridpack/share/gridpack/smoke/contingency_analysis \
  gridpack-cudss-arm64:cuda13-cudss0.8 \
  input_14_auto_n1.xml
```

Exercise cuDSS in strict mode with the same small fixture:

```bash
docker run --rm --gpus all \
  --env GRIDPACK_LINEAR_SOLVER_BACKEND=cudss \
  --env GRIDPACK_CUDSS_STRICT=true \
  --workdir /opt/gridpack/share/gridpack/smoke/contingency_analysis \
  gridpack-cudss-arm64:cuda13-cudss0.8 \
  input_14_auto_n1.xml
```

This is a small smoke test, not a production contingency sweep.

## Reproducibility and compatibility

The Dockerfile pins architecture-specific image manifests and verifies the
cuDSS archive checksum:

- GridPACK dependency image revision `bb411473`, with Boost 1.81.0,
  Global Arrays 5.9.1, PETSc 3.24.2, and PETSc's SuiteSparse/KLU build
- NVIDIA CUDA 13.0.0 devel and runtime Arm64 manifests
- NVIDIA cuDSS 0.8.0.10 CUDA 13 SBSA archive, SHA-256
  `c5fe7e5796792e10c3c5971bbb169ab3040ba61fe6fc99bdcbc02cf0f1ed9409`

NVIDIA's CUDA stages use Ubuntu 24.04 while the pinned GridPACK dependency
image uses Ubuntu 25.10. The CUDA toolkit is self-contained and is copied into
the newer-glibc image; this cross-Ubuntu composition is an explicit
compatibility risk and must remain covered by the load check and GPU smoke
test above. The NVIDIA driver is supplied by the host at runtime and is not
baked into the image.
