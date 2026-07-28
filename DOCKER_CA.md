# Docker build and run: optimized contingency analysis

This image installs one `ca.x` containing both the PETSc/KLU CPU path and the
opt-in NVIDIA cuDSS path. The user-facing command remains the normal GridPACK
command:

```bash
mpirun -n 20 ca.x input.xml
```

## Requirements

- NVIDIA DGX Spark / GB10 (ARM64)
- NVIDIA driver compatible with CUDA 13
- Docker Engine and NVIDIA Container Toolkit

Verify GPU container access:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

## Build

From the repository root:

```bash
cd /home/alh360/Documents/GriDSSPack
docker build --platform linux/arm64 \
  -t griddsspack:ca-gpu-20260727 \
  .
```

The first build compiles Boost, Global Arrays, PETSc, GridPACK, and the Python
bindings. Later source-only rebuilds reuse the dependency layers.

Verify the installed executable:

```bash
docker run --rm --gpus all \
  griddsspack:ca-gpu-20260727 \
  bash -lc 'command -v ca.x && ldd "$(command -v ca.x)" | grep -E "cudss|cuda"'
```

## Run Texas7k with all-rank GPU waves

The case directory must contain `input.xml` and the RAW file referenced by its
`networkConfiguration` element. For the supplied case:

```bash
cd /home/alh360/GridLensProjects/Texas7k_v2/original_inputs

docker run --rm --gpus all \
  --shm-size=32g \
  --ulimit memlock=-1:-1 \
  -v "$PWD":/app/workspace \
  -w /app/workspace \
  griddsspack:ca-gpu-20260727 \
  mpirun -n 20 ca.x input.xml
```

This differs from native/base GridPACK only by the outer `docker run`; inside
the container the invocation is still `mpirun -n 20 ca.x input.xml`.

The equivalent interactive workflow, matching the stock
`pnnl/gridpack:ca-scalability-v2` usage, is:

```bash
docker run -it --rm --gpus all \
  --shm-size=32g \
  --ulimit memlock=-1:-1 \
  -v "$PWD":/app/workspace \
  griddsspack:ca-gpu-20260727 bash

# Inside the container:
mpirun -n 20 ca.x input.xml
```

The GPU is explicitly enabled in `Contingency_analysis/GPU`. Every MPI rank
claims independent waves of eight from the shared task stream. Generator,
islanding, non-converged, and controller-changing cases fall back locally to
PETSc/KLU.

## Texas7k XML profile

The supplied
`/home/alh360/GridLensProjects/Texas7k_v2/original_inputs/input.xml` has been
updated in place. Its GPU and output settings are:

```xml
<outputFormat>csv_flat</outputFormat>
<sharedFlatFile>true</sharedFlatFile>
<bufferFlatOutput>false</bufferFlatOutput>
<outputFile>Texas7k_v2</outputFile>
<GPU>
  <enabled>true</enabled>
  <batched>true</batched>
  <waveSize>8</waveSize>
  <warmStart>true</warmStart>
  <screen>true</screen>
</GPU>
```

The power-flow solver block selects cuDSS while retaining the stock PETSc/KLU
options for local fallbacks:

```xml
<LinearSolver>
  <Backend>cudss</Backend>
  <constantFactor>true</constantFactor>
  <PETScOptions>
    -ksp_type preonly
    -pc_type lu
    -pc_factor_mat_solver_type klu
  </PETScOptions>
</LinearSolver>
```

## Run the same image CPU-only

Set these values in a copy of `input.xml`:

```xml
<GPU>
  <enabled>false</enabled>
  <batched>false</batched>
  <waveSize>8</waveSize>
  <warmStart>true</warmStart>
  <screen>true</screen>
</GPU>
```

and:

```xml
<LinearSolver>
  <Backend>petsc</Backend>
  <PETScOptions>
    -ksp_type preonly
    -pc_type lu
    -pc_factor_mat_solver_type klu
  </PETScOptions>
</LinearSolver>
```

Run it with the same Docker command. `--gpus all` may be omitted for the
CPU-only input.

## Expected runtime confirmation

The startup log for the GPU input should contain:

```text
GPU batched contingency engine ENABLED on all 20 rank(s), waveSize=8
```

At completion it should contain:

```text
[GPU all-rank summary] ranks=20 waveSize=8 ... inspected=<total contingencies>
```

`inspected` must equal the reported total contingency count. The default
`csv_flat` artifact retains the documented 15-column schema.

## Verified full Texas run

The image was built and exercised on an NVIDIA GB10 with the supplied full
branch-and-generator N-1 Texas case using the 20-rank command above:

```text
Image:          griddsspack:ca-gpu-20260727
Image platform: linux/arm64
MPI ranks:      20
Wave size:      8 per rank
Contingencies:  8,891 inspected
Rank waves:     1,112
Wall time:      87 seconds
GridPACK time:  84.745 seconds maximum
```

All 20 ranks received work (416 to 488 contingencies per rank). The run
reported 7,099 GPU-eligible cases, retained 6,414 on the GPU path after
non-convergence and controller fallbacks, and exited successfully.

It produced the required aggregate artifacts:

```text
Texas7k_v2_flat.csv         9,204,331,359 bytes
Texas7k_v2_convergence.csv        654,989 bytes
Texas7k_v2_buses.csv              300,658 bytes
```

Their verified headers are:

```text
event_idx,contingency,from_bus,to_bus,circuit_id,p_from_mw,q_from_mvar,mva_from,rate_mva,loading_percent,viol,v_from_pu,v_to_pu,ang_from_deg,ang_to_deg
event_idx,contingency,type,converged,iterations,final_tolerance,max_p_bus,max_p_mismatch,max_q_bus,max_q_mismatch,status_code
bus_id,bus_name,base_kv,area,zone,owner,area_name,zone_name,owner_name
```
