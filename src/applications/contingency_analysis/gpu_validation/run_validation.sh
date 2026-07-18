#!/usr/bin/env bash
# -----------------------------------------------------------------------------
#     Copyright (c) 2013 Battelle Memorial Institute
#     Licensed under modified BSD License. A copy of this license can be found
#     in the LICENSE file in the top level directory of this distribution.
# -----------------------------------------------------------------------------
#
# End-to-end parity harness for the GPU (cuDSS) contingency-analysis path.
#
# From ONE base input.xml it derives a CPU run (Backend=petsc, GPU disabled) and
# a GPU run (Backend=cudss, GPU enabled) that are identical in every other
# respect, runs both through ca.x, and compares the three CA output CSVs
# (_delta/_flat, _buses, _convergence) to FP64 round-off with compare_ca_csv.py.
#
# The CPU run is the golden oracle; the GPU run is the candidate.  On a machine
# without a GPU (or a binary built without cuDSS) the "GPU" run transparently
# falls back to the CPU path, so this script still exercises the plumbing and
# reports PARITY OK (identical CPU vs CPU) -- useful in CI.
#
# Usage:
#   run_validation.sh <ca.x> <base_input.xml> [np] [atol] [rtol]
#
# Example:
#   run_validation.sh ./ca.x input_14.xml 1 1e-6 1e-6
# -----------------------------------------------------------------------------
set -u

CA_X=${1:?"usage: run_validation.sh <ca.x> <base_input.xml> [np] [atol] [rtol]"}
BASE_XML=${2:?"usage: run_validation.sh <ca.x> <base_input.xml> [np] [atol] [rtol]"}
NP=${3:-1}
ATOL=${4:-1e-6}
RTOL=${5:-1e-6}

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cmp_py="${here}/compare_ca_csv.py"

if [ ! -f "${BASE_XML}" ]; then echo "ERROR: no such file: ${BASE_XML}"; exit 2; fi

base="$(basename "${BASE_XML}" .xml)"
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

cpu_xml="${work}/${base}_cpu.xml"
gpu_xml="${work}/${base}_gpu.xml"

# Derive the CPU variant: force PETSc backend + GPU disabled + distinct prefix.
sed -E \
  -e 's#<Backend>[^<]*</Backend>#<Backend>petsc</Backend>#' \
  -e 's#<enabled>[^<]*</enabled>#<enabled>false</enabled>#' \
  -e "s#<outputFile>[^<]*</outputFile>#<outputFile>${base}_cpu_golden</outputFile>#" \
  "${BASE_XML}" > "${cpu_xml}"

# Derive the GPU variant: request cuDSS + GPU enabled + distinct prefix.
sed -E \
  -e 's#<Backend>[^<]*</Backend>#<Backend>cudss</Backend>#' \
  -e 's#<enabled>[^<]*</enabled>#<enabled>true</enabled>#' \
  -e "s#<outputFile>[^<]*</outputFile>#<outputFile>${base}_gpu_candidate</outputFile>#" \
  "${BASE_XML}" > "${gpu_xml}"

# If the base xml lacked a <Backend> tag, inject one so the CPU/GPU choice is
# unambiguous (idempotent when it already matched above).
grep -q "<Backend>" "${cpu_xml}" || \
  sed -i -E 's#(<LinearSolver>)#\1\n      <Backend>petsc</Backend>#' "${cpu_xml}"
grep -q "<Backend>" "${gpu_xml}" || \
  sed -i -E 's#(<LinearSolver>)#\1\n      <Backend>cudss</Backend>#' "${gpu_xml}"

run() {
  local xml="$1" tag="$2"
  echo "=== running ${tag}: mpirun -n ${NP} ${CA_X} ${xml} ==="
  if ! mpirun -n "${NP}" "${CA_X}" "${xml}"; then
    echo "ERROR: ${tag} run failed"; exit 3
  fi
}

run "${cpu_xml}" "CPU golden"
run "${gpu_xml}" "GPU candidate"

status=0
for suffix in _delta.csv _flat.csv _buses.csv _convergence.csv; do
  g="${base}_cpu_golden${suffix}"
  c="${base}_gpu_candidate${suffix}"
  if [ -f "${g}" ] && [ -f "${c}" ]; then
    echo "--- comparing ${suffix} ---"
    # --status-aware: for _convergence.csv, non-converged cases must agree on the
    # discrete outcome (converged flag, iteration count, status) but their
    # diverging-residual diagnostics are backend-dependent and are not compared.
    if ! python3 "${cmp_py}" "${g}" "${c}" --atol "${ATOL}" --rtol "${RTOL}" --status-aware; then
      status=1
    fi
  fi
done

if [ "${status}" -eq 0 ]; then
  echo "ALL PARITY CHECKS PASSED"
else
  echo "PARITY CHECKS FAILED"
fi
exit "${status}"
