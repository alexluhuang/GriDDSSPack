#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
data_dir=${1:-/home/alh360/Documents/GriDSSPack_Docker_Container/verification_gpu_7k}
image_name=${GRIDSS_COMPARE_IMAGE:-griddsspack-result-compare:rapids-26.06}

mkdir -p "${script_dir}/results" "${script_dir}/scratch"

docker build \
  --platform linux/arm64 \
  -t "${image_name}" \
  "${script_dir}"

rapids_gid=$(docker run --rm --entrypoint id "${image_name}" -g)

docker run --rm --gpus all \
  --shm-size=16g \
  --ulimit memlock=-1:-1 \
  --user "$(id -u):${rapids_gid}" \
  -e HOME=/tmp \
  -v "${data_dir}:/data:ro" \
  -v "${script_dir}/results:/output" \
  -v "${script_dir}/scratch:/scratch" \
  "${image_name}" \
  --gpu-dir /data \
  --cpu-dir /data/cpu_results \
  --output-dir /output \
  --temp-dir /scratch
