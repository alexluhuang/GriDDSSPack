#!/bin/sh
set -eu

if [ "$(uname -m)" != "aarch64" ]; then
  echo "error: the cuDSS image must be built natively on an aarch64 host" >&2
  echo "QEMU and amd64 cross-builds are intentionally unsupported" >&2
  exit 1
fi

image=${1:-gridpack-cudss-arm64:cuda13-cudss0.8}
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(dirname -- "${script_dir}")

cd "${repo_root}"

exec docker build \
  --pull \
  --platform=linux/arm64 \
  --file=Dockerfile.cudss-arm64 \
  --tag="${image}" \
  .
