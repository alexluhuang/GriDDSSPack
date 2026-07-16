#!/bin/sh
set -eu

case "${1:-}" in
  /bin/* | /usr/bin/* | bash | sh | mpiexec | mpirun)
    exec "$@"
    ;;
  ca.x)
    shift
    ;;
esac

exec /opt/gridpack/bin/ca.x "$@"
