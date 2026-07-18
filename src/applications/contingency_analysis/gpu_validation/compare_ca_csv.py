#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#     Copyright (c) 2013 Battelle Memorial Institute
#     Licensed under modified BSD License. A copy of this license can be found
#     in the LICENSE file in the top level directory of this distribution.
# -----------------------------------------------------------------------------
"""
Compare two GridPACK contingency-analysis CSV outputs for numerical parity.

This is the acceptance oracle for the GPU (cuDSS) path: a GPU run must reproduce
the CPU (PETSc/direct-LU) run's CSVs to FP64 round-off.  Because CA writes rows
in rank/completion order (documented in the CA README), rows are re-sorted by
their key columns (event_idx plus any bus/circuit/type/status columns) before
comparison, exactly as the README recommends
(`sort -t, -k1,1n`) but generalized to a stable multi-column key.

Comparison rules per column:
  * integer-valued columns (event_idx, from_bus, to_bus, iterations, ...):
        compared exactly, and used as sort keys
  * string columns (contingency, circuit_id, type, status, ...):
        compared exactly, and used as sort keys
  * floating-point columns (loadings, voltages, angles, mismatch, ...):
        compared with a mixed absolute/relative tolerance

Exit status is 0 on parity, 1 on any mismatch (suitable for CI / make check).

Usage:
    compare_ca_csv.py GOLDEN.csv CANDIDATE.csv [--atol 1e-6] [--rtol 1e-6]
                                               [--max-report 20]
"""

import argparse
import csv
import math
import sys


def _classify(values):
    """Return 'int', 'float', or 'str' for a list of raw string cell values."""
    is_int = True
    is_float = True
    for v in values:
        s = v.strip()
        if s == "":
            # treat empty as string-ish so it is compared exactly
            return "str"
        try:
            f = float(s)
        except ValueError:
            return "str"
        if is_int:
            try:
                int(s, 10)
            except ValueError:
                # not a plain integer literal (e.g. "1.0" or "3e2")
                if f != math.floor(f) or math.isinf(f) or math.isnan(f):
                    is_int = False
                else:
                    # numerically integral but written as float -> treat float
                    is_int = False
    return "int" if is_int else ("float" if is_float else "str")


def _read(path):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise SystemExit("ERROR: %s is empty" % path)
    header, data = rows[0], rows[1:]
    return header, data


def _column_types(header, data):
    types = {}
    for j, name in enumerate(header):
        col = [r[j] for r in data if j < len(r)]
        types[j] = _classify(col) if col else "str"
    return types


def _sort_key(header, types):
    # Key = every non-float column (int + str), in column order, so ordering is
    # deterministic regardless of the rank/completion order CA emitted rows in.
    key_cols = [j for j in range(len(header)) if types[j] != "float"]

    def keyfn(row):
        parts = []
        for j in key_cols:
            v = row[j] if j < len(row) else ""
            if types[j] == "int":
                try:
                    parts.append((0, int(v)))
                except ValueError:
                    parts.append((1, v))
            else:
                parts.append((1, v))
        return parts

    return keyfn


def _close(a, b, atol, rtol):
    try:
        fa, fb = float(a), float(b)
    except ValueError:
        return a.strip() == b.strip()
    if math.isnan(fa) and math.isnan(fb):
        return True
    if math.isinf(fa) or math.isinf(fb):
        return fa == fb
    return abs(fa - fb) <= (atol + rtol * abs(fb))


# For _convergence.csv rows that did NOT converge, the residual-magnitude
# diagnostics (final_tolerance, max_*_mismatch, worst-bus ids) are not physical
# quantities: a diverging Newton iteration amplifies round-off, so two different
# exact-LU backends (cuDSS vs SuperLU_DIST/KLU) legitimately report slightly
# different diverging residuals even though they agree the case diverged. In
# status-aware mode we therefore require only the DISCRETE outcome of a
# non-converged case to match exactly and skip its diagnostic columns.
_CONV_KEEP_FOR_DIVERGED = {
    "event_idx", "contingency", "type", "converged", "iterations", "status_code",
}


def _row_converged(row, conv_idx, status_idx):
    if conv_idx is not None and conv_idx < len(row):
        v = row[conv_idx].strip().lower()
        if v in ("true", "1", "yes"):
            return True
        if v in ("false", "0", "no"):
            return False
    if status_idx is not None and status_idx < len(row):
        return row[status_idx].strip().upper() in ("OK", "CONVERGED")
    return True


def compare(golden, candidate, atol, rtol, max_report, status_aware=False):
    gh, gd = _read(golden)
    ch, cd = _read(candidate)

    problems = []

    if gh != ch:
        problems.append("header mismatch:\n  golden=%s\n  candidate=%s" % (gh, ch))
        return problems

    # status-aware handling for _convergence.csv
    hl = [h.strip().lower() for h in gh]
    conv_idx = hl.index("converged") if "converged" in hl else None
    status_idx = hl.index("status_code") if "status_code" in hl else None
    status_aware = status_aware and (conv_idx is not None or status_idx is not None)

    if len(gd) != len(cd):
        problems.append("row count differs: golden=%d candidate=%d"
                        % (len(gd), len(cd)))
        # continue to compare the overlap for extra diagnostics
    types = _column_types(gh, gd)
    keyfn = _sort_key(gh, types)
    gd_sorted = sorted(gd, key=keyfn)
    cd_sorted = sorted(cd, key=keyfn)

    n = min(len(gd_sorted), len(cd_sorted))
    ncol = len(gh)
    mismatches = 0
    max_abs = 0.0
    max_rel = 0.0
    for i in range(n):
        gr, cr = gd_sorted[i], cd_sorted[i]
        # For a non-converged case in status-aware mode, compare only the
        # discrete outcome and skip backend-dependent diagnostic columns.
        skip_diag = status_aware and not _row_converged(gr, conv_idx, status_idx)
        for j in range(ncol):
            if skip_diag and hl[j] not in _CONV_KEEP_FOR_DIVERGED:
                continue
            gv = gr[j] if j < len(gr) else ""
            cv = cr[j] if j < len(cr) else ""
            if types[j] == "float":
                if not _close(gv, cv, atol, rtol):
                    mismatches += 1
                    try:
                        d = abs(float(gv) - float(cv))
                        max_abs = max(max_abs, d)
                        if float(gv) != 0.0:
                            max_rel = max(max_rel, d / abs(float(gv)))
                    except ValueError:
                        pass
                    if len(problems) < max_report:
                        problems.append(
                            "row %d col '%s': golden=%s candidate=%s (|d|=%s)"
                            % (i, gh[j], gv, cv,
                               (str(abs(float(gv) - float(cv)))
                                if _is_num(gv) and _is_num(cv) else "n/a")))
            else:
                if gv.strip() != cv.strip():
                    mismatches += 1
                    if len(problems) < max_report:
                        problems.append(
                            "row %d col '%s': golden=%r candidate=%r"
                            % (i, gh[j], gv, cv))

    if mismatches:
        problems.append("total cell mismatches: %d (max |abs|=%.3e, max rel=%.3e)"
                        % (mismatches, max_abs, max_rel))
    return problems


def _is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("golden")
    ap.add_argument("candidate")
    ap.add_argument("--atol", type=float, default=1e-6,
                    help="absolute tolerance for float columns (default 1e-6)")
    ap.add_argument("--rtol", type=float, default=1e-6,
                    help="relative tolerance for float columns (default 1e-6)")
    ap.add_argument("--max-report", type=int, default=20,
                    help="max number of mismatches to print (default 20)")
    ap.add_argument("--status-aware", action="store_true",
                    help="for _convergence.csv, only require the discrete outcome "
                         "of non-converged cases to match (skip backend-dependent "
                         "diverging-residual diagnostics)")
    args = ap.parse_args(argv)

    problems = compare(args.golden, args.candidate,
                       args.atol, args.rtol, args.max_report,
                       status_aware=args.status_aware)
    if problems:
        print("PARITY FAIL: %s vs %s" % (args.golden, args.candidate))
        for p in problems:
            print("  " + p)
        return 1
    print("PARITY OK: %s == %s (atol=%g rtol=%g)"
          % (args.golden, args.candidate, args.atol, args.rtol))
    return 0


if __name__ == "__main__":
    sys.exit(main())
