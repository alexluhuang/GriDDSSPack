#!/usr/bin/env python3
"""Compare GriDSSPack GPU and stock GridPACK CPU result sets with RAPIDS."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cudf
import dask

dask.config.set(
    {
        "dataframe.backend": "cudf",
        "dataframe.query-planning": True,
    }
)
import dask.dataframe as dd  # noqa: E402


DEFAULT_DATA_DIR = Path(
    "/home/alh360/Documents/GriDSSPack_Docker_Container/verification_gpu_training"
)

FLAT_DTYPES = {
    "event_idx": "int32",
    "contingency": "str",
    "from_bus": "int64",
    "to_bus": "int64",
    "circuit_id": "str",
    "p_from_mw": "float64",
    "q_from_mvar": "float64",
    "mva_from": "float64",
    "rate_mva": "float64",
    "loading_percent": "float64",
    "viol": "int8",
    "v_from_pu": "float64",
    "v_to_pu": "float64",
    "ang_from_deg": "float64",
    "ang_to_deg": "float64",
}

FLAT_KEYS = ["event_idx", "from_bus", "to_bus", "circuit_id"]

FLAT_NUMERIC_COLUMNS = {
    "p_from_mw": {"unit": "MW"},
    "q_from_mvar": {"unit": "MVAr"},
    "mva_from": {"unit": "MVA"},
    "rate_mva": {"unit": "MVA"},
    "loading_percent": {"unit": "percent"},
    "v_from_pu": {"unit": "p.u."},
    "v_to_pu": {"unit": "p.u."},
    "ang_from_deg": {"unit": "degrees", "circular": True},
    "ang_to_deg": {"unit": "degrees", "circular": True},
}

CONVERGENCE_DTYPES = {
    "event_idx": "int32",
    "contingency": "str",
    "type": "str",
    "converged": "bool",
    "iterations": "int32",
    "final_tolerance": "float64",
    "max_p_bus": "int64",
    "max_p_mismatch": "float64",
    "max_q_bus": "int64",
    "max_q_mismatch": "float64",
    "status_code": "str",
}

CONVERGENCE_NUMERIC_COLUMNS = {
    "iterations": {"unit": "iterations"},
    "final_tolerance": {"unit": "p.u."},
    "max_p_mismatch": {"unit": "p.u."},
    "max_q_mismatch": {"unit": "p.u."},
}

BUS_DTYPES = {
    "bus_id": "int64",
    "bus_name": "str",
    "base_kv": "float64",
    "area": "int32",
    "zone": "int32",
    "owner": "int32",
    "area_name": "str",
    "zone_name": "str",
    "owner_name": "str",
}

TIMING_HEADER_RE = re.compile(r"^Timing statistics for:\s*(.+?)\s*$")
TIMING_VALUE_RE = re.compile(
    r"^\s*(Average|Maximum|Minimum) time:\s*([0-9.eE+-]+)\s*$"
)
TIMING_RMS_RE = re.compile(r"^\s*RMS deviation:\s*([0-9.eE+-]+)\s*$")
TASK_RE = re.compile(r"Number of tasks on process\s+(\d+):\s+(\d+)")
TOTAL_CONTINGENCIES_RE = re.compile(r"Total contingencies to analyze:\s*(\d+)")
GPU_SUMMARY_RE = re.compile(r"\[GPU all-rank summary\]\s*(.*)")
KEY_VALUE_RE = re.compile(r"([A-Za-z_]+)=([0-9]+)")


def scalar(value: Any) -> Any:
    """Move a scalar result to a JSON-safe Python scalar."""
    if value is None:
        return None
    try:
        value = value.item()
    except AttributeError:
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def percent(numerator: float, denominator: float) -> float | None:
    value = ratio(numerator, denominator)
    return None if value is None else 100.0 * value


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return next(csv.reader(stream))


def require_columns(path: Path, expected: Iterable[str]) -> list[str]:
    header = read_header(path)
    missing = sorted(set(expected) - set(header))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return header


@dataclass
class MetricAccumulator:
    """Streaming aggregate for one pair of corresponding numerical columns."""

    zero_threshold: float
    absolute_tolerance: float
    relative_tolerance: float
    circular: bool = False
    compared: int = 0
    missing_either: int = 0
    reference_nonzero: int = 0
    symmetric_nonzero: int = 0
    within_tolerance: int = 0
    cpu_sum: float = 0.0
    gpu_sum: float = 0.0
    abs_diff_sum: float = 0.0
    signed_diff_sum: float = 0.0
    squared_diff_sum: float = 0.0
    max_abs_diff: float = 0.0
    abs_pct_sum: float = 0.0
    signed_pct_sum: float = 0.0
    symmetric_abs_pct_sum: float = 0.0

    def update(self, gpu_series: cudf.Series, cpu_series: cudf.Series) -> None:
        valid = gpu_series.notna() & cpu_series.notna()
        valid_count = int(scalar(valid.sum()))
        self.missing_either += len(gpu_series) - valid_count
        if valid_count == 0:
            return

        gpu_values = gpu_series[valid].astype("float64")
        cpu_values = cpu_series[valid].astype("float64")
        difference = gpu_values - cpu_values
        if self.circular:
            difference = ((difference + 180.0) % 360.0) - 180.0
        absolute_difference = difference.abs()

        self.compared += valid_count
        self.cpu_sum += float(scalar(cpu_values.sum()))
        self.gpu_sum += float(scalar(gpu_values.sum()))
        self.abs_diff_sum += float(scalar(absolute_difference.sum()))
        self.signed_diff_sum += float(scalar(difference.sum()))
        self.squared_diff_sum += float(scalar((difference * difference).sum()))
        bucket_max = float(scalar(absolute_difference.max()))
        self.max_abs_diff = max(self.max_abs_diff, bucket_max)

        tolerance = self.absolute_tolerance + self.relative_tolerance * cpu_values.abs()
        self.within_tolerance += int(scalar((absolute_difference <= tolerance).sum()))

        reference_mask = cpu_values.abs() > self.zero_threshold
        reference_count = int(scalar(reference_mask.sum()))
        self.reference_nonzero += reference_count
        if reference_count:
            self.abs_pct_sum += float(
                scalar(
                    (
                        absolute_difference[reference_mask]
                        / cpu_values[reference_mask].abs()
                        * 100.0
                    ).sum()
                )
            )
            self.signed_pct_sum += float(
                scalar(
                    (
                        difference[reference_mask]
                        / cpu_values[reference_mask]
                        * 100.0
                    ).sum()
                )
            )

        symmetric_denominator = cpu_values.abs() + gpu_values.abs()
        symmetric_mask = symmetric_denominator > self.zero_threshold
        symmetric_count = int(scalar(symmetric_mask.sum()))
        self.symmetric_nonzero += symmetric_count
        if symmetric_count:
            self.symmetric_abs_pct_sum += float(
                scalar(
                    (
                        200.0
                        * absolute_difference[symmetric_mask]
                        / symmetric_denominator[symmetric_mask]
                    ).sum()
                )
            )

    def finish(self) -> dict[str, Any]:
        return {
            "compared_values": self.compared,
            "missing_either": self.missing_either,
            "cpu_mean": ratio(self.cpu_sum, self.compared),
            "gpu_mean": ratio(self.gpu_sum, self.compared),
            "mean_signed_difference": ratio(self.signed_diff_sum, self.compared),
            "mean_absolute_difference": ratio(self.abs_diff_sum, self.compared),
            "root_mean_squared_difference": (
                math.sqrt(self.squared_diff_sum / self.compared)
                if self.compared
                else None
            ),
            "maximum_absolute_difference": (
                self.max_abs_diff if self.compared else None
            ),
            "mean_absolute_percent_difference": ratio(
                self.abs_pct_sum, self.reference_nonzero
            ),
            "mean_signed_percent_change": ratio(
                self.signed_pct_sum, self.reference_nonzero
            ),
            "symmetric_mean_absolute_percent_difference": ratio(
                self.symmetric_abs_pct_sum, self.symmetric_nonzero
            ),
            "reference_zero_excluded": self.compared - self.reference_nonzero,
            "within_tolerance_count": self.within_tolerance,
            "within_tolerance_percent": percent(
                self.within_tolerance, self.compared
            ),
            "circular_difference": self.circular,
        }


def stage_flat_csv(
    source: Path,
    destination: Path,
    block_size_bytes: int,
    bucket_events: int,
) -> None:
    """Convert a large CSV to event-bucketed Parquet without materializing it."""
    require_columns(source, FLAT_DTYPES)
    print(f"Staging {source} into GPU-readable event buckets...")
    frame = dd.read_csv(
        str(source),
        blocksize=block_size_bytes,
        dtype=FLAT_DTYPES,
    )
    frame["event_bucket"] = (frame["event_idx"] // bucket_events).astype("int32")
    frame.to_parquet(
        str(destination),
        compression="snappy",
        partition_on=["event_bucket"],
        write_index=False,
    )


def discover_buckets(dataset: Path) -> dict[int, list[Path]]:
    result: dict[int, list[Path]] = {}
    for directory in dataset.rglob("event_bucket=*"):
        if not directory.is_dir():
            continue
        try:
            bucket = int(directory.name.split("=", 1)[1])
        except (IndexError, ValueError):
            continue
        files = sorted(directory.glob("*.parquet"))
        if files:
            result.setdefault(bucket, []).extend(files)
    return result


def normalize_flat(frame: cudf.DataFrame) -> cudf.DataFrame:
    frame["contingency_key"] = frame["contingency"].str.strip()
    frame["circuit_id"] = frame["circuit_id"].str.strip()
    return frame.drop(columns=["contingency", "event_bucket"], errors="ignore")


def duplicate_count(frame: cudf.DataFrame, keys: list[str]) -> int:
    if frame.empty:
        return 0
    return int(scalar(frame.duplicated(subset=keys, keep=False).sum()))


def compare_flat_results(
    gpu_dataset: Path,
    cpu_dataset: Path,
    zero_threshold: float,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    gpu_buckets = discover_buckets(gpu_dataset)
    cpu_buckets = discover_buckets(cpu_dataset)
    buckets = sorted(set(gpu_buckets) | set(cpu_buckets))
    if not buckets:
        raise RuntimeError("No event-bucketed Parquet files were created")

    accumulators = {
        column: MetricAccumulator(
            zero_threshold=zero_threshold,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            circular=bool(specification.get("circular", False)),
        )
        for column, specification in FLAT_NUMERIC_COLUMNS.items()
    }
    coverage = {
        "gpu_rows": 0,
        "cpu_rows": 0,
        "matched_rows": 0,
        "gpu_only_rows": 0,
        "cpu_only_rows": 0,
        "gpu_duplicate_key_rows": 0,
        "cpu_duplicate_key_rows": 0,
        "contingency_name_mismatches": 0,
        "violation_flag_mismatches": 0,
    }
    gpu_only_contingencies: dict[tuple[int, str], int] = {}
    cpu_only_contingencies: dict[tuple[int, str], int] = {}
    gpu_only_branches: dict[tuple[int, int, str], int] = {}
    cpu_only_branches: dict[tuple[int, int, str], int] = {}

    for position, bucket in enumerate(buckets, start=1):
        gpu_files = [str(path) for path in gpu_buckets.get(bucket, [])]
        cpu_files = [str(path) for path in cpu_buckets.get(bucket, [])]
        gpu_frame = (
            normalize_flat(cudf.read_parquet(gpu_files))
            if gpu_files
            else cudf.DataFrame()
        )
        cpu_frame = (
            normalize_flat(cudf.read_parquet(cpu_files))
            if cpu_files
            else cudf.DataFrame()
        )

        coverage["gpu_rows"] += len(gpu_frame)
        coverage["cpu_rows"] += len(cpu_frame)
        if gpu_frame.empty:
            coverage["cpu_only_rows"] += len(cpu_frame)
            continue
        if cpu_frame.empty:
            coverage["gpu_only_rows"] += len(gpu_frame)
            continue

        gpu_duplicates = duplicate_count(gpu_frame, FLAT_KEYS)
        cpu_duplicates = duplicate_count(cpu_frame, FLAT_KEYS)
        coverage["gpu_duplicate_key_rows"] += gpu_duplicates
        coverage["cpu_duplicate_key_rows"] += cpu_duplicates
        if gpu_duplicates or cpu_duplicates:
            raise ValueError(
                "Flat comparison key is not unique in event bucket "
                f"{bucket}: gpu duplicate rows={gpu_duplicates}, "
                f"cpu duplicate rows={cpu_duplicates}"
            )

        gpu_frame["present_gpu"] = 1
        cpu_frame["present_cpu"] = 1
        merged = gpu_frame.merge(
            cpu_frame,
            on=FLAT_KEYS,
            how="outer",
            suffixes=("_gpu", "_cpu"),
        )
        gpu_present = merged["present_gpu"].notna()
        cpu_present = merged["present_cpu"].notna()
        matched_mask = gpu_present & cpu_present
        matched_count = int(scalar(matched_mask.sum()))
        coverage["matched_rows"] += matched_count
        coverage["gpu_only_rows"] += int(scalar((gpu_present & ~cpu_present).sum()))
        coverage["cpu_only_rows"] += int(scalar((~gpu_present & cpu_present).sum()))

        if matched_count:
            matched = merged[matched_mask]
            coverage["contingency_name_mismatches"] += int(
                scalar(
                    (
                        matched["contingency_key_gpu"]
                        != matched["contingency_key_cpu"]
                    ).sum()
                )
            )
            coverage["violation_flag_mismatches"] += int(
                scalar((matched["viol_gpu"] != matched["viol_cpu"]).sum())
            )
            for column, accumulator in accumulators.items():
                accumulator.update(
                    matched[f"{column}_gpu"],
                    matched[f"{column}_cpu"],
                )

        for side, mask, event_target, branch_target in (
            (
                "gpu",
                gpu_present & ~cpu_present,
                gpu_only_contingencies,
                gpu_only_branches,
            ),
            (
                "cpu",
                ~gpu_present & cpu_present,
                cpu_only_contingencies,
                cpu_only_branches,
            ),
        ):
            if not bool(scalar(mask.any())):
                continue
            name_column = f"contingency_key_{side}"
            unmatched = merged[mask][["event_idx", name_column]]
            grouped = (
                unmatched.groupby(["event_idx", name_column])
                .size()
                .reset_index(name="row_count")
            )
            for record in grouped.to_pandas().to_dict(orient="records"):
                key = (int(record["event_idx"]), str(record[name_column]))
                event_target[key] = event_target.get(key, 0) + int(
                    record["row_count"]
                )
            branch_counts = (
                merged[mask]
                .groupby(["from_bus", "to_bus", "circuit_id"])
                .size()
                .reset_index(name="row_count")
            )
            for record in branch_counts.to_pandas().to_dict(orient="records"):
                key = (
                    int(record["from_bus"]),
                    int(record["to_bus"]),
                    str(record["circuit_id"]),
                )
                branch_target[key] = branch_target.get(key, 0) + int(
                    record["row_count"]
                )

        if position == 1 or position % 10 == 0 or position == len(buckets):
            print(
                f"Compared {position}/{len(buckets)} event buckets; "
                f"{coverage['matched_rows']:,} rows matched"
            )
        del gpu_frame, cpu_frame, merged
        gc.collect()

    coverage["gpu_row_match_percent"] = percent(
        coverage["matched_rows"], coverage["gpu_rows"]
    )
    coverage["cpu_row_match_percent"] = percent(
        coverage["matched_rows"], coverage["cpu_rows"]
    )
    coverage["violation_flag_mismatch_percent"] = percent(
        coverage["violation_flag_mismatches"], coverage["matched_rows"]
    )
    metrics = {}
    for column, accumulator in accumulators.items():
        metrics[column] = {
            "unit": FLAT_NUMERIC_COLUMNS[column]["unit"],
            **accumulator.finish(),
        }
    return {
        "coverage": coverage,
        "gpu_only_contingencies": [
            {
                "event_idx": event_idx,
                "contingency": contingency,
                "row_count": row_count,
            }
            for (event_idx, contingency), row_count in sorted(
                gpu_only_contingencies.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        "cpu_only_contingencies": [
            {
                "event_idx": event_idx,
                "contingency": contingency,
                "row_count": row_count,
            }
            for (event_idx, contingency), row_count in sorted(
                cpu_only_contingencies.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        "gpu_only_branches": [
            {
                "from_bus": from_bus,
                "to_bus": to_bus,
                "circuit_id": circuit_id,
                "row_count": row_count,
            }
            for (from_bus, to_bus, circuit_id), row_count in sorted(
                gpu_only_branches.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        "cpu_only_branches": [
            {
                "from_bus": from_bus,
                "to_bus": to_bus,
                "circuit_id": circuit_id,
                "row_count": row_count,
            }
            for (from_bus, to_bus, circuit_id), row_count in sorted(
                cpu_only_branches.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        "numeric_metrics": metrics,
    }


def normalize_convergence(frame: cudf.DataFrame) -> cudf.DataFrame:
    frame["contingency_key"] = frame["contingency"].str.strip()
    frame["type"] = frame["type"].str.strip()
    frame["status_code"] = frame["status_code"].str.strip()
    return frame.drop(columns=["contingency"])


def series_stats(series: cudf.Series) -> dict[str, Any]:
    values = series.dropna().astype("float64")
    if values.empty:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {
        "count": len(values),
        "mean": float(scalar(values.mean())),
        "median": float(scalar(values.median())),
        "p95": float(scalar(values.quantile(0.95))),
        "minimum": float(scalar(values.min())),
        "maximum": float(scalar(values.max())),
    }


def count_records(frame: cudf.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    counts = frame.groupby(columns, dropna=False).size().reset_index(name="count")
    return counts.sort_values(columns).to_pandas().to_dict(orient="records")


def compare_convergence(
    gpu_path: Path,
    cpu_path: Path,
    zero_threshold: float,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    require_columns(gpu_path, CONVERGENCE_DTYPES)
    require_columns(cpu_path, CONVERGENCE_DTYPES)
    gpu = normalize_convergence(cudf.read_csv(gpu_path, dtype=CONVERGENCE_DTYPES))
    cpu = normalize_convergence(cudf.read_csv(cpu_path, dtype=CONVERGENCE_DTYPES))
    keys = ["contingency_key", "type"]
    gpu_duplicates = duplicate_count(gpu, keys)
    cpu_duplicates = duplicate_count(cpu, keys)
    if gpu_duplicates or cpu_duplicates:
        raise ValueError(
            "Convergence comparison key is not unique: "
            f"gpu duplicate rows={gpu_duplicates}, cpu duplicate rows={cpu_duplicates}"
        )

    gpu_counts = {
        "rows": len(gpu),
        "converged": int(scalar(gpu["converged"].sum())),
        "iterations": series_stats(gpu["iterations"]),
        "status_counts": count_records(gpu, ["status_code"]),
    }
    cpu_counts = {
        "rows": len(cpu),
        "converged": int(scalar(cpu["converged"].sum())),
        "iterations": series_stats(cpu["iterations"]),
        "status_counts": count_records(cpu, ["status_code"]),
    }
    gpu_counts["converged_percent"] = percent(
        gpu_counts["converged"], gpu_counts["rows"]
    )
    cpu_counts["converged_percent"] = percent(
        cpu_counts["converged"], cpu_counts["rows"]
    )

    gpu["present_gpu"] = 1
    cpu["present_cpu"] = 1
    merged = gpu.merge(cpu, on=keys, how="outer", suffixes=("_gpu", "_cpu"))
    gpu_present = merged["present_gpu"].notna()
    cpu_present = merged["present_cpu"].notna()
    matched_mask = gpu_present & cpu_present
    matched = merged[matched_mask]

    gpu_converged = matched["converged_gpu"]
    cpu_converged = matched["converged_cpu"]
    outcome_matrix = {
        "both_converged": int(scalar((gpu_converged & cpu_converged).sum())),
        "both_failed": int(scalar((~gpu_converged & ~cpu_converged).sum())),
        "gpu_only_converged": int(
            scalar((gpu_converged & ~cpu_converged).sum())
        ),
        "cpu_only_converged": int(
            scalar((~gpu_converged & cpu_converged).sum())
        ),
    }
    agreement = outcome_matrix["both_converged"] + outcome_matrix["both_failed"]

    def convergence_numeric_metrics(
        source: cudf.DataFrame,
    ) -> dict[str, dict[str, Any]]:
        result = {}
        for column, specification in CONVERGENCE_NUMERIC_COLUMNS.items():
            accumulator = MetricAccumulator(
                zero_threshold=zero_threshold,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            accumulator.update(source[f"{column}_gpu"], source[f"{column}_cpu"])
            result[column] = {
                "unit": specification["unit"],
                **accumulator.finish(),
            }
        return result

    matched_ok = matched[
        matched["converged_gpu"]
        & matched["converged_cpu"]
        & (matched["status_code_gpu"] == "OK")
        & (matched["status_code_cpu"] == "OK")
    ]
    numeric_metrics = convergence_numeric_metrics(matched_ok)
    all_case_numeric_metrics = convergence_numeric_metrics(matched)

    transitions_frame = matched[
        ["status_code_cpu", "status_code_gpu"]
    ].fillna("<null>")
    status_transitions = count_records(
        transitions_frame, ["status_code_cpu", "status_code_gpu"]
    )
    coverage = {
        "matched_rows": len(matched),
        "gpu_only_rows": int(scalar((gpu_present & ~cpu_present).sum())),
        "cpu_only_rows": int(scalar((~gpu_present & cpu_present).sum())),
        "event_index_mismatches": int(
            scalar((matched["event_idx_gpu"] != matched["event_idx_cpu"]).sum())
        ),
    }
    return {
        "coverage": coverage,
        "gpu": gpu_counts,
        "cpu": cpu_counts,
        "convergence_outcome_matrix": outcome_matrix,
        "convergence_agreement_percent": percent(agreement, len(matched)),
        "numeric_metric_scope": "matched events with status_code OK in both outputs",
        "numeric_metric_event_count": len(matched_ok),
        "numeric_metrics": numeric_metrics,
        "all_case_numeric_metrics": all_case_numeric_metrics,
        "max_p_bus_mismatches": int(
            scalar((matched["max_p_bus_gpu"] != matched["max_p_bus_cpu"]).sum())
        ),
        "max_q_bus_mismatches": int(
            scalar((matched["max_q_bus_gpu"] != matched["max_q_bus_cpu"]).sum())
        ),
        "status_transitions": status_transitions,
    }


def annotate_flat_coverage(
    flat_report: dict[str, Any],
    gpu_convergence_path: Path,
    cpu_convergence_path: Path,
) -> None:
    """Attach convergence outcomes to unmatched flat-output contingencies."""
    gpu = cudf.read_csv(gpu_convergence_path, dtype=CONVERGENCE_DTYPES)
    cpu = cudf.read_csv(cpu_convergence_path, dtype=CONVERGENCE_DTYPES)
    gpu_records = {
        int(record["event_idx"]): record
        for record in gpu[
            ["event_idx", "converged", "status_code"]
        ].to_pandas().to_dict(orient="records")
    }
    cpu_records = {
        int(record["event_idx"]): record
        for record in cpu[
            ["event_idx", "converged", "status_code"]
        ].to_pandas().to_dict(orient="records")
    }
    for records in (
        flat_report["gpu_only_contingencies"],
        flat_report["cpu_only_contingencies"],
    ):
        for record in records:
            event_idx = record["event_idx"]
            gpu_record = gpu_records.get(event_idx)
            cpu_record = cpu_records.get(event_idx)
            record["gpu_converged"] = (
                bool(gpu_record["converged"]) if gpu_record else None
            )
            record["gpu_status_code"] = (
                str(gpu_record["status_code"]).strip() if gpu_record else None
            )
            record["cpu_converged"] = (
                bool(cpu_record["converged"]) if cpu_record else None
            )
            record["cpu_status_code"] = (
                str(cpu_record["status_code"]).strip() if cpu_record else None
            )


def compare_buses(
    gpu_path: Path,
    cpu_path: Path,
    zero_threshold: float,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    require_columns(gpu_path, BUS_DTYPES)
    require_columns(cpu_path, BUS_DTYPES)
    gpu = cudf.read_csv(gpu_path, dtype=BUS_DTYPES)
    cpu = cudf.read_csv(cpu_path, dtype=BUS_DTYPES)
    for frame in (gpu, cpu):
        for column in ["bus_name", "area_name", "zone_name", "owner_name"]:
            frame[column] = frame[column].str.strip()
    if duplicate_count(gpu, ["bus_id"]) or duplicate_count(cpu, ["bus_id"]):
        raise ValueError("bus_id must be unique in both bus metadata files")

    gpu["present_gpu"] = 1
    cpu["present_cpu"] = 1
    merged = gpu.merge(cpu, on=["bus_id"], how="outer", suffixes=("_gpu", "_cpu"))
    gpu_present = merged["present_gpu"].notna()
    cpu_present = merged["present_cpu"].notna()
    matched = merged[gpu_present & cpu_present]

    base_kv = MetricAccumulator(
        zero_threshold=zero_threshold,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    base_kv.update(matched["base_kv_gpu"], matched["base_kv_cpu"])
    categorical_mismatches = {}
    for column in [
        "bus_name",
        "area",
        "zone",
        "owner",
        "area_name",
        "zone_name",
        "owner_name",
    ]:
        categorical_mismatches[column] = int(
            scalar((matched[f"{column}_gpu"] != matched[f"{column}_cpu"]).sum())
        )
    return {
        "coverage": {
            "gpu_rows": len(gpu),
            "cpu_rows": len(cpu),
            "matched_rows": len(matched),
            "gpu_only_rows": int(scalar((gpu_present & ~cpu_present).sum())),
            "cpu_only_rows": int(scalar((~gpu_present & cpu_present).sum())),
        },
        "numeric_metrics": {
            "base_kv": {"unit": "kV", **base_kv.finish()}
        },
        "categorical_mismatches": categorical_mismatches,
    }


def parse_log(path: Path) -> dict[str, Any]:
    timers: dict[str, dict[str, float]] = {}
    tasks: dict[str, int] = {}
    total_contingencies = None
    gpu_summary: dict[str, int] = {}
    current_timer: str | None = None

    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.rstrip()
            timing_header = TIMING_HEADER_RE.match(line)
            if timing_header:
                current_timer = timing_header.group(1)
                timers.setdefault(current_timer, {})
                continue
            timing_value = TIMING_VALUE_RE.match(line)
            if current_timer and timing_value:
                timers[current_timer][timing_value.group(1).lower()] = float(
                    timing_value.group(2)
                )
                continue
            rms_value = TIMING_RMS_RE.match(line)
            if current_timer and rms_value:
                timers[current_timer]["rms_deviation"] = float(rms_value.group(1))
                current_timer = None
                continue

            task = TASK_RE.search(line)
            if task:
                tasks[task.group(1)] = int(task.group(2))
            total = TOTAL_CONTINGENCIES_RE.search(line)
            if total:
                total_contingencies = int(total.group(1))
            summary = GPU_SUMMARY_RE.search(line)
            if summary:
                gpu_summary = {
                    key: int(value)
                    for key, value in KEY_VALUE_RE.findall(summary.group(1))
                }

    task_values = list(tasks.values())
    task_stats: dict[str, Any] = {"by_rank": tasks}
    if task_values:
        task_mean = sum(task_values) / len(task_values)
        variance = sum((value - task_mean) ** 2 for value in task_values) / len(
            task_values
        )
        task_stats.update(
            {
                "ranks": len(task_values),
                "total": sum(task_values),
                "minimum": min(task_values),
                "maximum": max(task_values),
                "mean": task_mean,
                "standard_deviation": math.sqrt(variance),
                "coefficient_of_variation_percent": (
                    100.0 * math.sqrt(variance) / task_mean if task_mean else None
                ),
                "max_to_mean_ratio": ratio(max(task_values), task_mean),
            }
        )
    return {
        "path": str(path),
        "total_contingencies": total_contingencies,
        "timers": timers,
        "tasks": task_stats,
        "gpu_summary": gpu_summary,
    }


def compare_performance(gpu_log: Path, cpu_log: Path) -> dict[str, Any]:
    gpu = parse_log(gpu_log)
    cpu = parse_log(cpu_log)
    timer_names = sorted(
        set(gpu["timers"]) | set(cpu["timers"]),
        key=lambda name: (name != "Total Application", name),
    )
    timers = []
    for name in timer_names:
        gpu_timer = gpu["timers"].get(name, {})
        cpu_timer = cpu["timers"].get(name, {})
        row: dict[str, Any] = {
            "timer": name,
            "gpu_average_seconds": gpu_timer.get("average"),
            "cpu_average_seconds": cpu_timer.get("average"),
            "gpu_maximum_seconds": gpu_timer.get("maximum"),
            "cpu_maximum_seconds": cpu_timer.get("maximum"),
            "gpu_minimum_seconds": gpu_timer.get("minimum"),
            "cpu_minimum_seconds": cpu_timer.get("minimum"),
        }
        for statistic in ("average", "maximum"):
            gpu_value = gpu_timer.get(statistic)
            cpu_value = cpu_timer.get(statistic)
            row[f"{statistic}_speedup_cpu_over_gpu"] = (
                ratio(cpu_value, gpu_value)
                if gpu_value is not None and cpu_value is not None
                else None
            )
            row[f"{statistic}_time_reduction_percent"] = (
                100.0 * (cpu_value - gpu_value) / cpu_value
                if gpu_value is not None and cpu_value
                else None
            )
        timers.append(row)
    return {"gpu": gpu, "cpu": cpu, "timer_comparison": timers}


def metric_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset in ("flat", "convergence", "buses"):
        for column, metric in report[dataset].get("numeric_metrics", {}).items():
            rows.append({"dataset": dataset, "column": column, **metric})
    return rows


def flat_coverage_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for side in ("gpu", "cpu"):
        for record in report["flat"][f"{side}_only_contingencies"]:
            rows.append({"present_only_in": side, **record})
    return rows


def flat_branch_coverage_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for side in ("gpu", "cpu"):
        for record in report["flat"][f"{side}_only_branches"]:
            rows.append({"present_only_in": side, **record})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def display_number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def build_markdown(report: dict[str, Any]) -> str:
    flat = report["flat"]
    convergence = report["convergence"]
    performance = report["performance"]
    total_timer = next(
        (
            row
            for row in performance["timer_comparison"]
            if row["timer"] == "Total Application"
        ),
        {},
    )
    lines = [
        "# GriDSSPack GPU versus stock GridPACK CPU",
        "",
        "## Row coverage",
        "",
        "| Dataset | GPU rows | CPU rows | Matched | GPU only | CPU only |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Flat branch results | {flat['coverage']['gpu_rows']:,} | "
            f"{flat['coverage']['cpu_rows']:,} | "
            f"{flat['coverage']['matched_rows']:,} | "
            f"{flat['coverage']['gpu_only_rows']:,} | "
            f"{flat['coverage']['cpu_only_rows']:,} |"
        ),
        (
            f"| Convergence | {convergence['gpu']['rows']:,} | "
            f"{convergence['cpu']['rows']:,} | "
            f"{convergence['coverage']['matched_rows']:,} | "
            f"{convergence['coverage']['gpu_only_rows']:,} | "
            f"{convergence['coverage']['cpu_only_rows']:,} |"
        ),
        "",
        (
            f"Flat rows present only in the GPU output span "
            f"{len(flat['gpu_only_contingencies']):,} contingencies; flat rows "
            f"present only in the CPU output span "
            f"{len(flat['cpu_only_contingencies']):,} contingencies."
        ),
        (
            f"Those unmatched rows represent "
            f"{len(flat['gpu_only_branches']):,} GPU-only branch keys and "
            f"{len(flat['cpu_only_branches']):,} CPU-only branch keys."
        ),
        (
            f"Matched rows with different violation flags: "
            f"{flat['coverage']['violation_flag_mismatches']:,} "
            f"({display_number(flat['coverage']['violation_flag_mismatch_percent'])}%)."
        ),
        "",
        "## Numerical differences",
        "",
        (
            "MAPD is the mean absolute percent difference relative to the stock "
            "CPU value. CPU reference values at or below the configured zero "
            "threshold are excluded from MAPD and counted separately."
        ),
        "",
        "| Column | Unit | CPU mean | GPU mean | Mean absolute difference | MAPD | Symmetric MAPD | Max absolute difference |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for column, metric in flat["numeric_metrics"].items():
        lines.append(
            f"| `{column}` | {metric['unit']} | "
            f"{display_number(metric['cpu_mean'])} | "
            f"{display_number(metric['gpu_mean'])} | "
            f"{display_number(metric['mean_absolute_difference'])} | "
            f"{display_number(metric['mean_absolute_percent_difference'])}% | "
            f"{display_number(metric['symmetric_mean_absolute_percent_difference'])}% | "
            f"{display_number(metric['maximum_absolute_difference'])} |"
        )
    lines.extend(
        [
            "",
            (
                "Bus-angle differences use the shortest circular displacement in "
                "degrees for absolute-error statistics."
            ),
            "",
            "## Convergence",
            "",
            "| Characteristic | GPU | CPU |",
            "|---|---:|---:|",
            (
                f"| Converged cases | {convergence['gpu']['converged']:,} "
                f"({display_number(convergence['gpu']['converged_percent'])}%) | "
                f"{convergence['cpu']['converged']:,} "
                f"({display_number(convergence['cpu']['converged_percent'])}%) |"
            ),
            (
                f"| Mean iterations | "
                f"{display_number(convergence['gpu']['iterations']['mean'])} | "
                f"{display_number(convergence['cpu']['iterations']['mean'])} |"
            ),
            (
                f"| Median iterations | "
                f"{display_number(convergence['gpu']['iterations']['median'])} | "
                f"{display_number(convergence['cpu']['iterations']['median'])} |"
            ),
            (
                f"| 95th-percentile iterations | "
                f"{display_number(convergence['gpu']['iterations']['p95'])} | "
                f"{display_number(convergence['cpu']['iterations']['p95'])} |"
            ),
            "",
            (
                "Convergence outcome agreement: "
                f"{display_number(convergence['convergence_agreement_percent'])}%"
            ),
            (
                "Convergence-column numerical differences are calculated over "
                f"the {convergence['numeric_metric_event_count']:,} events with "
                "`OK` status in both outputs; all-case metrics remain available "
                "in the JSON report."
            ),
            "",
            "| Convergence column | Unit | CPU mean | GPU mean | Mean absolute difference | MAPD |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for column, metric in convergence["numeric_metrics"].items():
        lines.append(
            f"| `{column}` | {metric['unit']} | "
            f"{display_number(metric['cpu_mean'])} | "
            f"{display_number(metric['gpu_mean'])} | "
            f"{display_number(metric['mean_absolute_difference'])} | "
            f"{display_number(metric['mean_absolute_percent_difference'])}% |"
        )
    lines.extend(
        [
            "",
            "## Performance",
            "",
            "| Characteristic | GPU | CPU | Comparison |",
            "|---|---:|---:|---:|",
            (
                f"| Maximum total application time | "
                f"{display_number(total_timer.get('gpu_maximum_seconds'))} s | "
                f"{display_number(total_timer.get('cpu_maximum_seconds'))} s | "
                f"{display_number(total_timer.get('maximum_speedup_cpu_over_gpu'))}x speedup |"
            ),
            (
                f"| Mean tasks per rank | "
                f"{display_number(performance['gpu']['tasks'].get('mean'))} | "
                f"{display_number(performance['cpu']['tasks'].get('mean'))} | — |"
            ),
            (
                f"| Task-count coefficient of variation | "
                f"{display_number(performance['gpu']['tasks'].get('coefficient_of_variation_percent'))}% | "
                f"{display_number(performance['cpu']['tasks'].get('coefficient_of_variation_percent'))}% | — |"
            ),
            "",
            "See `comparison_report.json`, `numerical_differences.csv`, and "
            "`timing_comparison.csv` for complete statistics. "
            "`flat_coverage_by_contingency.csv` identifies every event associated "
            "with unmatched flat-output rows, and "
            "`flat_coverage_by_branch.csv` identifies the branch keys.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GPU-accelerated, memory-bounded comparison of GriDSSPack and stock "
            "GridPACK contingency-analysis outputs"
        )
    )
    parser.add_argument("--gpu-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--cpu-dir", type=Path, default=DEFAULT_DATA_DIR / "cpu_results"
    )
    parser.add_argument("--prefix", default="Texas7k_v2")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        help="Parent directory for temporary event-bucketed Parquet datasets",
    )
    parser.add_argument("--block-size-mib", type=int, default=256)
    parser.add_argument("--bucket-events", type=int, default=64)
    parser.add_argument("--zero-threshold", type=float, default=1.0e-9)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-6)
    parser.add_argument(
        "--scheduler",
        choices=["distributed", "synchronous"],
        default="distributed",
        help="Dask scheduler used while staging large CSV files",
    )
    parser.add_argument(
        "--device-memory-limit",
        default="default",
        help="Dask-CUDA device memory limit, such as 80GB or default",
    )
    parser.add_argument(
        "--enable-cudf-spill",
        action="store_true",
        help=(
            "Enable cuDF spilling on discrete-memory GPUs; unsupported on "
            "unified-memory SoCs such as GB10"
        ),
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Retain event-bucketed Parquet files after the report is written",
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> dict[str, Path]:
    if args.block_size_mib <= 0:
        raise ValueError("--block-size-mib must be positive")
    if args.bucket_events <= 0:
        raise ValueError("--bucket-events must be positive")
    paths = {
        "gpu_flat": args.gpu_dir / f"{args.prefix}_flat.csv",
        "cpu_flat": args.cpu_dir / f"{args.prefix}_flat.csv",
        "gpu_convergence": args.gpu_dir / f"{args.prefix}_convergence.csv",
        "cpu_convergence": args.cpu_dir / f"{args.prefix}_convergence.csv",
        "gpu_buses": args.gpu_dir / f"{args.prefix}_buses.csv",
        "cpu_buses": args.cpu_dir / f"{args.prefix}_buses.csv",
        "gpu_log": args.gpu_dir / "output.log",
        "cpu_log": args.cpu_dir / "output.log",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))
    return paths


def main() -> int:
    args = parse_arguments()
    paths = validate_arguments(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_parent = args.temp_dir or args.output_dir
    temp_parent.mkdir(parents=True, exist_ok=True)
    intermediate = Path(
        tempfile.mkdtemp(prefix="griddss-compare-", dir=str(temp_parent))
    )
    started = time.monotonic()
    cluster = None
    client = None

    try:
        if args.scheduler == "distributed":
            from dask_cuda import LocalCUDACluster
            from distributed import Client

            cluster = LocalCUDACluster(
                CUDA_VISIBLE_DEVICES="0",
                n_workers=1,
                threads_per_worker=1,
                device_memory_limit=args.device_memory_limit,
                enable_cudf_spill=args.enable_cudf_spill,
                local_directory=str(intermediate / "dask-worker-space"),
                pre_import="cudf",
            )
            client = Client(cluster)
            print(f"Dask dashboard: {client.dashboard_link}")
        else:
            dask.config.set(scheduler="synchronous")

        gpu_parquet = intermediate / "gpu-flat"
        cpu_parquet = intermediate / "cpu-flat"
        block_size_bytes = args.block_size_mib * 1024 * 1024
        stage_flat_csv(
            paths["gpu_flat"], gpu_parquet, block_size_bytes, args.bucket_events
        )
        stage_flat_csv(
            paths["cpu_flat"], cpu_parquet, block_size_bytes, args.bucket_events
        )

        if client is not None:
            client.close()
            client = None
        if cluster is not None:
            cluster.close()
            cluster = None

        print("Comparing event-bucketed flat results with cuDF...")
        report = {
            "metadata": {
                "gpu_directory": str(args.gpu_dir),
                "cpu_directory": str(args.cpu_dir),
                "prefix": args.prefix,
                "block_size_mib": args.block_size_mib,
                "bucket_events": args.bucket_events,
                "zero_threshold": args.zero_threshold,
                "absolute_tolerance": args.absolute_tolerance,
                "relative_tolerance": args.relative_tolerance,
                "gpu_flat_file_bytes": paths["gpu_flat"].stat().st_size,
                "cpu_flat_file_bytes": paths["cpu_flat"].stat().st_size,
                "rapids": {
                    "cudf_version": cudf.__version__,
                    "dask_version": dask.__version__,
                },
            },
            "flat": compare_flat_results(
                gpu_parquet,
                cpu_parquet,
                args.zero_threshold,
                args.absolute_tolerance,
                args.relative_tolerance,
            ),
            "convergence": compare_convergence(
                paths["gpu_convergence"],
                paths["cpu_convergence"],
                args.zero_threshold,
                args.absolute_tolerance,
                args.relative_tolerance,
            ),
            "buses": compare_buses(
                paths["gpu_buses"],
                paths["cpu_buses"],
                args.zero_threshold,
                args.absolute_tolerance,
                args.relative_tolerance,
            ),
            "performance": compare_performance(
                paths["gpu_log"], paths["cpu_log"]
            ),
        }
        annotate_flat_coverage(
            report["flat"],
            paths["gpu_convergence"],
            paths["cpu_convergence"],
        )
        report["metadata"]["comparison_elapsed_seconds"] = time.monotonic() - started

        with (args.output_dir / "comparison_report.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        write_csv(
            args.output_dir / "numerical_differences.csv", metric_rows(report)
        )
        write_csv(
            args.output_dir / "timing_comparison.csv",
            report["performance"]["timer_comparison"],
        )
        write_csv(
            args.output_dir / "convergence_status_transitions.csv",
            report["convergence"]["status_transitions"],
        )
        write_csv(
            args.output_dir / "flat_coverage_by_contingency.csv",
            flat_coverage_rows(report),
        )
        write_csv(
            args.output_dir / "flat_coverage_by_branch.csv",
            flat_branch_coverage_rows(report),
        )
        (args.output_dir / "summary.md").write_text(
            build_markdown(report), encoding="utf-8"
        )
        print(f"Comparison complete: {args.output_dir / 'summary.md'}")
        return 0
    finally:
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()
        if args.keep_intermediate:
            print(f"Intermediate data retained at {intermediate}")
        else:
            shutil.rmtree(intermediate, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
