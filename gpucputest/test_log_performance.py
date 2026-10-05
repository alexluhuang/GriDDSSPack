#!/usr/bin/env python3

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


class _DaskConfig:
    @staticmethod
    def set(_values):
        return None


def _load_compare_results():
    sys.modules.setdefault("cudf", types.ModuleType("cudf"))
    dask = types.ModuleType("dask")
    dask.config = _DaskConfig()
    sys.modules["dask"] = dask
    sys.modules["dask.dataframe"] = types.ModuleType("dask.dataframe")
    path = Path(__file__).with_name("compare_results.py")
    spec = importlib.util.spec_from_file_location(
        "compare_results_for_timer_test", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMPARE = _load_compare_results()


def _timer(name, average, maximum):
    return (
        f"Timing statistics for: {name}\n"
        f"    Average time: {average:.4f}\n"
        f"    Maximum time: {maximum:.4f}\n"
        f"    Minimum time: {average:.4f}\n"
        "    RMS deviation: 0.1000\n"
    )


class PerformanceComparisonTest(unittest.TestCase):
    def _compare(self, gpu_text, cpu_text):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpu = root / "gpu.log"
            cpu = root / "cpu.log"
            gpu.write_text(gpu_text, encoding="utf-8")
            cpu.write_text(cpu_text, encoding="utf-8")
            return COMPARE.compare_performance(gpu, cpu)

    def test_legacy_v1_categories_are_comparable_with_unmarked_stock(self):
        mapped = (
            "[profiling] schema=legacy-v1 gpu_legacy_mapping=1\n"
        )
        result = self._compare(
            mapped
            + _timer("Total Application", 10.0, 12.0)
            + _timer("Contingency: Total Application", 1.0, 1.2),
            _timer("Total Application", 20.0, 24.0)
            + _timer("Contingency: Total Application", 8.0, 9.0),
        )
        rows = {row["timer"]: row for row in result["timer_comparison"]}
        self.assertTrue(rows["Total Application"]["comparable"])
        self.assertEqual(
            rows["Total Application"]["maximum_speedup_cpu_over_gpu"], 2.0
        )
        self.assertTrue(rows["Contingency: Total Application"]["comparable"])
        self.assertEqual(
            rows["Contingency: Total Application"]["comparison_scope"],
            "GridPACK legacy category",
        )
        self.assertEqual(
            rows["Contingency: Total Application"][
                "maximum_speedup_cpu_over_gpu"
            ],
            7.5,
        )
        compatibility = result["profiling_compatibility"]
        self.assertTrue(compatibility["legacy_category_comparison_enabled"])
        self.assertEqual(
            compatibility["shared_legacy_categories"],
            ["Contingency: Total Application"],
        )

    def test_old_ca_v2_legacy_rows_are_not_reported_as_speedups(self):
        old_schema = "[profiling] schema=ca-v2 common_phases=6\n"
        result = self._compare(
            old_schema + _timer("Powerflow: Map to Matrix", 1.0, 1.0),
            _timer("Powerflow: Map to Matrix", 2.0, 2.0),
        )
        row = result["timer_comparison"][0]
        self.assertFalse(row["comparable"])
        self.assertIsNone(row["maximum_speedup_cpu_over_gpu"])
        self.assertFalse(
            result["profiling_compatibility"][
                "legacy_category_comparison_enabled"
            ]
        )

    def test_every_pasted_gridpack_category_receives_a_speedup(self):
        marker = "[profiling] schema=legacy-v1 gpu_legacy_mapping=1\n"
        legacy_timers = "".join(
            _timer(name, 1.0, 2.0)
            for name in COMPARE.GRIDPACK_LEGACY_TIMERS
        )
        result = self._compare(marker + legacy_timers, legacy_timers)
        rows = {row["timer"]: row for row in result["timer_comparison"]}
        for name in COMPARE.GRIDPACK_LEGACY_TIMERS:
            with self.subTest(timer=name):
                self.assertTrue(rows[name]["comparable"])
                self.assertEqual(
                    rows[name]["comparison_scope"],
                    "GridPACK legacy category",
                )
                self.assertEqual(
                    rows[name]["maximum_speedup_cpu_over_gpu"], 1.0
                )

    def test_pasted_gridpack_category_set_is_explicit(self):
        self.assertEqual(
            COMPARE.GRIDPACK_LEGACY_TIMERS,
            {
                "Powerflow: Total Application",
                "Powerflow: Network Parser",
                "Powerflow: Partition",
                "Powerflow: Factory Load",
                "Powerflow: Factory Set Components",
                "Powerflow: Factory Set Exchange",
                "Powerflow: Bus Update",
                "Powerflow: Factory Operations",
                "Powerflow: Create Mappers",
                "Powerflow: Map to Matrix",
                "Powerflow: Map to Vector",
                "Vector Map: New Vector",
                "Vector Map: Load Bus Data",
                "loadBusData: Add Vector Elements",
                "loadBusData: Fill Buffer",
                "loadBusData: Add Elements",
                "Vector Map: Set Vector",
                "Powerflow: Create Linear Solver",
                "Powerflow: Solve Linear Equation",
                "Powerflow: Map to Bus",
                "mapToBus: get Data",
                "mapToBus: set Data",
                "Contingency: Total Application",
                "Contingency: Write Results",
            },
        )

    def test_batch_preparation_is_new_diagnostic(self):
        mapped = "[profiling] schema=legacy-v1 gpu_legacy_mapping=1\n"
        result = self._compare(
            mapped + _timer("Contingency: Batch Preparation", 1.0, 1.0),
            _timer("Contingency: Batch Preparation", 2.0, 2.0),
        )
        row = result["timer_comparison"][0]
        self.assertFalse(row["comparable"])
        self.assertEqual(row["comparison_scope"], "new diagnostic")
        self.assertIsNone(row["maximum_speedup_cpu_over_gpu"])

    def test_new_category_without_stock_counterpart_stays_diagnostic(self):
        result = self._compare(
            _timer("Powerflow: GPU Wave Screening", 1.0, 1.0),
            _timer("Powerflow: GPU Wave Screening", 2.0, 2.0),
        )
        row = result["timer_comparison"][0]
        self.assertFalse(row["comparable"])
        self.assertEqual(row["comparison_scope"], "unclassified diagnostic")
        self.assertIsNone(row["maximum_speedup_cpu_over_gpu"])

    def test_old_ca_gpu_detail_is_never_cross_path_speedup(self):
        schema = "[profiling] schema=ca-v2 common_phases=6\n"
        result = self._compare(
            schema + _timer("CA GPU: Batch Newton", 4.0, 5.0),
            schema + _timer("CA GPU: Batch Newton", 12.0, 15.0),
        )
        row = result["timer_comparison"][0]
        self.assertFalse(row["comparable"])
        self.assertEqual(row["comparison_scope"], "GPU diagnostic")
        self.assertIsNone(row["maximum_speedup_cpu_over_gpu"])

    def test_marker_requires_legacy_v1_schema_and_enabled_mapping(self):
        result = self._compare(
            "[profiling] schema=other gpu_legacy_mapping=1\n"
            + _timer("Powerflow: Map to Matrix", 1.0, 1.0),
            _timer("Powerflow: Map to Matrix", 2.0, 2.0),
        )
        row = result["timer_comparison"][0]
        self.assertFalse(row["comparable"])
        self.assertIsNone(row["maximum_speedup_cpu_over_gpu"])


if __name__ == "__main__":
    unittest.main()
