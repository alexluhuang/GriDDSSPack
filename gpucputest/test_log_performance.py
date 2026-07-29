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

    def test_stock_logs_compare_only_total_application(self):
        result = self._compare(
            _timer("Total Application", 10.0, 12.0)
            + _timer("Contingency: Total Application", 1.0, 1.2),
            _timer("Total Application", 20.0, 24.0)
            + _timer("Contingency: Total Application", 8.0, 9.0),
        )
        rows = {row["timer"]: row for row in result["timer_comparison"]}
        self.assertTrue(rows["Total Application"]["comparable"])
        self.assertEqual(
            rows["Total Application"]["maximum_speedup_cpu_over_gpu"], 2.0
        )
        self.assertFalse(rows["Contingency: Total Application"]["comparable"])
        self.assertIsNone(
            rows["Contingency: Total Application"][
                "maximum_speedup_cpu_over_gpu"
            ]
        )

    def test_ca_v2_common_phases_are_comparable(self):
        schema = "[profiling] schema=ca-v2 common_phases=6\n"
        result = self._compare(
            schema + _timer("CA: Contingency Processing", 4.0, 5.0),
            schema + _timer("CA: Contingency Processing", 12.0, 15.0),
        )
        row = result["timer_comparison"][0]
        self.assertTrue(row["comparable"])
        self.assertEqual(row["comparison_scope"], "ca-v2 common phase")
        self.assertEqual(row["maximum_speedup_cpu_over_gpu"], 3.0)
        self.assertTrue(
            result["profiling_compatibility"]["common_phase_comparison_enabled"]
        )

    def test_gpu_detail_is_never_cross_path_speedup(self):
        schema = "[profiling] schema=ca-v2 common_phases=6\n"
        result = self._compare(
            schema + _timer("CA GPU: Batch Newton", 4.0, 5.0),
            schema + _timer("CA GPU: Batch Newton", 12.0, 15.0),
        )
        row = result["timer_comparison"][0]
        self.assertFalse(row["comparable"])
        self.assertEqual(row["comparison_scope"], "GPU diagnostic")
        self.assertIsNone(row["maximum_speedup_cpu_over_gpu"])

    def test_six_common_phases_account_for_total(self):
        schema = "[profiling] schema=ca-v2 common_phases=6\n"
        phases = "".join(
            _timer(name, value, value)
            for name, value in zip(
                COMPARE.CA_V2_COMMON_PHASES,
                (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            )
        )
        result = self._compare(
            schema + _timer("Total Application", 21.0, 21.0) + phases,
            schema + _timer("Total Application", 42.0, 42.0) + phases,
        )
        gpu_coverage = result["profiling_compatibility"][
            "gpu_phase_coverage"
        ]
        self.assertTrue(gpu_coverage["complete"])
        self.assertEqual(gpu_coverage["accounted_percent"], 100.0)
        self.assertEqual(gpu_coverage["unaccounted_average_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
