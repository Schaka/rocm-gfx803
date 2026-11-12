"""Offline regression tests for reviewed gfx803 TopK patch and benchmarks."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
def imported(name):
    p=ROOT/"tools/correctness-suite"/(name+".py")
    s=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

bench=imported("topk_bench")
compare=imported("topk_bench_compare")

class TopKReviewTests(unittest.TestCase):
    def test_patch_uses_shared_arch_helper_and_marker(self):
        patch=(ROOT/"patches/pytorch/gfx803-topk-sort-fallback.patch").read_text()
        self.assertIn('getCUDAHooks().isGPUArch({"gfx803"})',patch)
        self.assertIn("GFX803_TOPK_SORT_FALLBACK_PATCH",patch)
        self.assertNotIn("std::strncmp",patch)

    def test_driver_and_dockerfile_share_marker_contract(self):
        driver=(ROOT/"patches/pytorch/apply-gfx803-topk-sort-fallback.sh").read_text()
        docker=(ROOT/"docker/pytorch.Dockerfile").read_text()
        self.assertIn("GFX803_TOPK_SORT_FALLBACK_PATCH",driver)
        self.assertIn("patch -p1",driver)
        self.assertIn("grep -q",driver)
        self.assertIn("apply-gfx803-topk-sort-fallback.sh /pytorch",docker)
        self.assertNotIn("git -C /pytorch apply",docker)

    def test_benchmark_covers_reviewed_shapes(self):
        self.assertEqual(len(bench.CASES),9)
        self.assertEqual(bench.CASES[0],((2,4),2,"float32"))
        self.assertIn(((32,151936),50,"float16"),bench.CASES)
        self.assertIn(((16000000,),100,"float32"),bench.CASES)

    def test_comparison_preserves_fault_and_detects_overhead(self):
        key=("2048x64","8","float32")
        before={key:{"status":"ok","median_ms":"1"}}
        after={key:{"status":"ok","median_ms":"1.15"}}
        result=compare.compare(before,after)
        self.assertTrue(result[0]["over_10pct"])
        self.assertAlmostEqual(result[0]["slowdown_pct"],15)
        after[key]={"status":"crash(signal 6)"}
        result=compare.compare(before,after)
        self.assertIsNone(result[0]["slowdown_pct"])

    def test_mismatched_cases_refused(self):
        with self.assertRaises(ValueError):
            compare.compare({("2x4","2","float32"):{}}, {})

    def test_compile(self):
        for name in ("topk_bench","topk_bench_compare"):
            p=ROOT/"tools/correctness-suite"/(name+".py")
            compile(p.read_text(),str(p),"exec")

if __name__=="__main__":unittest.main()
