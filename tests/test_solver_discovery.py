"""Language proof sidecars are not standalone Project Euler executables."""

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("euler_runner", ROOT / "test.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


class SolverDiscoveryTests(unittest.TestCase):
    def test_language_proofs_are_excluded(self):
        for name in ("1.py.lean", "2.c.lean", "3.cpp.lean", "4_other.rs.lean"):
            with self.subTest(name=name):
                self.assertIsNone(runner.parse_solver_id(Path(name)))
                self.assertIsNone(runner.detect_language(Path(name)))

    def test_standalone_lean_solvers_remain_available(self):
        for name in ("1.lean", "1_other.lean"):
            with self.subTest(name=name):
                self.assertEqual(runner.parse_solver_id(Path(name)), 1)
                self.assertEqual(runner.detect_language(Path(name)), "lean")


if __name__ == "__main__":
    unittest.main()
