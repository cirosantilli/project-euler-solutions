"""
Check generalized solver contracts and regressions from truncated reference searches.
"""

import importlib
import inspect
from itertools import product
from pathlib import Path
import runpy
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/project-euler-statements/data/py"))


def implementations(problem):
    reference = importlib.import_module(f"project_euler_statements.p{problem}")
    optimized = runpy.run_path(str(ROOT / "solvers" / f"{problem}.py"))
    return reference, optimized


class UnboundedContractTests(unittest.TestCase):
    def test_shared_arguments(self):
        for problem in (66, 74, 77, 78, 85, 88, 92, 100):
            with self.subTest(problem=problem):
                reference, optimized = implementations(problem)
                self.assertEqual(list(inspect.signature(reference.naive).parameters),
                                 list(inspect.signature(optimized["solve"]).parameters))

    def test_pell_search_and_empty_range(self):
        reference, optimized = implementations(66)
        self.assertEqual(reference.minimal_x(13), 649)
        for bound, expected in ((-1, 0), (1, 0), (2, 2), (7, 5)):
            self.assertEqual(reference.naive(bound), expected)
            self.assertEqual(optimized["solve"](bound), expected)

    def test_triangle_paths_preserve_signed_input(self):
        reference, optimized = implementations(67)
        for height in range(4):
            size = height * (height + 1) // 2
            for weights in product((-1, 0, 1), repeat=size):
                triangle = []
                offset = 0
                for width in range(1, height + 1):
                    triangle.append(list(weights[offset:offset + width]))
                    offset += width
                before = [row[:] for row in triangle]
                expected = max((sum(triangle[level][sum(steps[:level])] for level in range(height))
                                for steps in product((0, 1), repeat=max(0, height - 1))), default=0)
                self.assertTrue(optimized["is_triangle"](triangle))
                self.assertEqual(optimized["solve"](triangle), expected)
                self.assertEqual(reference.naive(triangle), expected)
                self.assertEqual(triangle, before)
        self.assertFalse(optimized["is_triangle"]([[]]))
        self.assertFalse(optimized["is_triangle"]([[1], [2]]))

    def test_factorial_chain_longer_than_its_start(self):
        reference, optimized = implementations(74)
        self.assertEqual(reference.chain_len(3), 16)
        self.assertEqual(reference.chain_len(0), 2)
        self.assertEqual(reference.naive(4, 16), 1)
        self.assertEqual(optimized["solve"](4, 16), 1)
        self.assertEqual(reference.naive(2, 1), 1)
        self.assertEqual(optimized["solve"](2, 1), 1)

    def test_prime_partition_threshold(self):
        reference, optimized = implementations(77)
        for threshold, expected in ((-1, 2), (0, 2), (4, 10), (5000, 71)):
            self.assertEqual(reference.naive(threshold), expected)
            self.assertEqual(optimized["solve"](threshold), expected)

    def test_partition_divisor(self):
        reference, optimized = implementations(78)
        for modulus, expected in ((1, 1), (2, 2), (3, 3), (5, 4), (7, 5)):
            self.assertEqual(reference.naive(modulus), expected)
            self.assertEqual(optimized["solve"](modulus), expected)

    def test_rectangle_ties_and_large_dimensions(self):
        reference, optimized = implementations(85)
        for target, expected in ((-1, 1), (0, 1), (2, 1), (18, 6), (2_000_000, 2772)):
            self.assertEqual(reference.naive(target), expected)
            self.assertEqual(optimized["solve"](target), expected)
        target = (4000 * 4001 // 2) ** 2
        self.assertEqual(optimized["best_grid_area_near"](target), (4000, 4000, 16_000_000))

    def test_product_sum_minima(self):
        reference, optimized = implementations(88)
        for maximum, expected in ((-1, 0), (1, 0), (2, 4), (6, 30), (12, 61)):
            self.assertEqual(reference.naive(maximum), expected)
            self.assertEqual(optimized["solve"](maximum), expected)

    def test_square_chain_longer_than_its_start(self):
        reference, optimized = implementations(92)
        self.assertTrue(reference.ends_at_89(2))
        self.assertFalse(reference.ends_at_89(0))
        self.assertEqual(optimized["solve"](3), 1)
        for bound in (-1, 0, 1, 99, 100, 101, 1234):
            self.assertEqual(reference.naive(bound), optimized["solve"](bound))

    def test_disc_arrangements_need_two_draws(self):
        reference, optimized = implementations(100)
        for bound, expected in ((-1, 3), (0, 3), (3, 3), (4, 15), (21, 85)):
            self.assertEqual(reference.naive(bound), expected)
            self.assertEqual(optimized["solve"](bound), expected)


if __name__ == "__main__":
    unittest.main()
