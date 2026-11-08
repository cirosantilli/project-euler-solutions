"""Exact power ordering, including ties and inputs that floating logs merge."""

from fractions import Fraction
from pathlib import Path
import random
import runpy
import unittest


ROOT = Path(__file__).resolve().parents[1]
OPTIMIZED = runpy.run_path(str(ROOT / "solvers/99.py"))
REFERENCE = runpy.run_path(str(ROOT / "data/project-euler-statements/data/py/project_euler_statements/p99.py"))


class ExactPowerTests(unittest.TestCase):
    def test_power_enclosures_and_exact_precision(self):
        for base in range(1, 20):
            for exponent in range(101):
                value = base ** exponent
                for precision in (1, 2, 4, 16):
                    lower, upper, shift = OPTIMIZED["power_bounds"](base, exponent, precision)
                    self.assertLessEqual(lower << shift, value)
                    self.assertLessEqual(value, upper << shift)
                    self.assertLessEqual(upper.bit_length(), precision + 1)
                lower, upper, shift = OPTIMIZED["power_bounds"](
                    base, exponent, max(1, base.bit_length() * exponent))
                self.assertEqual((lower, upper, shift), (value, value, 0))

    def test_comparisons_against_rational_arithmetic(self):
        rng = random.Random(99)
        for _ in range(5000):
            a, b = rng.randrange(1, 100), rng.randrange(1, 100)
            m, n = rng.randrange(-200, 201), rng.randrange(-200, 201)
            x, y = Fraction(a) ** m, Fraction(b) ** n
            self.assertEqual(OPTIMIZED["compare_powers"](a, m, b, n), (x > y) - (x < y))
            self.assertEqual(REFERENCE["greater_power"]((a, m), (b, n)), x > y)
        for _ in range(200):
            pairs = [(rng.randrange(1, 20), rng.randrange(-20, 21)) for _ in range(rng.randrange(20))]
            expected = max(range(len(pairs)), key=lambda i: Fraction(pairs[i][0]) ** pairs[i][1]) + 1 if pairs else 0
            self.assertEqual(OPTIMIZED["solve"](pairs), expected)
            self.assertEqual(REFERENCE["naive"](pairs), expected)

    def test_close_powers_ties_and_large_exponents(self):
        for solve in (OPTIMIZED["solve"], REFERENCE["naive"]):
            self.assertEqual(solve([]), 0)
            self.assertEqual(solve([(2**53, 1), (2**53 + 1, 1)]), 2)
            self.assertEqual(solve([(2, 6), (4, 3), (8, 2)]), 1)
            self.assertEqual(solve([(2, -6), (4, -3), (8, -2)]), 1)
            self.assertEqual(solve([(1, -1000), (2, 0), (9, -1)]), 1)
            self.assertEqual(solve([(10**308, -10**308)]), 1)
            for invalid in ([(0, 1)], [(2, 1), (-1, 2)]):
                with self.assertRaises(ValueError):
                    solve(invalid)
        huge = 10**100
        self.assertEqual(OPTIMIZED["compare_powers"](2, huge, 4, huge // 2), 0)
        self.assertEqual(OPTIMIZED["compare_powers"](2, huge + 1, 4, huge // 2), 1)
        self.assertEqual(OPTIMIZED["compare_powers"](2, -huge - 1, 4, -huge // 2), -1)
        self.assertEqual(OPTIMIZED["compare_scaled"](2, huge, 1, huge + 1), 0)
        self.assertEqual(OPTIMIZED["compare_scaled"](0, huge, 1, 0), -1)

    def test_statement_dataset(self):
        with (ROOT / "data/project-euler-statements/data/documents/0099_base_exp.txt").open() as source:
            pairs = OPTIMIZED["parse_pairs"](source)
        self.assertEqual(OPTIMIZED["solve"](pairs[:2]), 2)
        self.assertEqual(OPTIMIZED["solve"](pairs), 709)


if __name__ == "__main__":
    unittest.main()
