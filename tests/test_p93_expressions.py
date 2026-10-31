"""
Check rational arithmetic, all five four-leaf tree shapes, and operand selection.
"""

from fractions import Fraction
from itertools import combinations, permutations, product
from pathlib import Path
import runpy
import unittest


ROOT = Path(__file__).resolve().parents[1]
OPTIMIZED = runpy.run_path(str(ROOT / "solvers/93.py"))
REFERENCE = runpy.run_path(str(ROOT / "data/project-euler-statements/data/py/project_euler_statements/p93.py"))


def apply(operation, left, right):
    if left is None or right is None:
        return None
    if operation == "+":
        return left + right
    if operation == "-":
        return left - right
    if operation == "*":
        return left * right
    return left / right if right else None


def fraction_oracle(digits):
    results = set()
    for a, b, c, d in set(permutations(map(Fraction, digits))):
        for x, y, z in product("+-*/", repeat=3):
            results.update((
                apply(z, apply(y, apply(x, a, b), c), d),
                apply(z, apply(x, a, apply(y, b, c)), d),
                apply(x, a, apply(z, apply(y, b, c), d)),
                apply(x, a, apply(y, b, apply(z, c, d))),
                apply(y, apply(x, a, b), apply(z, c, d)),
            ))
    results.discard(None)
    return results


def reference_values(digits):
    return {Fraction(numerator, denominator)
            for order in permutations(digits)
            for numerator, denominator in REFERENCE["all_results"](order)}


class ArithmeticExpressionTests(unittest.TestCase):
    def test_canonical_rationals(self):
        for numerator, denominator in product(range(-8, 9), repeat=2):
            if denominator:
                value = Fraction(numerator, denominator)
                self.assertEqual(OPTIMIZED["rational"](numerator, denominator),
                                 (value.numerator, value.denominator))

    def test_all_tree_shapes_with_fraction_oracle(self):
        for digits in ((0, 0, 0, 0), (0, 1, 2, 3), (1, 2, 3, 4), (1, 3, 4, 6),
                       (1, 2, 5, 8), (1, 1, 3, 3), (-3, -1, 2, 5), (2, 3, 11, 13)):
            with self.subTest(digits=digits):
                expected = fraction_oracle(digits)
                self.assertEqual(reference_values(digits), expected)
                self.assertEqual({Fraction(*value) for value in OPTIMIZED["all_results"](digits)}, expected)
        self.assertIn(Fraction(24), reference_values((1, 3, 4, 6)))

    def test_all_decimal_operand_sets(self):
        for digits in combinations(range(10), 4):
            with self.subTest(digits=digits):
                length, integers = OPTIMIZED["consecutive_length"](digits)
                expected = reference_values(digits)
                self.assertEqual(integers, {int(value) for value in expected if value > 0 and value.denominator == 1})
                self.assertEqual(length, REFERENCE["consecutive_for_digits"](digits))

    def test_shared_bounds_and_default(self):
        for bound in (-100, -1, 0, 1, 2, 3, 4, 5, 9, 10, 11):
            with self.subTest(bound=bound):
                self.assertEqual(OPTIMIZED["solve"](bound), REFERENCE["naive"](bound))
        self.assertEqual(OPTIMIZED["solve"](), (1, 2, 5, 8))
        self.assertEqual(OPTIMIZED["solve"](2), ())
        self.assertEqual(OPTIMIZED["solve"](3), (0, 1, 2, 3))
        self.assertEqual(OPTIMIZED["consecutive_length"]((1, 2, 5, 8))[0], 51)

    def test_empty_and_repeated_operands(self):
        for digits in ((), (0,), (7,), (-2,), (1, 1), (0, 0, 2), (2, 2, 2)):
            with self.subTest(digits=digits):
                self.assertEqual({Fraction(*value) for value in OPTIMIZED["all_results"](digits)},
                                 reference_values(digits))


if __name__ == "__main__":
    unittest.main()
