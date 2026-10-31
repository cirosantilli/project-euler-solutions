"""
Check generalized solver contracts and regressions from truncated reference searches.
"""

import importlib
import inspect
from itertools import permutations, product
from math import comb
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
        for problem in (66, 68, 74, 77, 78, 79, 85, 88, 90, 92, 93, 100):
            with self.subTest(problem=problem):
                reference, optimized = implementations(problem)
                self.assertEqual(list(inspect.signature(reference.naive).parameters),
                                 list(inspect.signature(optimized["solve"]).parameters))

    def test_die_faces_and_required_pairs(self):
        reference, optimized = implementations(90)
        for solve in (reference.naive, optimized["solve"]):
            self.assertEqual(solve(6, optimized["SQUARE_PAIRS"]), 1217)
            for faces in (-10, -1, *range(12), 10**100):
                designs = comb(10, faces) if 0 <= faces <= 10 else 0
                self.assertEqual(solve(faces, []), designs * (designs + 1) // 2)
            self.assertEqual(solve(0, [(0, 0)]), 0)
            self.assertEqual(solve(1, [(0, 0)]), 1)
            # Printed 6 and printed 9 are distinct designs: (6,6), (6,9), (9,9).
            self.assertEqual(solve(1, [(6, 9)]), 3)
            pairs = [(6, 9), (9, 6), (6, 9)]
            self.assertEqual(solve(1, pairs), 3)
            self.assertEqual(pairs, [(6, 9), (9, 6), (6, 9)])
            self.assertEqual(solve(10, [(x, y) for x in range(10) for y in range(10)]), 1)

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

    def test_magic_ring_size_and_decimal_length(self):
        reference, optimized = implementations(68)
        for size, digits, expected in ((-1, 16, ""), (2, 6, ""), (3, 8, ""),
                                       (3, 9, "432621513"), (4, 12, "462723831516"), (4, 13, "")):
            with self.subTest(size=size, digits=digits):
                self.assertEqual(reference.naive(size, digits), expected)
                self.assertEqual(optimized["solve"](size, digits), expected)
        self.assertEqual(optimized["solve"](), "6531031914842725")
        self.assertEqual(optimized["solve"](5, 17), "28797161103104548")

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

    def test_passcode_ordering_and_iterables(self):
        reference, optimized = implementations(79)
        for attempts, expected in (([], ""), (["789", "123"], "123789"),
                                   (["012", "234"], "01234"), (["123", "123"], "123"),
                                   (["١٢٣", "٣٤٥"], "١٢٣٤٥"), (["0²3"], "0²3")):
            for solve in (reference.naive, optimized["solve"]):
                self.assertEqual(solve(attempts), expected)
                self.assertEqual(solve(iter(attempts)), expected)

    def test_passcode_cycles_and_validation(self):
        reference, optimized = implementations(79)
        for attempts in (["121"], ["123", "341"]):
            for solve in (reference.naive, optimized["solve"]):
                with self.assertRaisesRegex(ValueError, "Constraints contain a cycle"):
                    solve(attempts)
        for invalid in ("", "12", "1234", "12x", "😀12"):
            for solve in (reference.naive, optimized["solve"]):
                with self.assertRaises(ValueError) as caught:
                    solve([invalid])
                self.assertEqual(caught.exception.args, (f"Invalid attempt: {invalid!r}",))

        def observations():
            yield "12"
            raise AssertionError("read past an invalid observation")

        for solve in (reference.naive, optimized["solve"]):
            with self.assertRaises(ValueError):
                solve(observations())

    def test_passcode_first_invalid_attempt_depends_on_input_order(self):
        reference, optimized = implementations(79)
        attempts = {"", "x"}
        for ordered in (("", "x"), ("x", ""), attempts):
            expected = (f"Invalid attempt: {next(iter(ordered))!r}",)
            for solve in (reference.naive, optimized["solve"]):
                with self.subTest(solver=solve.__name__, order=ordered):
                    with self.assertRaises(ValueError) as caught:
                        solve(ordered)
                    self.assertEqual(caught.exception.args, expected)
        self.assertEqual(attempts, {"", "x"})

    def test_passcode_matches_exhaustive_permutations(self):
        reference, optimized = implementations(79)
        triples = ["".join(chars) for chars in permutations("0123", 3)]
        cases = [["".join(chars)] for chars in product("0123", repeat=3)]
        cases.extend(list(pair) for pair in product(triples, repeat=2))
        for attempts in cases:
            digits = sorted(set("".join(attempts)))
            valid = []
            for chars in permutations(digits):
                candidate = "".join(chars)
                if all(candidate.index(a) < candidate.index(b) < candidate.index(c)
                       for a, b, c in attempts):
                    valid.append(candidate)
            expected = min(valid, default=None)
            for solve in (reference.naive, optimized["solve"]):
                with self.subTest(attempts=attempts, implementation=solve.__name__):
                    if expected is None:
                        with self.assertRaises(ValueError):
                            solve(attempts)
                    else:
                        self.assertEqual(solve(attempts), expected)

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
