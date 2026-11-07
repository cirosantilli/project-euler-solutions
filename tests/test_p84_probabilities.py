"""
Check the uniform-card model and exact stationary equations against an independent enumeration.
"""

from pathlib import Path
import runpy
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data/project-euler-statements/data/py'))
from project_euler_statements import p84

SOLVER = runpy.run_path(str(ROOT / 'solvers/84.py'))


class MonopolyProbabilityTests(unittest.TestCase):
    def test_square_effects(self):
        landing = SOLVER['resolve_landing']
        for square in range(40):
            self.assertEqual(sum(landing(square).values()), 256)
        self.assertEqual(landing(30), {(10, True): 256})
        self.assertEqual(landing(10), {(10, False): 256})
        self.assertEqual(landing(2), {(2, False): 224, (0, False): 16, (10, True): 16})
        # Back three at CH3 triggers CC3: stay there, advance to GO, or go to jail.
        self.assertEqual(landing(36)[(33, False)], 14)
        self.assertEqual(landing(36)[(0, False)], 17)
        self.assertEqual(landing(36)[(10, True)], 17)

    def test_doubles_and_jail(self):
        self.assertEqual(p84.advance(0, 2, 1, 1, 0, 0), (10, 0))
        self.assertEqual(p84.advance(6, 1, 2, 2, 0, 0), (10, 2))
        self.assertEqual(p84.advance(26, 1, 2, 2, 0, 0), (10, 0))
        self.assertEqual(p84.advance(34, 1, 1, 1, 1, 9), (10, 0))
        self.assertEqual(p84.advance(0, 2, 1, 2, 0, 0), (3, 0))

    def test_complete_transitions_and_stationary_distributions(self):
        for sides, expected in ((1, '101214'), (2, '101316'), (3, None), (4, '101524'),
                                (5, None), (6, '102400'), (7, None), (8, None)):
            with self.subTest(sides=sides):
                reference = p84.transition_counts(sides)
                optimized = SOLVER['build_transition'](sides)
                for row, sparse in zip(reference, optimized):
                    self.assertEqual({j: count for j, count in enumerate(row) if count}, sparse)
                    self.assertEqual(sum(row), 256 * sides * sides)
                weights = SOLVER['stationary_weights'](optimized, 256 * sides * sides)
                reference_weights = p84.stationary_weights(reference, 256 * sides * sides)
                self.assertGreater(sum(weights), 0)
                self.assertGreaterEqual(min(weights), 0)
                for target, weight in enumerate(weights):
                    self.assertEqual(sum(weights[source] * row.get(target, 0)
                                         for source, row in enumerate(optimized)),
                                     256 * sides * sides * weight)
                    self.assertEqual(weight * sum(reference_weights), reference_weights[target] * sum(weights))
                self.assertEqual(SOLVER['solve'](sides), p84.naive(sides))
                if expected is not None:
                    self.assertEqual(SOLVER['solve'](sides), expected)

    def test_exact_elimination(self):
        # Include periodicity, a transient state, row pivoting, and arbitrarily slow mixing.
        for rows, denominator, expected in (
                ([{1: 1}, {2: 1}, {0: 1}], 1, [1, 1, 1]),
                ([{1: 1}, {1: 1}], 1, [0, 1]),
                ([{0: 1}, {0: 1}], 1, [1, 0]),
                ([{0: 10**40 - 1, 1: 1}, {0: 2, 1: 10**40 - 2}], 10**40, [2, 1]),
                ([{0: 7}], 7, [1])):
            dense = [[row.get(j, 0) for j in range(len(rows))] for row in rows]
            for weights in (SOLVER['stationary_weights'](rows, denominator),
                            p84.stationary_weights(dense, denominator)):
                self.assertGreater(sum(weights), 0)
                self.assertEqual([weight * sum(expected) for weight in weights],
                                 [weight * sum(weights) for weight in expected])
        with self.assertRaises(ArithmeticError):
            SOLVER['stationary_weights']([{0: 1}, {1: 1}], 1)
        with self.assertRaises(ArithmeticError):
            p84.stationary_weights([[1, 0], [0, 1]], 1)
        for solve in (SOLVER['solve'], p84.naive):
            for sides in (-1, 0):
                with self.assertRaises(ValueError):
                    solve(sides)


if __name__ == '__main__':
    unittest.main()
