"""
Check the source clique search against exhaustive searches on supplied graphs.
"""

import ast
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import random
import runpy
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "solvers" / "60.py"


def graph_solver(vertices, edges, source=SOURCE):
    """
    Supply a graph while retaining the source's complete search control flow.
    """
    module = ast.parse(source.read_text())
    solve = deepcopy(next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "solve"))
    pair = next(node for node in solve.body if isinstance(node, ast.FunctionDef) and node.name == "pair_ok")
    pair.body = ast.parse("return graph_relation(p, q)").body
    power = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "pow10_for")
    isolated = ast.fix_missing_locations(ast.Module(body=[power, solve], type_ignores=[]))
    scope = {
        "primes_up_to": lambda limit: [p for p in vertices if p <= limit],
        "is_prime": lambda n: True,
        "graph_relation": lambda p, q: tuple(sorted((p, q))) in edges,
    }
    exec(compile(isolated, str(source), "exec"), scope)
    return scope["solve"]


def clique_edges(clique):
    return set(combinations(clique, 2))


class PrimePairSearchTests(unittest.TestCase):
    def test_large_later_vertex_can_improve_incumbent(self):
        first = [3, 31, 37, 41, 43]
        better = [7, 11, 13, 17, 101]
        vertices = sorted(first + better)
        solve = graph_solver(vertices, clique_edges(first) | clique_edges(better))
        self.assertEqual(solve(101), (149, better))

    def test_equal_sum_retains_first_clique(self):
        first = [3, 11, 17, 23, 47]
        later = [7, 13, 19, 29, 33]
        vertices = sorted(first + later)
        solve = graph_solver(vertices, clique_edges(first) | clique_edges(later))
        self.assertEqual(solve(47), (101, first))

    def test_small_graphs_match_exhaustive_search(self):
        rng = random.Random(60)
        for case in range(60):
            vertices = sorted(rng.sample([3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 101], rng.randrange(4, 10)))
            edges = {pair for pair in combinations(vertices, 2) if rng.randrange(4) != 0}
            expected = (10**30, [])
            for clique in combinations(vertices, 5):
                if all(pair in edges for pair in combinations(clique, 2)) and sum(clique) < expected[0]:
                    expected = (sum(clique), list(clique))
            with self.subTest(case=case):
                self.assertEqual(graph_solver(vertices, edges)(max(vertices)), expected)

    def test_default_prime_result(self):
        solve = runpy.run_path(str(SOURCE))["solve"]
        self.assertEqual(solve(), (26033, [13, 5197, 5701, 6733, 8389]))


if __name__ == "__main__":
    unittest.main()
