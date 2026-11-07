"""
Check anagram-square searches against exhaustive injective digit assignments.
"""

import importlib
from itertools import combinations, permutations, product
import json
from math import isqrt
from pathlib import Path
import runpy
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/project-euler-statements/data/py"))
REFERENCE = importlib.import_module("project_euler_statements.p98")
SOLVE = runpy.run_path(str(ROOT / "solvers/98.py"))["solve"]


def assignment_oracle(first, second):
    letters = sorted(set(first))
    best = 0
    for digits in permutations("0123456789", len(letters)):
        mapping = dict(zip(letters, digits))
        if mapping[first[0]] == "0" or mapping[second[0]] == "0":
            continue
        a = int("".join(mapping[c] for c in first))
        b = int("".join(mapping[c] for c in second))
        if isqrt(a) ** 2 == a and isqrt(b) ** 2 == b:
            best = max(best, a, b)
    return best


class AnagramSquareTests(unittest.TestCase):
    def test_small_anagrams_with_assignment_oracle(self):
        for width in range(1, 5):
            groups = {}
            for chars in product("ABC", repeat=width):
                word = "".join(chars)
                groups.setdefault("".join(sorted(word)), []).append(word)
            for group in groups.values():
                for first, second in combinations(group, 2):
                    with self.subTest(first=first, second=second):
                        expected = assignment_oracle(first, second)
                        self.assertEqual(REFERENCE.naive([first, second]), expected)
                        self.assertEqual(SOLVE([first, second]), expected)

    def test_empty_words_duplicates_and_example(self):
        for words, expected in (([], 0), ([""], 0), (["", ""], 0),
                                (["CARE"], 0), (["CARE", "CARE"], 0),
                                (["CARE", "RACE"], 9216), (["", "CARE", "RACE", "CARE"], 9216)):
            with self.subTest(words=words):
                self.assertEqual(REFERENCE.naive(words), expected)
                self.assertEqual(SOLVE(words), expected)

    def test_supplied_words(self):
        words = json.loads("[" + (ROOT / "data/project-euler-statements/data/documents/0098_words.txt").read_text() + "]")
        self.assertEqual(REFERENCE.naive(words), 18769)
        self.assertEqual(SOLVE(words), 18769)


if __name__ == "__main__":
    unittest.main()
