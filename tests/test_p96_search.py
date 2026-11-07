"""Compare Sudoku propagation with exhaustive row-major backtracking."""

from pathlib import Path
import runpy
import unittest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = runpy.run_path(str(ROOT / "data/project-euler-statements/data/py/project_euler_statements/p96.py"))
OPTIMIZED = runpy.run_path(str(ROOT / "solvers/96.py"))
SOLUTION = [
    "483921657", "967345821", "251876493",
    "548132976", "729564138", "136798245",
    "372689514", "814253769", "695417382",
]


class SudokuSearchTests(unittest.TestCase):
    def test_every_single_missing_clue(self):
        for position in range(81):
            grid = [list(map(int, row)) for row in SOLUTION]
            grid[position // 9][position % 9] = 0
            before = [row[:] for row in grid]
            self.assertTrue(REFERENCE["complete"](grid))
            self.assertEqual(grid, [list(map(int, row)) for row in SOLUTION])
            text = "Grid 01\n" + "\n".join("".join(map(str, row)) for row in before)
            for solve in (REFERENCE["naive"], OPTIMIZED["solve"]):
                self.assertEqual(solve(text), 483)

    def test_empty_and_multiple_records(self):
        text = "\n ignored \n Grid 01 \n" + "\n\n".join(SOLUTION)
        for solve in (REFERENCE["naive"], OPTIMIZED["solve"]):
            self.assertEqual(solve(""), 0)
            self.assertEqual(solve("\n ignored \n"), 0)
            self.assertEqual(solve(text + "\n" + text), 966)

    def test_invalid_units_and_failed_branch_restoration(self):
        for first, second in ((0, 1), (0, 9), (0, 10)):
            grid = [[0] * 9 for _ in range(9)]
            grid[first // 9][first % 9] = 1
            grid[second // 9][second % 9] = 1
            self.assertFalse(REFERENCE["grid_ok"](grid))
            text = "Grid 01\n" + "\n".join("".join(map(str, row)) for row in grid)
            for solve in (REFERENCE["naive"], OPTIMIZED["solve"]):
                with self.assertRaises(ValueError):
                    solve(text)
        grid = [[0] * 9 for _ in range(9)]
        grid[0] = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        grid[1][0] = 9
        before = [row[:] for row in grid]
        self.assertTrue(REFERENCE["grid_ok"](grid))
        self.assertFalse(REFERENCE["complete"](grid))
        self.assertEqual(grid, before)

    def test_all_statement_puzzles(self):
        text = (ROOT / "data/project-euler-statements/data/documents/0096_sudoku.txt").read_text()
        self.assertEqual(REFERENCE["naive"](text), 24702)
        self.assertEqual(OPTIMIZED["solve"](text), 24702)
        for lines in OPTIMIZED["parse_puzzles"](text):
            grid = [list(map(int, row)) for row in lines]
            clues = [row[:] for row in grid]
            self.assertTrue(REFERENCE["complete"](grid))
            self.assertTrue(REFERENCE["grid_ok"](grid))
            self.assertTrue(all(0 not in row for row in grid))
            for r in range(9):
                for c in range(9):
                    self.assertTrue(clues[r][c] == 0 or clues[r][c] == grid[r][c])
            candidate_solution = OPTIMIZED["solve_cands"](OPTIMIZED["grid_to_cands"](lines))
            self.assertEqual([OPTIMIZED["bit_to_digit"](bits) for bits in candidate_solution],
                             [digit for row in grid for digit in row])


if __name__ == "__main__":
    unittest.main()
