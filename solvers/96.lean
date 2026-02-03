import ProjectEulerSolutions.P96

open ProjectEulerSolutions.P96

def main : IO Unit := do
  let text ← IO.FS.readFile "0096_sudoku.txt"
  let puzzles := parsePuzzles text
  IO.println (solve puzzles)
