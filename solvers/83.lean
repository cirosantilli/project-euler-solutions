import ProjectEulerSolutions.P83

open ProjectEulerSolutions.P83

def main : IO Unit := do
  let text ← IO.FS.readFile "0083_matrix.txt"
  IO.println (solve (parseMatrix text))
