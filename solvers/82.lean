import ProjectEulerSolutions.P82

open ProjectEulerSolutions.P82

def main : IO Unit := do
  let text ← IO.FS.readFile "0082_matrix.txt"
  IO.println (solve (parseMatrix text))
