import ProjectEulerSolutions.P81

open ProjectEulerSolutions.P81

def main : IO Unit := do
  let text ← IO.FS.readFile "0081_matrix.txt"
  IO.println (solve (parseMatrix text))
