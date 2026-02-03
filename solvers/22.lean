import ProjectEulerSolutions.P22

open ProjectEulerSolutions.P22

def main : IO Unit := do
  let text ← IO.FS.readFile "0022_names.txt"
  IO.println (solve (parseNames text))
