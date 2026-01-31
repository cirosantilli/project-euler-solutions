import ProjectEulerSolutions.P54

open ProjectEulerSolutions.P54

def main : IO Unit := do
  let text ← IO.FS.readFile "0054_poker.txt"
  IO.println (solve (parseLines text))
