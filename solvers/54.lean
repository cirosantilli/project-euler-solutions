import ProjectEulerSolutions.P54

open ProjectEulerSolutions.P54

def main : IO Unit := do
  let text ← IO.FS.readFile "0054_poker.txt"
  let lines := parseLines text
  IO.println (solveLines lines)
