import ProjectEulerSolutions.P93

open ProjectEulerSolutions.P93

def main : IO Unit := do
  IO.println ((solve 9).foldl (fun acc d => acc * 10 + d) 0)
