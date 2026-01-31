import ProjectEulerSolutions.P24

open ProjectEulerSolutions.P24

def main : IO Unit := do
  IO.println (serialize (solve (List.range 10) 1000000))
