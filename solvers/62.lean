import ProjectEulerSolutions.P62

open ProjectEulerSolutions.P62

def main : IO Unit := do
  match solve 5 with
  | some value => IO.println value
  | none => throw (IO.userError "No cube class has the requested size")
