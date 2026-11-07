import ProjectEulerSolutions.P52

open ProjectEulerSolutions.P52

def main : IO Unit := do
  match solve 6 with
  | some value => IO.println value
  | none => throw (IO.userError "No permuted multiple exists")
