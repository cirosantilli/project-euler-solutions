import ProjectEulerSolutions.P78

open ProjectEulerSolutions.P78

def main : IO Unit := do
  match solve 1_000_000 with
  | some n => IO.println n
  | none => pure ()
