import ProjectEulerSolutions.P51

open ProjectEulerSolutions.P51

def main : IO Unit := do
  IO.println (solve 8 (ProjectEulerStatements.P51.exists_prime_family 8 (by decide)))
