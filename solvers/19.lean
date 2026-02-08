import ProjectEulerSolutions.P19

open ProjectEulerSolutions.P19

def startDate : ProjectEulerStatements.P19.Date :=
  ProjectEulerStatements.P19.mkFirstOfMonth 1901 1 (by decide) (by decide)

def endDate : ProjectEulerStatements.P19.Date :=
  ProjectEulerStatements.P19.mkFirstOfMonth 2000 12 (by decide) (by decide)

def main : IO Unit := do
  IO.println (solve startDate endDate)
