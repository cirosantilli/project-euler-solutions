import ProjectEulerStatements.P52

namespace ProjectEulerSolutions.P52

/-- Inspect each candidate with ten-counter digit signatures. -/
def step (limit current : Nat) : Sum Nat Nat :=
  if PermutedMultiples.accepts limit current then .inl current else .inr (current + 1)

def solve (limit : Nat) : Option Nat :=
  ProjectEulerStatements.UnboundedSearch.run (step limit) 1

end ProjectEulerSolutions.P52
