import ProjectEulerSolutions.P52

namespace ProjectEulerSolutions.P52

/-- Counter equality and sorted-digit equality give the same search, including divergence. -/
theorem equiv (limit : Nat) : ProjectEulerStatements.P52.naive limit = solve limit := by
  have steps : ProjectEulerStatements.P52.step limit = step limit := by
    funext current
    simp only [ProjectEulerStatements.P52.step, step, ProjectEulerStatements.P52.accepts_eq]
  simp only [ProjectEulerStatements.P52.naive, solve, steps]

end ProjectEulerSolutions.P52
