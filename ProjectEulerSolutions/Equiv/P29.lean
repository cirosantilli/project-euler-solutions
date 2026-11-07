import ProjectEulerSolutions.P29
namespace ProjectEulerSolutions.P29

theorem equiv (aMax bMax : Nat) : ProjectEulerStatements.P29.naive aMax bMax = solve aMax bMax := by
  simp [ProjectEulerStatements.P29.naive, ProjectEulerStatements.P29.powers, solve, List.flatMap_def]
end ProjectEulerSolutions.P29
