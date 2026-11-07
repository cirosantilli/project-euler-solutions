import ProjectEulerSolutions.P57

namespace ProjectEulerSolutions.P57

theorem equiv (limit : Nat) : ProjectEulerStatements.P57.naive limit = solve limit :=
  (SqrtConvergents.loop_initial_eq_naive limit).symm

end ProjectEulerSolutions.P57
