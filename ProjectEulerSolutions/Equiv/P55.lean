import ProjectEulerSolutions.P55

namespace ProjectEulerSolutions.P55

private theorem iterate_eq (k n : Nat) :
    ProjectEulerStatements.P55.iterate ProjectEulerStatements.P55.reverseAdd k n =
      Lychrel.iterate k n := by
  induction k generalizing n with
  | zero => rfl
  | succ k ih => exact ih _

private theorem candidate_eq (n fuel : Nat) :
    ProjectEulerStatements.P55.isLychrel n fuel = Lychrel.candidate n fuel := by
  rw [Lychrel.candidate_eq_not_any]
  simp only [ProjectEulerStatements.P55.isLychrel,
    ProjectEulerStatements.P55.reachesPalindrome, ProjectEulerStatements.P55.palAt,
    iterate_eq]
  rfl

theorem equiv (limit maxIters : Nat) :
    ProjectEulerStatements.P55.naive limit maxIters = solve limit maxIters := by
  simp only [ProjectEulerStatements.P55.naive, solve, Lychrel.count, candidate_eq]

end ProjectEulerSolutions.P55
