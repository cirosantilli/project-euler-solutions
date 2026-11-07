import ProjectEulerSolutions.P26
namespace ProjectEulerSolutions.P26

theorem equiv (n : Nat) : ProjectEulerStatements.P26.naive n = solve n := by
  unfold ProjectEulerStatements.P26.naive solve
  cases h : List.range' 1 (n - 1) with
  | nil => rfl
  | cons first rest =>
      rw [ProjectEulerStatements.ArgMaximum.argmax_cons]
      rfl
end ProjectEulerSolutions.P26
