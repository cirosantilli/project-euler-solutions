import ProjectEulerSolutions.P37
namespace ProjectEulerSolutions.P37
open ProjectEulerStatements.TruncatablePrimeBase

private theorem total_step (base depth : Nat) :
    (accepted base (depth+1)).sum = (accepted base depth).sum +
      ((level base (depth+1)).filter (leftCheck base)).sum := by
  simp [accepted, candidates, List.range_succ, List.flatMap_append, List.filter_append, List.sum_append]

theorem search_correct (base : Nat) (success : Exhausts base) (depth : Nat)
    (frontier : List Nat) (total : Nat) (frontierCorrect : frontier = level base depth)
    (before : depth ≤ Nat.find success) (totalCorrect : total = (accepted base depth).sum) :
    search base success depth frontier total frontierCorrect before = answer base success := by
  rw [search]
  split
  · rename_i empty
    have least : Nat.find success ≤ depth := Nat.find_min' success (frontierCorrect ▸ empty)
    have same : depth = Nat.find success := by omega
    simpa [answer, same] using totalCorrect
  · apply search_correct
    rw [total_step, ← totalCorrect]
    simp [frontierCorrect, level]
termination_by Nat.find success - depth
decreasing_by
  have nonempty : frontier ≠ [] := by assumption
  have different : depth ≠ Nat.find success := by
    intro same
    exact nonempty (by rw [frontierCorrect, same]; exact Nat.find_spec success)
  omega

/-- The common base-dependent reference is the sum of every truncatable prime. -/
theorem equiv (base : Nat) (success : Exhausts base) : answer base success = solve base success := by
  apply (search_correct base success 0 (roots base) 0 rfl (Nat.zero_le _) ?_).symm
  simp [accepted, candidates]
end ProjectEulerSolutions.P37
