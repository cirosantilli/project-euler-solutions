import ProjectEulerSolutions.P86
namespace ProjectEulerSolutions.P86
open ProjectEulerSolutions.CuboidRoute

private theorem search_eq (target : Int) (c count : Nat) (correct : count = total c) :
    search target c count correct = scan target c := by
  rw [search, scan]
  split
  · rename_i active
    rw [dif_pos (by simpa [correct] using (by assumption : (count : Int) ≤ target))]
    exact search_eq target (c+1) _ _
  · rename_i done
    rw [dif_neg (by simpa [correct] using done)]
termination_by searchBound target - c
decreasing_by
  have := active_below_bound target c (by simpa [correct] using (by assumption : (count : Int) ≤ target))
  omega

theorem equiv (target : Int) : ProjectEulerStatements.P86.naive target = solve target :=
  (search_eq target 0 0 rfl).symm
end ProjectEulerSolutions.P86
