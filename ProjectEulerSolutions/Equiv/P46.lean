import ProjectEulerSolutions.P46
namespace ProjectEulerSolutions.P46
open PrimeTwiceSquareSuccessor

theorem search_correct (after : Int) (success : Success after) (index : Nat)
    (before : index ≤ winnerIndex after success) : search after success index before = answer after success := by
  rw [search]
  split
  · rename_i found
    have least := winner_minimal after success index (by simpa [acceptsFast_correct] using found)
    have same : index = winnerIndex after success := by omega
    simp [same, answer]
  · exact search_correct after success (index+1) _
termination_by winnerIndex after success - index
decreasing_by
  have failed : ¬ acceptsFast after (value index) = true := by assumption
  have different : index ≠ winnerIndex after success := by
    intro same
    have hit := winner_accepted after success
    exact failed (by simpa [acceptsFast_correct, answer, same] using hit)
  omega

/-- Least counterexample above the threshold, on precisely the terminating domain. -/
theorem equiv (after : Int) (success : Success after) : answer after success = solve after success :=
  (search_correct after success 0 _).symm
end ProjectEulerSolutions.P46
