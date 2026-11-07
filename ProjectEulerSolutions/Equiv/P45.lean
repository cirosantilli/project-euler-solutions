import ProjectEulerSolutions.P45
namespace ProjectEulerSolutions.P45
open TriPentHex TriPentHexUnbounded

theorem search_correct (after : Int) (index : Nat) (before : index ≤ winner after) :
    search after index before = answer after := by
  rw [search]
  split
  · rename_i accepted
    have least := winner_minimal after index accepted
    have same : index = winner after := by omega
    simp [same, answer]
  · exact search_correct after (index+1) _
termination_by winner after - index
decreasing_by
  have rejected : ¬ Eligible after index := by assumption
  have different : index ≠ winner after := by
    intro same
    exact rejected (same ▸ winner_spec after)
  omega

/-- The least common polygonal value above the threshold, matching Python's unbounded reference. -/
theorem equiv (after : Int) : answer after = solve after :=
  (search_correct after 1 _).symm
end ProjectEulerSolutions.P45
