import ProjectEulerSolutions.Lib.TriPentHexUnbounded
namespace ProjectEulerSolutions.P45
open TriPentHex TriPentHexUnbounded

/-- Every hexagonal number is triangular, so only pentagonality needs testing. -/
def search (after : Int) (index : Nat) (before : index ≤ winner after) : Nat :=
  if accepted : Eligible after index then hexagonal index
  else search after (index+1) (by
    have different : index ≠ winner after := by
      intro same
      exact accepted (same ▸ winner_spec after)
    omega)
termination_by winner after - index
decreasing_by
  have rejected : ¬ Eligible after index := by assumption
  have different : index ≠ winner after := by
    intro same
    exact rejected (same ▸ winner_spec after)
  omega

def solve (after : Int) : Nat := search after 1 (winner_spec after).1
end ProjectEulerSolutions.P45
