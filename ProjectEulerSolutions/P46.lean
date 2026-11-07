import ProjectEulerSolutions.Lib.PrimeTwiceSquareSuccessor
import ProjectEulerSolutions.Lib.PrimeTwiceSquareSearch
namespace ProjectEulerSolutions.P46
open PrimeTwiceSquare PrimeTwiceSquareSuccessor

/-- Check only square offsets; use the six-step trial primality test. -/
def acceptsFast (after : Int) (n : Nat) : Bool :=
  decide (after < (n : Int)) && oddComposite n && !certifiedFast n

theorem acceptsFast_correct (after : Int) (n : Nat) : acceptsFast after n = accepts after n := by
  simp [acceptsFast, accepts, counterexample, certifiedFast_correct, Bool.and_assoc]

/-- The witness is erased; execution scans odd candidates until a counterexample is found. -/
def search (after : Int) (success : Success after) (index : Nat)
    (before : index ≤ winnerIndex after success) : Nat :=
  if found : acceptsFast after (value index) then value index
  else search after success (index+1) (by
    have different : index ≠ winnerIndex after success := by
      intro same
      have hit := winner_accepted after success
      exact found (by simpa [acceptsFast_correct, answer, same] using hit)
    omega)
termination_by winnerIndex after success - index
decreasing_by
  have failed : ¬ acceptsFast after (value index) = true := by assumption
  have different : index ≠ winnerIndex after success := by
    intro same
    have hit := winner_accepted after success
    exact failed (by simpa [acceptsFast_correct, answer, same] using hit)
  omega

def solve (after : Int) (success : Success after) : Nat := search after success 0 (Nat.zero_le _)

theorem defaultSuccess : Success 0 := by
  refine ⟨2884, ?_⟩
  apply accepts_of
  · decide
  · exact PrimeTwiceSquareSearch.winner_accepted
end ProjectEulerSolutions.P46
