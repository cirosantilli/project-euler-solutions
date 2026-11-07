import ProjectEulerStatements.Lib.FirstWitness
import ProjectEulerSolutions.Lib.PrimeReplacementSearch

namespace ProjectEulerSolutions.P51

/-- The Python solver's equal-digit groups, bit masks, and large-family pruning. -/
def accepts (target n : Nat) : Bool :=
  decide (Nat.Prime n) && (List.range 10).any (PrimeReplacementSearch.query target n)

theorem accepts_eq (target n : Nat) :
    accepts target n = ProjectEulerStatements.P51.hasPrimeFamily target n := by
  apply Bool.eq_iff_iff.mpr
  simp only [accepts, Bool.and_eq_true, decide_eq_true_eq,
    PrimeReplacementSearch.hasPrimeFamily_iff_query]

/-- The existence premise is erased at runtime and expresses the unbounded search's domain. -/
def solve (target : Nat) (success : ∃ n, ProjectEulerStatements.P51.hasPrimeFamily target n = true) : Nat :=
  ProjectEulerStatements.FirstWitness.first (accepts target) (show ∃ n, accepts target n = true from by simpa only [accepts_eq] using success)

end ProjectEulerSolutions.P51
