import ProjectEulerSolutions.Lib.CliqueSearch
import Mathlib.Data.List.Sort

namespace ProjectEulerSolutions.CliqueSearch

/-- If a stored row is an ordered sublist and every neighbor passes the
cutoff, candidate collection returns that row without any further filtering. -/
theorem candidates_eq_row (relation : Nat → Nat → Bool) (need : Nat) (sum : Int) (p : Nat)
    (remaining neighbors : List Nat) (bound : Int)
    (ordered : remaining.Pairwise (· < ·)) (sublist : neighbors.Sublist remaining)
    (within : ∀ q ∈ neighbors, sum + (p : Int) + (q : Int) < bound)
    (row : relation p = neighbors.contains) :
    candidates relation need sum p remaining bound = neighbors := by
  apply (ordered.sublist (candidates_sublist relation need sum p remaining bound)).eq_of_mem_iff
    (ordered.sublist sublist)
  intro q
  rw [candidates_mem_iff relation need sum p remaining bound
    (ordered.imp (fun less => Nat.le_of_lt less)), row, List.contains_iff_mem]
  constructor
  · exact fun accepted => accepted.2.2
  · exact fun member => ⟨sublist.mem member, within q member, member⟩

end ProjectEulerSolutions.CliqueSearch
