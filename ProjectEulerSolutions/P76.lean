import ProjectEulerStatements.P76
import ProjectEulerSolutions.Lib.CoinChangeArray

/-! A complete partition table for the supplied signed input. -/
namespace ProjectEulerSolutions.P76
open CoinChangeArray
set_option Elab.async false

def countSummations (n : Nat) : Nat := (compute n (List.range' 1 (n-1))).getD n 0

theorem countSummations_correct (n : Nat) :
    countSummations n = ProjectEulerStatements.P76.countPartitions n := by
  have positive : ∀ part ∈ List.range' 1 (n-1), 0 < part := by
    intro part member
    obtain ⟨index, smaller, same⟩ := List.mem_range'.mp member
    omega
  rw [countSummations, compute_encode _ _ positive]
  simp [encode, ProjectEulerStatements.P76.countPartitions,
    ProjectEulerStatements.P76.countWays_coins, Array.getD]

def solve (n : Int) : Int := if n < 0 then 0 else countSummations n.toNat

theorem solve_result (n : Int) : solve n = ProjectEulerStatements.P76.result n := by
  by_cases negative : n < 0
  · simpa only [solve, if_pos negative] using
      (ProjectEulerStatements.P76.result_negative n negative).symm
  · simp only [solve, if_neg negative, countSummations_correct]
    have counted := ProjectEulerStatements.P76.result_nat n.toNat
    simpa only [Int.toNat_of_nonneg (by omega : 0 ≤ n)] using counted.symm

end ProjectEulerSolutions.P76
