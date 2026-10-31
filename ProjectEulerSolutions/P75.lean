import ProjectEulerStatements.P75
import ProjectEulerSolutions.Lib.PythagoreanArraySieve

/-! Count singular perimeters using the complete Euclidean sieve for the supplied bound. -/
namespace ProjectEulerSolutions.P75
open PythagoreanArraySieve
set_option Elab.async false

def countArray (cells : Array Nat) : Nat :=
  cells.foldl (fun total value => if value = 1 then total + 1 else total) 0

def solveCore (limit : Nat) : Nat := countArray (compute limit)

theorem solveCore_correct (limit : Nat) : solveCore limit =
    ProjectEulerStatements.P75.countSingular limit := by
  rw [solveCore, countArray, compute_encode, ← Array.foldl_toList, encode_toList, List.foldl_map]
  exact ProjectEulerStatements.P75.countSingular_table limit

def solve (limit : Int) : Int := if limit < 0 then 0 else solveCore limit.toNat

theorem solve_result (limit : Int) : solve limit = ProjectEulerStatements.P75.result limit := by
  by_cases negative : limit < 0
  · simpa only [solve, if_pos negative] using
      (ProjectEulerStatements.P75.result_negative limit negative).symm
  · simp only [solve, if_neg negative, solveCore_correct]
    have counted := ProjectEulerStatements.P75.result_nat limit.toNat
    simpa only [Int.toNat_of_nonneg (by omega : 0 ≤ limit)] using counted.symm

end ProjectEulerSolutions.P75
