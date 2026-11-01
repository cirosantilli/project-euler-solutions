import ProjectEulerSolutions.P91
import ProjectEulerSolutions.Lib.LatticeTriangleCount
import ProjectEulerSolutions.Lib.NaturalIntervalSum

/-! Structural summation of the existing optimized solver's two grid loops. -/
namespace ProjectEulerSolutions.LatticeRightTriangles
set_option Elab.async false
open scoped BigOperators

def rowSum (n x first : Nat) : Nat := ∑ y ∈ Finset.Icc first n, pointContribution n x y
def rowsSum (n first : Nat) : Nat := ∑ x ∈ Finset.Icc first n, rowSum n x 0

theorem optimized_row (n x y acc : Nat) :
    P91.countRightTriangles.loopX.loopY n x y acc = acc + rowSum n x y := by
  by_cases inside : y ≤ n
  · have step := NaturalIntervalSum.step (pointContribution n x) y n inside
    change rowSum n x y = pointContribution n x y + rowSum n x (y + 1) at step
    rw [P91.countRightTriangles.loopX.loopY.eq_1, if_neg (by omega)]
    simp only [Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq]
    by_cases origin : x = 0 ∧ y = 0
    · rw [if_pos origin, optimized_row n x (y + 1) acc, step]
      simp [pointContribution, origin]
    · rw [if_neg origin]
      by_cases axis : x = 0 ∨ y = 0
      · rw [if_pos axis, optimized_row n x (y + 1) (acc + n), step]
        simp [pointContribution, origin, axis, Nat.add_assoc]
      · rw [if_neg axis, optimized_row n x (y + 1), step]
        simp only [pointContribution, origin, axis, ↓reduceIte, P91.gcd, upperSteps, lowerSteps, Nat.add_assoc]
  · rw [P91.countRightTriangles.loopX.loopY.eq_1, if_pos (by omega)]
    have empty := NaturalIntervalSum.empty (pointContribution n x) y n (by omega)
    simp only [rowSum, empty, Nat.add_zero]
termination_by n + 1 - y
decreasing_by all_goals omega

theorem optimized_rows (n x acc : Nat) :
    P91.countRightTriangles.loopX n x acc = acc + rowsSum n x := by
  by_cases inside : x ≤ n
  · have step := NaturalIntervalSum.step (fun x => rowSum n x 0) x n inside
    change rowsSum n x = rowSum n x 0 + rowsSum n (x + 1) at step
    rw [P91.countRightTriangles.loopX.eq_1, if_neg (by omega), optimized_row,
      optimized_rows n (x + 1), step]
    omega
  · rw [P91.countRightTriangles.loopX.eq_1, if_pos (by omega)]
    have empty := NaturalIntervalSum.empty (fun x => rowSum n x 0) x n (by omega)
    simp only [rowsSum, empty, Nat.add_zero]
termination_by n + 1 - x
decreasing_by omega

/-- The executable optimized Lean solver implements the geometric count without changing its loops. -/
theorem natural_optimized_eq (n : Nat) : P91.solve n = optimizedCount n := by
  change P91.countRightTriangles.loopX n 0 (n * n) = _
  rw [optimized_rows]
  simp only [rowsSum, rowSum, ← Nat.range_succ_eq_Icc_zero, optimizedCount]

end ProjectEulerSolutions.LatticeRightTriangles
