import ProjectEulerStatements.P91
import ProjectEulerSolutions.Lib.LatticeTriangleCount
import Mathlib.Algebra.BigOperators.Group.List.Basic

/-! The existing natural-coordinate reference folds implement the exhaustive geometric count. -/
namespace ProjectEulerSolutions.LatticeRightTriangles
set_option Elab.async false
open scoped BigOperators

theorem axisDistance_cast (a b : Nat) :
    ((if b ≥ a then b - a else a - b : Nat) : Int) = axisDistance a b := by
  by_cases order : a ≤ b
  · simp [axisDistance, order, Int.natCast_sub order]
  · have reverse : b ≤ a := by omega
    simp [axisDistance, order, Int.natCast_sub reverse]

theorem natural_isRight (a b c d : Nat) :
    ProjectEulerStatements.P91.isRight a b c d = isRight a b c d := by
  apply Bool.eq_iff_iff.mpr
  simp only [ProjectEulerStatements.P91.isRight, isRight, decide_eq_true_eq, norm, distance,
    ← axisDistance_cast]
  norm_cast

theorem reference_step (a b c d acc : Nat) :
    (if a = 0 ∧ b = 0 then acc
      else if c = 0 ∧ d = 0 then acc
      else if a = c ∧ b = d then acc
      else if ProjectEulerStatements.P91.isRight a b c d then acc + 1 else acc) =
      acc + rightWeight ((a : Int), (b : Int)) ((c : Int), (d : Int)) := by
  simp only [rightWeight, nondegenerate, ne_eq, Prod.mk.injEq, Int.natCast_inj, Int.natCast_eq_zero,
    ← natural_isRight]
  have reversed : (c = a ∧ d = b) ↔ (a = c ∧ b = d) := by omega
  simp only [reversed]
  split_ifs <;> first | omega | tauto

private theorem fold_add_sum (items : List Nat) (f : Nat → Nat) (initial : Nat) :
    items.foldl (fun acc x => acc + f x) initial = initial + (items.map f).sum := by
  induction items generalizing initial with
  | nil => simp
  | cons first rest ih => simp [ih, Nat.add_assoc]

theorem fold_range_add (size : Nat) (f : Nat → Nat) (initial : Nat) :
    (List.range size).foldl (fun acc x => acc + f x) initial = initial + ∑ x ∈ Finset.range size, f x := by
  rw [fold_add_sum]
  have equal : (List.range size).toFinset = Finset.range size := by
    ext x
    simp
  rw [← equal, List.sum_toFinset f (List.nodup_range (n := size))]

/-- The four nested source-order folds count every ordered nondegenerate pair exactly once. -/
theorem natural_reference_eq (n : Nat) : ProjectEulerStatements.P91.naive n = referenceCount n := by
  unfold ProjectEulerStatements.P91.naive ProjectEulerStatements.P91.countTriangles referenceCount
  simp_rw [reference_step, fold_range_add]
  simp only [Nat.zero_add]

end ProjectEulerSolutions.LatticeRightTriangles
