import ProjectEulerSolutions.P6
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Tactic
open scoped BigOperators
namespace ProjectEulerSolutions.P6

lemma sum_Icc_id_mul_two (n : Nat) :
    (∑ i ∈ Finset.Icc 1 n, i) * 2 = n * (n + 1) := by
  induction n with
  | zero => simp
  | succ n ih =>
      rw [Finset.sum_Icc_succ_top]
      · nlinarith
      · omega

lemma sum_Icc_id (n : Nat) :
    (∑ i ∈ Finset.Icc 1 n, i) = n * (n + 1) / 2 := by
  rw [← sum_Icc_id_mul_two n]
  exact (Nat.mul_div_cancel _ (by decide : 0 < 2)).symm

lemma sum_Icc_sq_mul_six (n : Nat) :
    (∑ i ∈ Finset.Icc 1 n, i ^ 2) * 6 = n * (n + 1) * (2 * n + 1) := by
  induction n with
  | zero => simp
  | succ n ih =>
      rw [Finset.sum_Icc_succ_top]
      · nlinarith
      · omega

lemma sum_Icc_sq (n : Nat) :
    (∑ i ∈ Finset.Icc 1 n, i ^ 2) = n * (n + 1) * (2 * n + 1) / 6 := by
  rw [← sum_Icc_sq_mul_six n]
  exact (Nat.mul_div_cancel _ (by decide : 0 < 6)).symm

theorem equiv (n : Nat) : ProjectEulerStatements.P6.naive n = solve n := by
  unfold ProjectEulerStatements.P6.naive ProjectEulerStatements.P6.squareSum
    ProjectEulerStatements.P6.sumSquares
  unfold solve squareOfSum sumOfSquares
  rw [sum_Icc_id, sum_Icc_sq]
  simp [pow_two]
end ProjectEulerSolutions.P6
