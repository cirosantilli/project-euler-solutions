import ProjectEulerSolutions.P28
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Tactic
open scoped BigOperators
namespace ProjectEulerSolutions.P28

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

lemma sixteen_mul_sq_formula (m : Nat) :
    16 * (m * (m + 1) * (2 * m + 1) / 6) =
      16 * m * (m + 1) * (2 * m + 1) / 6 := by
  let S := ∑ i ∈ Finset.Icc 1 m, i ^ 2
  have h : S * 6 = m * (m + 1) * (2 * m + 1) := sum_Icc_sq_mul_six m
  calc
    16 * (m * (m + 1) * (2 * m + 1) / 6) = 16 * ((S * 6) / 6) := by rw [h]
    _ = 16 * S := by rw [Nat.mul_div_cancel S (by decide : 0 < 6)]
    _ = (16 * S * 6) / 6 := by rw [Nat.mul_div_cancel (16 * S) (by decide : 0 < 6)]
    _ = 16 * m * (m + 1) * (2 * m + 1) / 6 := by
      rw [Nat.mul_assoc 16 S 6, h]
      simp [Nat.mul_assoc]

lemma four_mul_id_formula (m : Nat) :
    4 * (m * (m + 1) / 2) = 4 * m * (m + 1) / 2 := by
  let S := ∑ i ∈ Finset.Icc 1 m, i
  have h : S * 2 = m * (m + 1) := sum_Icc_id_mul_two m
  calc
    4 * (m * (m + 1) / 2) = 4 * ((S * 2) / 2) := by rw [h]
    _ = 4 * S := by rw [Nat.mul_div_cancel S (by decide : 0 < 2)]
    _ = (4 * S * 2) / 2 := by rw [Nat.mul_div_cancel (4 * S) (by decide : 0 < 2)]
    _ = 4 * m * (m + 1) / 2 := by
      rw [Nat.mul_assoc 4 S 2, h]
      simp [Nat.mul_assoc]

lemma spiralDiagSum_fold_eq (m : Nat) :
    (List.range (m + 1)).foldl
      (fun acc k => if k = 0 then acc + 1 else acc + ProjectEulerStatements.P28.layerSum k) 0 =
    1 + ∑ k ∈ Finset.Icc 1 m, ProjectEulerStatements.P28.layerSum k := by
  induction m with
  | zero => simp [ProjectEulerStatements.P28.layerSum]
  | succ m ih =>
      rw [show m.succ + 1 = (m + 1).succ by rfl, List.range_succ, List.foldl_append]
      rw [ih]
      rw [Finset.sum_Icc_succ_top]
      · simp [Nat.add_assoc]
      · omega

lemma sum_layerSum (m : Nat) :
    (∑ k ∈ Finset.Icc 1 m, ProjectEulerStatements.P28.layerSum k) =
      16 * (m * (m + 1) * (2 * m + 1) / 6) + 4 * (m * (m + 1) / 2) + 4 * m := by
  simp [ProjectEulerStatements.P28.layerSum, Finset.sum_add_distrib]
  rw [show (∑ x ∈ Finset.Icc 1 m, 16 * x ^ 2) =
        16 * (∑ x ∈ Finset.Icc 1 m, x ^ 2) by
        exact (Finset.mul_sum (s := Finset.Icc 1 m) (f := fun x => x ^ 2) 16).symm]
  rw [show (∑ x ∈ Finset.Icc 1 m, 4 * x) =
        4 * (∑ x ∈ Finset.Icc 1 m, x) by
        exact (Finset.mul_sum (s := Finset.Icc 1 m) (f := fun x => x) 4).symm]
  rw [sum_Icc_sq, sum_Icc_id]
  rw [Nat.mul_comm m 4]

theorem equiv (n : Nat) : ProjectEulerStatements.P28.naive n = solve n := by
  unfold ProjectEulerStatements.P28.naive ProjectEulerStatements.P28.spiralDiagSum solve
  set m := (n - 1) / 2
  rw [spiralDiagSum_fold_eq m, sum_layerSum m]
  rw [sixteen_mul_sq_formula, four_mul_id_formula]
  simp [Nat.add_assoc]
end ProjectEulerSolutions.P28
