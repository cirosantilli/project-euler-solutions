import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Data.Rat.BigOperators
import Mathlib.Tactic.Linarith

/-! Uniqueness of normalized stationary weights when every state can reach a common state. -/
namespace ProjectEulerSolutions.StationaryWeights
open scoped BigOperators
set_option Elab.async false

variable {ι : Type*} [Fintype ι]

/-- Taking the positive part preserves stationarity: total mass leaves no room for cancellation. -/
theorem positive_part (weights : ι → ι → ℚ) (denominator : ℚ)
    (nonnegative : ∀ i j, 0 ≤ weights i j) (rowSum : ∀ i, ∑ j, weights i j = denominator)
    (value : ι → ℚ)
    (stationary : ∀ j, denominator * value j = ∑ i, weights i j * value i) :
    ∀ j, denominator * max (value j) 0 = ∑ i, weights i j * max (value i) 0 := by
  classical
  have lower (j : ι) : denominator * max (value j) 0 ≤ ∑ i, weights i j * max (value i) 0 := by
    by_cases sign : 0 ≤ value j
    · rw [max_eq_left sign, stationary j]
      exact Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (le_max_left _ _) (nonnegative i j)
    · rw [max_eq_right (le_of_not_ge sign), mul_zero]
      exact Finset.sum_nonneg fun i _ => mul_nonneg (nonnegative i j) (le_max_right _ _)
  have total : (∑ j, ∑ i, weights i j * max (value i) 0) = ∑ j, denominator * max (value j) 0 := by
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro i _
    rw [← Finset.sum_mul, rowSum i]
  have zero : (∑ j, ((∑ i, weights i j * max (value i) 0) - denominator * max (value j) 0)) = 0 := by
    rw [Finset.sum_sub_distrib, total, sub_self]
  intro j
  have each := (Finset.sum_eq_zero_iff_of_nonneg (fun i _ => sub_nonneg.mpr (lower i))).mp zero j (Finset.mem_univ j)
  linarith

/-- A strictly positive component of a stationary signed vector stays positive along an edge. -/
theorem positive_step (weights : ι → ι → ℚ) (denominator : ℚ)
    (nonnegative : ∀ i j, 0 ≤ weights i j) (rowSum : ∀ i, ∑ j, weights i j = denominator)
    (value : ι → ℚ)
    (stationary : ∀ j, denominator * value j = ∑ i, weights i j * value i)
    (i j : ι) (source : 0 < value i) (edge : 0 < weights i j) : 0 < value j := by
  classical
  have bounded : weights i j * max (value i) 0 ≤ ∑ k, weights k j * max (value k) 0 :=
    Finset.single_le_sum (fun k _ => mul_nonneg (nonnegative k j) (le_max_right _ _)) (Finset.mem_univ i)
  have contribution : 0 < weights i j * max (value i) 0 :=
    mul_pos edge (lt_of_lt_of_le source (le_max_left _ _))
  have destination : 0 < denominator * max (value j) 0 := by
    rw [positive_part weights denominator nonnegative rowSum value stationary j]
    exact lt_of_lt_of_le contribution bounded
  by_contra missing
  have sign : value j ≤ 0 := le_of_not_gt missing
  simp [max_eq_right sign] at destination

private theorem positive_reaches (weights : ι → ι → ℚ) (denominator : ℚ)
    (nonnegative : ∀ i j, 0 ≤ weights i j) (rowSum : ∀ i, ∑ j, weights i j = denominator)
    (value : ι → ℚ)
    (stationary : ∀ j, denominator * value j = ∑ i, weights i j * value i)
    (i j : ι) (source : 0 < value i)
    (path : Relation.ReflTransGen (fun a b => 0 < weights a b) i j) : 0 < value j := by
  induction path with
  | refl => exact source
  | tail path edge ih => exact positive_step weights denominator nonnegative rowSum value stationary _ _ ih edge

/-- A stationary signed vector with zero total mass must vanish. -/
theorem zero_of_sum_zero (weights : ι → ι → ℚ) (denominator : ℚ)
    (nonnegative : ∀ i j, 0 ≤ weights i j) (rowSum : ∀ i, ∑ j, weights i j = denominator)
    (root : ι)
    (reachable : ∀ i, Relation.ReflTransGen (fun a b => 0 < weights a b) i root)
    (value : ι → ℚ) (stationary : ∀ j, denominator * value j = ∑ i, weights i j * value i)
    (zeroSum : ∑ i, value i = 0) : ∀ i, value i = 0 := by
  classical
  have negStationary (j : ι) : denominator * -value j = ∑ i, weights i j * -value i := by
    simp only [mul_neg, Finset.sum_neg_distrib, stationary j]
  have noPositive : ¬∃ i, 0 < value i := by
    rintro ⟨i, source⟩
    have rootPositive := positive_reaches weights denominator nonnegative rowSum value stationary i root source (reachable i)
    by_cases hasNegative : ∃ j, value j < 0
    · obtain ⟨j, negative⟩ := hasNegative
      have rootNegative := positive_reaches weights denominator nonnegative rowSum (fun k => -value k)
        negStationary j root (by linarith) (reachable j)
      linarith
    · have signs (j : ι) : 0 ≤ value j := le_of_not_gt (fun negative => hasNegative ⟨j, negative⟩)
      have bound : value i ≤ ∑ j, value j := Finset.single_le_sum (fun j _ => signs j) (Finset.mem_univ i)
      rw [zeroSum] at bound
      linarith
  have signs (i : ι) : 0 ≤ -value i := by
    have := not_exists.mp noPositive i
    linarith
  have zero : (∑ i, -value i) = 0 := by rw [Finset.sum_neg_distrib, zeroSum, neg_zero]
  intro i
  have each := (Finset.sum_eq_zero_iff_of_nonneg (fun j _ => signs j)).mp zero i (Finset.mem_univ i)
  linarith

/-- Normalization makes the stationary vector unique, including chains with transient states or periods. -/
theorem unique (weights : ι → ι → ℚ) (denominator : ℚ)
    (nonnegative : ∀ i j, 0 ≤ weights i j) (rowSum : ∀ i, ∑ j, weights i j = denominator)
    (root : ι)
    (reachable : ∀ i, Relation.ReflTransGen (fun a b => 0 < weights a b) i root)
    (left right : ι → ℚ)
    (leftStationary : ∀ j, denominator * left j = ∑ i, weights i j * left i)
    (rightStationary : ∀ j, denominator * right j = ∑ i, weights i j * right i)
    (sameSum : ∑ i, left i = ∑ i, right i) : left = right := by
  have difference (j : ι) : denominator * (left j - right j) = ∑ i, weights i j * (left i - right i) := by
    simp only [mul_sub, Finset.sum_sub_distrib, leftStationary j, rightStationary j]
  have zero := zero_of_sum_zero weights denominator nonnegative rowSum root reachable
    (fun i => left i - right i) difference (by rw [Finset.sum_sub_distrib, sameSum, sub_self])
  funext i
  exact sub_eq_zero.mp (zero i)

/-- A stationary vector of positive total mass cannot have a negative component. -/
theorem nonnegative_of_positive_sum (weights : ι → ι → ℚ) (denominator : ℚ)
    (nonnegative : ∀ i j, 0 ≤ weights i j) (rowSum : ∀ i, ∑ j, weights i j = denominator)
    (root : ι) (reachable : ∀ i, Relation.ReflTransGen (fun a b => 0 < weights a b) i root)
    (value : ι → ℚ) (stationary : ∀ j, denominator * value j = ∑ i, weights i j * value i)
    (positiveSum : 0 < ∑ i, value i) : ∀ i, 0 ≤ value i := by
  classical
  have source : ∃ i, 0 < value i := by
    by_contra missing
    have signs : ∀ i, value i ≤ 0 := fun i => le_of_not_gt (fun h => missing ⟨i, h⟩)
    exact (not_le_of_gt positiveSum) (Finset.sum_nonpos (fun i _ => signs i))
  obtain ⟨i, positive⟩ := source
  have rootPositive := positive_reaches weights denominator nonnegative rowSum value stationary i root positive (reachable i)
  intro j
  by_contra negative
  have negStationary (k : ι) : denominator * -value k = ∑ i, weights i k * -value i := by
    simp only [mul_neg, Finset.sum_neg_distrib, stationary k]
  have rootNegative := positive_reaches weights denominator nonnegative rowSum (fun k => -value k)
    negStationary j root (by linarith) (reachable j)
  linarith

end ProjectEulerSolutions.StationaryWeights
