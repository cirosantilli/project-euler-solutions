import ProjectEulerSolutions.Lib.StationaryWeights
import Mathlib.LinearAlgebra.FiniteDimensional.Basic

/-! Replacing one redundant stationarity equation by normalization gives an invertible system. -/
namespace ProjectEulerSolutions.StationaryWeights
open scoped BigOperators
set_option Elab.async false

variable {ι : Type*} [Fintype ι]

def normalizedSystem [DecidableEq ι] (weights : ι → ι → ℚ) (denominator : ℚ) (root : ι) :
    (ι → ℚ) →ₗ[ℚ] (ι → ℚ) where
  toFun value j := if j = root then ∑ i, value i else
    denominator * value j - ∑ i, weights i j * value i
  map_add' left right := by
    funext j
    by_cases same : j = root
    · simp [same, Finset.sum_add_distrib]
    · simp [same, mul_add, Finset.sum_add_distrib, sub_add_sub_comm]
  map_smul' scalar value := by
    funext j
    by_cases same : j = root
    · simp [same, Finset.mul_sum]
    · simp [same, mul_sub, Finset.mul_sum, mul_left_comm]

/-- The omitted stationary equation follows from conservation of total mass. -/
theorem stationary_of_except (weights : ι → ι → ℚ) (denominator : ℚ)
    (rowSum : ∀ i, ∑ j, weights i j = denominator) (root : ι) (value : ι → ℚ)
    (stationary : ∀ j, j ≠ root → denominator * value j = ∑ i, weights i j * value i) :
    ∀ j, denominator * value j = ∑ i, weights i j * value i := by
  classical
  have total : (∑ j, (denominator * value j - ∑ i, weights i j * value i)) = 0 := by
    rw [Finset.sum_sub_distrib, Finset.sum_comm]
    have equal : (∑ i, ∑ j, weights i j * value i) = ∑ i, denominator * value i := by
      apply Finset.sum_congr rfl
      intro i _
      rw [← Finset.sum_mul, rowSum i]
    rw [equal, sub_self]
  have single : (∑ j, (denominator * value j - ∑ i, weights i j * value i)) =
      denominator * value root - ∑ i, weights i root * value i := by
    apply Finset.sum_eq_single root
    · intro j _ different
      exact sub_eq_zero.mpr (stationary j different)
    · simp
  intro j
  by_cases same : j = root
  · subst j
    exact sub_eq_zero.mp (single.symm.trans total)
  · exact stationary j same

/-- Every right-hand side has exactly one solution in the normalized stationary system. -/
theorem normalizedSystem_bijective [DecidableEq ι] (weights : ι → ι → ℚ) (denominator : ℚ)
    (nonnegative : ∀ i j, 0 ≤ weights i j) (rowSum : ∀ i, ∑ j, weights i j = denominator)
    (omitted root : ι) (reachable : ∀ i, Relation.ReflTransGen (fun a b => 0 < weights a b) i root) :
    Function.Bijective (normalizedSystem weights denominator omitted) := by
  have injective : Function.Injective (normalizedSystem weights denominator omitted) := by
    intro left right equal
    have zeroSystem : normalizedSystem weights denominator omitted (left - right) = 0 := by
      rw [map_sub, equal, sub_self]
    have zeroSum : ∑ i, (left - right) i = 0 := by
      simpa [normalizedSystem] using congrFun zeroSystem omitted
    have stationary := stationary_of_except weights denominator rowSum omitted (left - right) (by
      intro j different
      apply sub_eq_zero.mp
      simpa [normalizedSystem, different] using congrFun zeroSystem j)
    have zero := zero_of_sum_zero weights denominator nonnegative rowSum root reachable (left - right) stationary zeroSum
    funext i
    exact sub_eq_zero.mp (zero i)
  exact ⟨injective, LinearMap.surjective_of_injective injective⟩

/-- A normalized rational probability vector exists; no numerical convergence assumption is needed. -/
theorem exists_stationary (weights : ι → ι → ℚ) (denominator : ℚ)
    (nonnegative : ∀ i j, 0 ≤ weights i j) (rowSum : ∀ i, ∑ j, weights i j = denominator)
    (root : ι) (reachable : ∀ i, Relation.ReflTransGen (fun a b => 0 < weights a b) i root) :
    ∃ value : ι → ℚ, (∀ i, 0 ≤ value i) ∧ (∑ i, value i = 1) ∧
      ∀ j, denominator * value j = ∑ i, weights i j * value i := by
  classical
  obtain ⟨value, equations⟩ := (normalizedSystem_bijective weights denominator nonnegative rowSum root root reachable).surjective
    (fun j => if j = root then 1 else 0)
  have normalized : ∑ i, value i = 1 := by
    simpa [normalizedSystem] using congrFun equations root
  have stationary := stationary_of_except weights denominator rowSum root value (by
    intro j different
    apply sub_eq_zero.mp
    simpa [normalizedSystem, different] using congrFun equations j)
  exact ⟨value, nonnegative_of_positive_sum weights denominator nonnegative rowSum root reachable value stationary
    (by rw [normalized]; norm_num), normalized, stationary⟩

end ProjectEulerSolutions.StationaryWeights
