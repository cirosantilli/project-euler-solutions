import ProjectEulerSolutions.Lib.StationarySystem
import Mathlib.LinearAlgebra.Matrix.ToLin

/-! The transposed stationary equations in the sign convention used by integer elimination. -/
namespace ProjectEulerSolutions.StationaryWeights
open scoped BigOperators
set_option Elab.async false

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

def coefficientMatrix (weights : ι → ι → ℚ) (denominator : ℚ) (root : ι) : Matrix ι ι ℚ :=
  fun row column => if row = root then 1 else weights column row - if column = row then denominator else 0

theorem coefficientMatrix_mulVec (weights : ι → ι → ℚ) (denominator : ℚ) (root : ι)
    (value : ι → ℚ) (row : ι) :
    (coefficientMatrix weights denominator root).mulVec value row =
      if row = root then normalizedSystem weights denominator root value row
      else -(normalizedSystem weights denominator root value row) := by
  by_cases same : row = root
  · simp [coefficientMatrix, Matrix.mulVec, dotProduct, normalizedSystem, same]
  · simp [coefficientMatrix, Matrix.mulVec, dotProduct, normalizedSystem, same,
      sub_mul, Finset.sum_sub_distrib, ite_mul]

/-- Negating the non-normalization equations preserves nonsingularity. -/
theorem coefficientMatrix_bijective (weights : ι → ι → ℚ) (denominator : ℚ) (root : ι)
    (bijective : Function.Bijective (normalizedSystem weights denominator root)) :
    Function.Bijective (coefficientMatrix weights denominator root).mulVec := by
  have injective : Function.Injective (coefficientMatrix weights denominator root).mulVec := by
    intro left right equal
    apply bijective.injective
    funext row
    have equalRow := congrFun equal row
    rw [coefficientMatrix_mulVec, coefficientMatrix_mulVec] at equalRow
    by_cases same : row = root <;> simpa [same] using equalRow
  exact ⟨injective, LinearMap.surjective_of_injective (f := (coefficientMatrix weights denominator root).mulVecLin) injective⟩

end ProjectEulerSolutions.StationaryWeights
