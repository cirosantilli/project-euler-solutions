import ProjectEulerSolutions.Lib.MonopolyEncoded
import ProjectEulerSolutions.Lib.StationaryMatrix

/-! Exact coefficients and nonsingularity of the 120-state Python linear system. -/
namespace ProjectEulerSolutions.Monopoly
open scoped BigOperators
set_option Elab.async false

def coefficients (sides : Nat) : Matrix (Fin 120) (Fin 120) Int :=
  fun row column => if row = 119 then 1 else
    encodedWeight sides column row - if column = row then 256 * (sides : Int) * sides else 0

def rightHandSide (row : Fin 120) : Int := if row = 119 then 1 else 0

theorem coefficients_eq (sides : Nat) :
    (fun row column => (coefficients sides row column : ℚ)) =
      (StationaryWeights.coefficientMatrix (matrix sides) (256 * (sides : ℚ) * sides)
        (stateIndex.symm 119)).submatrix stateIndex.symm stateIndex.symm := by
  funext row column
  simp [coefficients, StationaryWeights.coefficientMatrix, Matrix.submatrix, encodedWeight_eq]

/-- The coefficient matrix actually used by integer elimination is nonsingular. -/
theorem coefficients_bijective (sides : Nat) (positive : 0 < sides) :
    Function.Bijective (Matrix.mulVec (fun row column => (coefficients sides row column : ℚ))) := by
  rw [coefficients_eq]
  have system := StationaryWeights.coefficientMatrix_bijective (matrix sides) (256 * (sides : ℚ) * sides)
    (stateIndex.symm 119) (stationary_system_bijective sides positive _)
  have injective : Function.Injective
      ((StationaryWeights.coefficientMatrix (matrix sides) (256 * (sides : ℚ) * sides)
        (stateIndex.symm 119)).submatrix stateIndex.symm stateIndex.symm).mulVec := by
    intro left right equal
    simp only [Matrix.submatrix_mulVec_equiv] at equal
    have equalState :
        (StationaryWeights.coefficientMatrix (matrix sides) (256 * (sides : ℚ) * sides) (stateIndex.symm 119)).mulVec
          (left ∘ stateIndex) =
        (StationaryWeights.coefficientMatrix (matrix sides) (256 * (sides : ℚ) * sides) (stateIndex.symm 119)).mulVec
          (right ∘ stateIndex) := by
      funext state
      simpa only [Function.comp_apply, Equiv.symm_apply_apply] using congrFun equal (stateIndex state)
    have same := system.injective equalState
    funext index
    simpa only [Function.comp_apply, Equiv.apply_symm_apply] using congrFun same (stateIndex.symm index)
  exact ⟨injective, LinearMap.surjective_of_injective
    (f := ((StationaryWeights.coefficientMatrix (matrix sides) (256 * (sides : ℚ) * sides)
      (stateIndex.symm 119)).submatrix stateIndex.symm stateIndex.symm).mulVecLin) injective⟩

end ProjectEulerSolutions.Monopoly
