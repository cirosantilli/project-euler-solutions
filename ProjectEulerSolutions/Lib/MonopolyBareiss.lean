import ProjectEulerSolutions.Lib.BareissBackSubstitution
import ProjectEulerSolutions.Lib.MonopolySystem

namespace ProjectEulerSolutions.Monopoly
open scoped BigOperators
set_option Elab.async false

def augmentedMatrix (sides : Nat) : Matrix (Fin 120) (Fin 121) Int :=
  fun row => Fin.snoc (coefficients sides row) (rightHandSide row)

theorem augmented_coefficients (sides : Nat) : Bareiss.Coefficients (augmentedMatrix sides) = coefficients sides := by
  funext row column
  exact Fin.snoc_castSucc (α := fun _ => Int) (rightHandSide row) (coefficients sides row) column

def initialElimination (sides : Nat) (positive : 0 < sides) : Bareiss.State (augmentedMatrix sides) 0 :=
  Bareiss.initial (augmentedMatrix sides) (by
    simpa only [augmented_coefficients] using (coefficients_bijective sides positive).injective)

/-- Mathematical elimination reaches the source's final back-substitution stage for every positive die size. -/
theorem elimination_exists (sides : Nat) (positive : 0 < sides) :
    ∃ state : Bareiss.State (augmentedMatrix sides) 119, Bareiss.Runs (initialElimination sides positive) state :=
  (initialElimination sides positive).finish 119 119 rfl (by decide)

theorem elimination_solution (sides : Nat) (state : Bareiss.State (augmentedMatrix sides) 119) :
    ∀ row, ∑ column, coefficients sides row column * state.solution column =
      rightHandSide row * state.matrix 119 119 := by
  simpa only [Bareiss.Coefficients, augmentedMatrix, Fin.snoc_castSucc, Fin.snoc_last] using state.solution_equations.1

theorem elimination_sum (sides : Nat) (state : Bareiss.State (augmentedMatrix sides) 119) :
    ∑ column, state.solution column = state.matrix 119 119 := by
  simpa [coefficients, rightHandSide] using elimination_solution sides state 119

end ProjectEulerSolutions.Monopoly
