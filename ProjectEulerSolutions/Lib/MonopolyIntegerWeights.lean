import ProjectEulerSolutions.Lib.MonopolyBareiss

namespace ProjectEulerSolutions.Monopoly
open scoped BigOperators
set_option Elab.async false

/-- Integer stationary weights with a positive common normalization scale. -/
def IntegerWeights (sides : Nat) (values : Fin 120 → Int) : Prop :=
  0 < ∑ index, values index ∧
    ∀ row, ∑ column, coefficients sides row column * values column = rightHandSide row * ∑ index, values index

theorem signed_solution (sides : Nat) (state : Bareiss.State (augmentedMatrix sides) 119) :
    IntegerWeights sides (fun index => if state.matrix 119 119 < 0 then -state.solution index else state.solution index) := by
  have total := elimination_sum sides state
  have equations := elimination_solution sides state
  have nonzero : state.matrix 119 119 ≠ 0 := state.last_nonzero
  by_cases negative : state.matrix 119 119 < 0
  · simp only [IntegerWeights, if_pos negative, Finset.sum_neg_distrib, mul_neg, total]
    exact ⟨by omega, fun row => congrArg Neg.neg (equations row)⟩
  · simp only [IntegerWeights, if_neg negative, total]
    exact ⟨by omega, equations⟩

/-- Any two exact integer solvers differ only by their positive total weight. -/
theorem IntegerWeights.proportional {sides : Nat} (positive : 0 < sides) {left right : Fin 120 → Int}
    (validLeft : IntegerWeights sides left) (validRight : IntegerWeights sides right) (index : Fin 120) :
    left index * (∑ column, right column) = right index * (∑ column, left column) := by
  have equal : (fun column => (left column : ℚ) * ((∑ index, right index : Int) : ℚ)) =
      (fun column => (right column : ℚ) * ((∑ index, left index : Int) : ℚ)) := by
    apply (coefficients_bijective sides positive).injective
    funext row
    have leftEquation : (∑ column, (coefficients sides row column : ℚ) * (left column : ℚ)) =
        (rightHandSide row : ℚ) * ((∑ index, left index : Int) : ℚ) := by exact_mod_cast validLeft.2 row
    have rightEquation : (∑ column, (coefficients sides row column : ℚ) * (right column : ℚ)) =
        (rightHandSide row : ℚ) * ((∑ index, right index : Int) : ℚ) := by exact_mod_cast validRight.2 row
    simp only [Matrix.mulVec, dotProduct, ← mul_assoc, ← Finset.sum_mul, leftEquation, rightEquation]
    ring
  exact_mod_cast congrFun equal index

end ProjectEulerSolutions.Monopoly
