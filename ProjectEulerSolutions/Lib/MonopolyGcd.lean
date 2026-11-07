import ProjectEulerSolutions.Lib.GcdEliminationSolution
import ProjectEulerSolutions.Lib.MonopolyIntegerWeights

namespace ProjectEulerSolutions.Monopoly
open scoped BigOperators
set_option Elab.async false

def initialGcdElimination (sides : Nat) (positive : 0 < sides) : GcdElimination.State (augmentedMatrix sides) 0 :=
  GcdElimination.initial (augmentedMatrix sides) (by
    simpa only [GcdElimination.Coefficients, augmentedMatrix, Fin.snoc_castSucc] using (coefficients_bijective sides positive).injective)

theorem gcd_elimination_exists (sides : Nat) (positive : 0 < sides) :
    ∃ state : GcdElimination.State (augmentedMatrix sides) 120, GcdElimination.Runs (initialGcdElimination sides positive) state :=
  (initialGcdElimination sides positive).finish 120 120 rfl (Nat.le_refl _)

theorem gcd_elimination_solution (sides : Nat) (state : GcdElimination.State (augmentedMatrix sides) 120) :
    ∀ row, ∑ column, coefficients sides row column * state.solution column = rightHandSide row * state.scale := by
  simpa only [GcdElimination.Coefficients, augmentedMatrix, Fin.snoc_castSucc, Fin.snoc_last] using state.solution_equations.1

theorem gcd_elimination_sum (sides : Nat) (state : GcdElimination.State (augmentedMatrix sides) 120) :
    ∑ column, state.solution column = state.scale := by
  simpa [coefficients, rightHandSide] using gcd_elimination_solution sides state 119

def signedDivisor (values : Fin 120 → Int) (scale : Int) : Int :=
  if scale < 0 then -(IntegerGcdRow.divisor values : Int) else IntegerGcdRow.divisor values

def gcdNormalized (values : Fin 120 → Int) (scale : Int) : Fin 120 → Int :=
  fun index => (values index).fdiv (signedDivisor values scale)

theorem signedDivisor_dvd (values : Fin 120 → Int) (scale : Int) (index : Fin 120) :
    signedDivisor values scale ∣ values index := by
  unfold signedDivisor
  split <;> simpa only [neg_dvd] using IntegerGcdRow.divisor_dvd values index

theorem normalized_scaled (values : Fin 120 → Int) (scale : Int) (index : Fin 120) :
    signedDivisor values scale * gcdNormalized values scale index = values index := by
  rw [gcdNormalized, Int.fdiv_eq_ediv_of_dvd (signedDivisor_dvd values scale index)]
  exact Int.mul_ediv_cancel_of_dvd (signedDivisor_dvd values scale index)

/-- The reference's final signed gcd makes the total positive without changing any stationary equation. -/
theorem gcd_normalized_weights (sides : Nat) (values : Fin 120 → Int) (scale : Int)
    (nonzero : scale ≠ 0) (total : ∑ index, values index = scale)
    (equations : ∀ row, ∑ column, coefficients sides row column * values column = rightHandSide row * scale) :
    IntegerWeights sides (gcdNormalized values scale) := by
  have scaledTotal : signedDivisor values scale * (∑ index, gcdNormalized values scale index) = scale := by
    rw [Finset.mul_sum]
    simpa only [normalized_scaled] using total
  have divisorNonzero : signedDivisor values scale ≠ 0 := by
    intro zero
    rw [zero, zero_mul] at scaledTotal
    exact nonzero scaledTotal.symm
  have naturalPositive : 0 < (IntegerGcdRow.divisor values : Int) := by
    have notZero : IntegerGcdRow.divisor values ≠ 0 := by
      intro zero
      apply divisorNonzero
      simp [signedDivisor, zero]
    exact_mod_cast Nat.pos_of_ne_zero notZero
  refine ⟨?_, ?_⟩
  · by_cases negative : scale < 0
    · simp only [signedDivisor, if_pos negative] at scaledTotal
      nlinarith
    · simp only [signedDivisor, if_neg negative] at scaledTotal
      have positive : 0 < scale := by omega
      nlinarith
  · intro row
    apply mul_left_cancel₀ divisorNonzero
    calc
      signedDivisor values scale * (∑ column, coefficients sides row column * gcdNormalized values scale column) =
          ∑ column, coefficients sides row column * values column := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro column _
        rw [mul_left_comm, normalized_scaled]
      _ = rightHandSide row * scale := equations row
      _ = signedDivisor values scale * (rightHandSide row * ∑ index, gcdNormalized values scale index) := by
        rw [mul_left_comm, scaledTotal]

theorem gcd_solution_weights (sides : Nat) (state : GcdElimination.State (augmentedMatrix sides) 120) :
    IntegerWeights sides (gcdNormalized state.solution state.scale) :=
  gcd_normalized_weights sides state.solution state.scale state.scale_nonzero
    (gcd_elimination_sum sides state) (gcd_elimination_solution sides state)

end ProjectEulerSolutions.Monopoly
