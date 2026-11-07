import ProjectEulerSolutions.Lib.GcdEliminationState
import ProjectEulerSolutions.Lib.BareissBackSubstitution
import Mathlib.LinearAlgebra.Matrix.Block

/-! The product of triangular pivots is an exact integer scale for reverse substitution. -/
namespace ProjectEulerSolutions.GcdElimination
open scoped BigOperators
set_option Elab.async false

variable {n : Nat} {original : Matrix (Fin n) (Fin (n + 1)) Int}

theorem State.triangular (state : State original n) (row column : Fin n) (below : column.val < row.val) :
    state.matrix row column.castSucc = 0 := state.eliminated row column.castSucc column.isLt below

theorem State.diagonal_nonzero (state : State original n) (row : Fin n) :
    state.matrix row row.castSucc ≠ 0 := by
  obtain ⟨candidate, bound, nonzero⟩ := GaussianPivot.exists_pivot_of_injective
    (fun index column => (Coefficients state.matrix index column : ℚ)) state.nonsingular row
    (by
      intro later earlier _ below
      change (state.matrix later earlier.castSucc : ℚ) = 0
      exact_mod_cast state.triangular later earlier below)
  have same : candidate = row := by
    by_contra different
    have below : row.val < candidate.val := by
      have unequal : row.val ≠ candidate.val := by intro equal; exact different (Fin.ext equal.symm)
      omega
    exact nonzero (by exact_mod_cast state.triangular candidate row below)
  subst candidate
  exact_mod_cast nonzero

def State.scale (state : State original n) : Int := ∏ row, state.matrix row row.castSucc

theorem State.scale_nonzero (state : State original n) : state.scale ≠ 0 :=
  Finset.prod_ne_zero_iff.mpr (fun row _ => state.diagonal_nonzero row)

theorem State.determinant (state : State original n) : (Coefficients state.matrix).det = state.scale :=
  Matrix.det_of_upperTriangular (by intro row column below; exact state.triangular row column below)

/-- Cramer's integer vector supplies a witness at the source's product-of-diagonals scale. -/
theorem State.integer_witness (state : State original n) :
    ∃ value : Fin n → Int,
      (∀ row, ∑ column, Coefficients original row column * value column = original row (Fin.last n) * state.scale) ∧
      (∀ row, ∑ column, Coefficients state.matrix row column * value column = state.matrix row (Fin.last n) * state.scale) := by
  let value := (Coefficients state.matrix).cramer (fun row => state.matrix row (Fin.last n))
  have finalEquations : ∀ row, ∑ column, Coefficients state.matrix row column * value column =
      state.matrix row (Fin.last n) * state.scale := by
    intro row
    simpa only [Matrix.mulVec, dotProduct, state.determinant, Pi.smul_apply, smul_eq_mul, mul_comm] using
      congrFun (Matrix.mulVec_cramer (Coefficients state.matrix) (fun row => state.matrix row (Fin.last n))) row
  refine ⟨value, ?_, finalEquations⟩
  apply (Bareiss.scaled_equations_iff original value state.scale).mp
  apply (state.equations _).mp
  exact (Bareiss.scaled_equations_iff state.matrix value state.scale).mpr finalEquations

theorem State.numerator_eq (state : State original n) (scale : Int) (value known : Fin n → Int)
    (equations : ∀ row, ∑ column, Coefficients state.matrix row column * value column = state.matrix row (Fin.last n) * scale)
    (row : Fin n) (agrees : ∀ column, row.val < column.val → known column = value column) :
    Bareiss.numerator state.matrix scale known row = state.matrix row row.castSucc * value row := by
  rw [Bareiss.numerator, ← equations row, Finset.sum_filter, ← Finset.sum_sub_distrib]
  have collapsed := Finset.sum_eq_single row (s := Finset.univ)
    (f := fun column => Coefficients state.matrix row column * value column -
      if row.val < column.val then Coefficients state.matrix row column * known column else 0)
    (by
      intro column _ different
      by_cases above : row.val < column.val
      · simp only [if_pos above, agrees column above, sub_self]
      · have below : column.val < row.val := by
          have unequal : column.val ≠ row.val := by intro equal; exact different (Fin.ext equal)
          omega
        simp [Coefficients, state.triangular row column below, above])
    (by simp)
  simpa only [Nat.lt_irrefl, if_false, sub_zero] using collapsed

theorem State.back_value (state : State original n) (scale : Int) (value known : Fin n → Int)
    (equations : ∀ row, ∑ column, Coefficients state.matrix row column * value column = state.matrix row (Fin.last n) * scale)
    (row : Fin n) (agrees : ∀ column, row.val < column.val → known column = value column) :
    (Bareiss.numerator state.matrix scale known row).fdiv (state.matrix row row.castSucc) = value row := by
  rw [state.numerator_eq scale value known equations row agrees]
  rw [Int.fdiv_eq_ediv_of_dvd (dvd_mul_right _ _)]
  exact Int.mul_ediv_cancel_left _ (state.diagonal_nonzero row)

theorem State.backwards_correct (state : State original n) (scale : Int) (value : Fin n → Int)
    (equations : ∀ row, ∑ column, Coefficients state.matrix row column * value column = state.matrix row (Fin.last n) * scale)
    (count : Nat) (bound : count ≤ n) (known : Fin n → Int)
    (agrees : ∀ row, count ≤ row.val → known row = value row) :
    Bareiss.backwards state.matrix scale count bound known = value := by
  induction count generalizing known with
  | zero => exact funext (fun row => agrees row (Nat.zero_le _))
  | succ count ih =>
      let row : Fin n := ⟨count, by omega⟩
      change Bareiss.backwards state.matrix scale count _
        (Function.update known row ((Bareiss.numerator state.matrix scale known row).fdiv (state.matrix row row.castSucc))) = value
      apply ih
      intro index inside
      by_cases same : index = row
      · subst index
        rw [Function.update_self]
        exact state.back_value scale value known equations row (fun column above => agrees column (by dsimp [row] at above; omega))
      · rw [Function.update_of_ne same]
        apply agrees index
        have different : index.val ≠ count := by intro equal; exact same (Fin.ext equal)
        omega

def State.solution (state : State original n) : Fin n → Int :=
  Bareiss.backwards state.matrix state.scale n (Nat.le_refl _) (fun _ => 0)

/-- Reverse substitution performs only exact divisions and solves both the original and final system. -/
theorem State.solution_equations (state : State original n) :
    (∀ row, ∑ column, Coefficients original row column * state.solution column = original row (Fin.last n) * state.scale) ∧
    (∀ row, ∑ column, Coefficients state.matrix row column * state.solution column = state.matrix row (Fin.last n) * state.scale) := by
  obtain ⟨value, originalEquations, finalEquations⟩ := state.integer_witness
  have correct := state.backwards_correct state.scale value finalEquations n (Nat.le_refl _) (fun _ => 0)
    (by intro row outside; have inside := row.isLt; omega)
  simpa only [State.solution, correct] using And.intro originalEquations finalEquations

end ProjectEulerSolutions.GcdElimination
