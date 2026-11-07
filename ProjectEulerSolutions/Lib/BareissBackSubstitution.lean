import ProjectEulerSolutions.Lib.BareissWitness

/-! Reverse substitution computes the integer solution at the final determinant scale. -/
namespace ProjectEulerSolutions.Bareiss
open scoped BigOperators
set_option Elab.async false

variable {n : Nat}

def numerator (matrix : Matrix (Fin n) (Fin (n + 1)) Int) (scale : Int) (known : Fin n → Int) (row : Fin n) : Int :=
  matrix row (Fin.last n) * scale -
    ∑ column ∈ Finset.univ.filter (fun column : Fin n => row.val < column.val), Coefficients matrix row column * known column

theorem State.numerator_eq {size : Nat} {original : Matrix (Fin (size + 1)) (Fin (size + 2)) Int}
    (state : State original size) (scale : Int) (value known : Fin (size + 1) → Int)
    (equations : ∀ row, ∑ column, Coefficients state.matrix row column * value column = state.matrix row (Fin.last (size + 1)) * scale)
    (row : Fin (size + 1)) (agrees : ∀ column, row.val < column.val → known column = value column) :
    numerator state.matrix scale known row = state.matrix row row.castSucc * value row := by
  rw [numerator, ← equations row, Finset.sum_filter, ← Finset.sum_sub_distrib]
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

theorem State.back_value {size : Nat} {original : Matrix (Fin (size + 1)) (Fin (size + 2)) Int}
    (state : State original size) (scale : Int) (value known : Fin (size + 1) → Int)
    (equations : ∀ row, ∑ column, Coefficients state.matrix row column * value column = state.matrix row (Fin.last (size + 1)) * scale)
    (row : Fin (size + 1)) (agrees : ∀ column, row.val < column.val → known column = value column) :
    (numerator state.matrix scale known row).fdiv (state.matrix row row.castSucc) = value row := by
  rw [state.numerator_eq scale value known equations row agrees]
  rw [Int.fdiv_eq_ediv_of_dvd (dvd_mul_right _ _)]
  exact Int.mul_ediv_cancel_left _ (state.diagonal_nonzero row)

def backwards (matrix : Matrix (Fin n) (Fin (n + 1)) Int) (scale : Int) :
    (count : Nat) → count ≤ n → (Fin n → Int) → (Fin n → Int)
  | 0, _, known => known
  | count + 1, bound, known =>
      let row : Fin n := ⟨count, Nat.lt_of_lt_of_le (Nat.lt_succ_self count) bound⟩
      backwards matrix scale count (by omega)
        (Function.update known row ((numerator matrix scale known row).fdiv (matrix row row.castSucc)))

/-- Every reverse step uses an exact quotient and extends the already-correct suffix by one entry. -/
theorem State.backwards_correct {size : Nat} {original : Matrix (Fin (size + 1)) (Fin (size + 2)) Int}
    (state : State original size) (scale : Int) (value : Fin (size + 1) → Int)
    (equations : ∀ row, ∑ column, Coefficients state.matrix row column * value column = state.matrix row (Fin.last (size + 1)) * scale)
    (count : Nat) (bound : count ≤ size + 1) (known : Fin (size + 1) → Int)
    (agrees : ∀ row, count ≤ row.val → known row = value row) :
    backwards state.matrix scale count bound known = value := by
  induction count generalizing known with
  | zero => exact funext (fun row => agrees row (Nat.zero_le _))
  | succ count ih =>
      let row : Fin (size + 1) := ⟨count, by omega⟩
      change backwards state.matrix scale count _
        (Function.update known row ((numerator state.matrix scale known row).fdiv (state.matrix row row.castSucc))) = value
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

def State.solution {size : Nat} {original : Matrix (Fin (size + 1)) (Fin (size + 2)) Int}
    (state : State original size) : Fin (size + 1) → Int :=
  backwards state.matrix (state.matrix (Fin.last size) (Fin.last size).castSucc) (size + 1) (Nat.le_refl _) (fun _ => 0)

/-- Complete reverse substitution solves the original integer system at the nonzero final-diagonal scale. -/
theorem State.solution_equations {size : Nat} {original : Matrix (Fin (size + 1)) (Fin (size + 2)) Int}
    (state : State original size) :
    (∀ row, ∑ column, Coefficients original row column * state.solution column =
      original row (Fin.last (size + 1)) * state.matrix (Fin.last size) (Fin.last size).castSucc) ∧
    (∀ row, ∑ column, Coefficients state.matrix row column * state.solution column =
      state.matrix row (Fin.last (size + 1)) * state.matrix (Fin.last size) (Fin.last size).castSucc) := by
  obtain ⟨value, originalEquations, finalEquations⟩ := state.integer_witness
  have correct := state.backwards_correct _ value finalEquations (size + 1) (Nat.le_refl _) (fun _ => 0)
    (by intro row outside; have inside := row.isLt; omega)
  simpa only [State.solution, correct] using And.intro originalEquations finalEquations

end ProjectEulerSolutions.Bareiss
