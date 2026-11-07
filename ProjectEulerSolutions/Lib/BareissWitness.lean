import ProjectEulerSolutions.Lib.BareissRun

/-! The final determinant scale admits an integer solution, making back-substitution exact. -/
namespace ProjectEulerSolutions.Bareiss
open scoped BigOperators
set_option Elab.async false

variable {n : Nat}

theorem scaled_equations_iff (matrix : Matrix (Fin n) (Fin (n + 1)) Int) (value : Fin n → Int) (scale : Int) :
    Equations matrix (Fin.snoc (α := fun _ => ℚ) (fun column => (value column : ℚ)) (-(scale : ℚ))) ↔
      ∀ row, ∑ column, Coefficients matrix row column * value column = matrix row (Fin.last n) * scale := by
  constructor
  · intro equations row
    have equation := equations row
    simp only [Fin.sum_univ_castSucc, Fin.snoc_castSucc, Fin.snoc_last, mul_neg, ← sub_eq_add_neg] at equation
    have equal := sub_eq_zero.mp equation
    exact_mod_cast equal
  · intro equations row
    have equal : (∑ column : Fin n, (matrix row column.castSucc : ℚ) * value column) =
        (matrix row (Fin.last n) : ℚ) * scale := by
      exact_mod_cast equations row
    simp only [Fin.sum_univ_castSucc, Fin.snoc_castSucc, Fin.snoc_last, mul_neg]
    rw [equal, add_neg_cancel]

theorem State.triangular {size : Nat} {original : Matrix (Fin (size + 1)) (Fin (size + 2)) Int}
    (state : State original size) (row column : Fin (size + 1)) (below : column.val < row.val) :
    state.matrix row column.castSucc = 0 :=
  state.eliminated row column.castSucc (by have bound := row.isLt; simp; omega) below

theorem State.diagonal_nonzero {size : Nat} {original : Matrix (Fin (size + 1)) (Fin (size + 2)) Int}
    (state : State original size) (row : Fin (size + 1)) : state.matrix row row.castSucc ≠ 0 := by
  obtain ⟨candidate, bound, nonzero⟩ := GaussianPivot.exists_pivot_of_injective
    (fun row column => (Coefficients state.matrix row column : ℚ)) state.nonsingular row
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

/-- Cramer's integer vector solves both the original and final systems at the actual final-pivot scale. -/
theorem State.integer_witness {size : Nat} {original : Matrix (Fin (size + 1)) (Fin (size + 2)) Int}
    (state : State original size) :
    ∃ value : Fin (size + 1) → Int,
      (∀ row, ∑ column, Coefficients original row column * value column =
        original row (Fin.last (size + 1)) * state.matrix (Fin.last size) (Fin.last size).castSucc) ∧
      (∀ row, ∑ column, Coefficients state.matrix row column * value column =
        state.matrix row (Fin.last (size + 1)) * state.matrix (Fin.last size) (Fin.last size).castSucc) := by
  let coefficients := (Coefficients original).submatrix state.labels id
  let rhs := fun row => original (state.labels row) (Fin.last (size + 1))
  let value := coefficients.cramer rhs
  have originalEquations : ∀ row, ∑ column, Coefficients original row column * value column =
      original row (Fin.last (size + 1)) * state.matrix (Fin.last size) (Fin.last size).castSucc := by
    intro row
    have solved := congrFun (Matrix.mulVec_cramer coefficients rhs) (state.labels.symm row)
    simpa only [coefficients, rhs, Matrix.mulVec, dotProduct, Matrix.submatrix_apply, id_eq,
      Equiv.apply_symm_apply, Pi.smul_apply, smul_eq_mul, ← state.last_determinant, mul_comm] using solved
  refine ⟨value, originalEquations, ?_⟩
  apply (scaled_equations_iff state.matrix value _).mp
  apply (state.equations _).mpr
  exact (scaled_equations_iff original value _).mpr originalEquations

end ProjectEulerSolutions.Bareiss
