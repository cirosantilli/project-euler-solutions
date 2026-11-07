import Mathlib.LinearAlgebra.Matrix.SchurComplement
import Mathlib.Tactic.Ring

/-! The bordered determinant identity behind exact fraction-free elimination. -/
namespace ProjectEulerSolutions.BorderedDeterminants
set_option Elab.async false

variable {ι R : Type*} [Fintype ι] [DecidableEq ι] [CommRing R]

def border (base : Matrix ι ι R) (columns : Matrix ι (Fin 2) R) (rows : Matrix (Fin 2) ι R)
    (corner : Matrix (Fin 2) (Fin 2) R) (row column : Fin 2) : Matrix (ι ⊕ Fin 1) (ι ⊕ Fin 1) R :=
  Matrix.fromBlocks base (fun index _ => columns index column) (fun _ index => rows row index) (fun _ _ => corner row column)

theorem border_det (base : Matrix ι ι R) [Invertible base] (columns : Matrix ι (Fin 2) R) (rows : Matrix (Fin 2) ι R)
    (corner : Matrix (Fin 2) (Fin 2) R) (row column : Fin 2) :
    (border base columns rows corner row column).det = base.det * (corner - rows * ⅟base * columns) row column := by
  rw [border, Matrix.det_fromBlocks₁₁, Matrix.det_unique (n := Fin 1)]
  rfl

/-- Two bordered minors multiply to a multiple of their common prefix determinant. -/
theorem identity (base : Matrix ι ι R) [Invertible base] (columns : Matrix ι (Fin 2) R) (rows : Matrix (Fin 2) ι R)
    (corner : Matrix (Fin 2) (Fin 2) R) :
    base.det * (Matrix.fromBlocks base columns rows corner).det =
      (border base columns rows corner 0 0).det * (border base columns rows corner 1 1).det -
        (border base columns rows corner 1 0).det * (border base columns rows corner 0 1).det := by
  rw [Matrix.det_fromBlocks₁₁, Matrix.det_fin_two]
  simp only [border_det]
  ring

theorem int_identity (base : Matrix ι ι Int) (nonzero : base.det ≠ 0)
    (columns : Matrix ι (Fin 2) Int) (rows : Matrix (Fin 2) ι Int) (corner : Matrix (Fin 2) (Fin 2) Int) :
    base.det * (Matrix.fromBlocks base columns rows corner).det =
      (border base columns rows corner 0 0).det * (border base columns rows corner 1 1).det -
        (border base columns rows corner 1 0).det * (border base columns rows corner 0 1).det := by
  let rationalBase : Matrix ι ι ℚ := base.map Int.cast
  have rationalNonzero : rationalBase.det ≠ 0 := by
    change (base.map fun value => (value : ℚ)).det ≠ 0
    rw [← Int.cast_det]
    exact_mod_cast nonzero
  letI : Invertible rationalBase := Matrix.invertibleOfIsUnitDet rationalBase (isUnit_iff_ne_zero.mpr rationalNonzero)
  have lifted := identity rationalBase (columns.map (fun value => (value : ℚ)))
    (rows.map (fun value => (value : ℚ))) (corner.map (fun value => (value : ℚ)))
  have mapBorder (row column : Fin 2) :
      border rationalBase (columns.map (fun value => (value : ℚ))) (rows.map (fun value => (value : ℚ)))
        (corner.map (fun value => (value : ℚ))) row column =
      (border base columns rows corner row column).map (fun value => (value : ℚ)) := by
    ext left right
    cases left <;> cases right <;> rfl
  simp only [mapBorder] at lifted
  have mapBlocks : Matrix.fromBlocks rationalBase (columns.map (fun value => (value : ℚ)))
      (rows.map (fun value => (value : ℚ))) (corner.map (fun value => (value : ℚ))) =
      (Matrix.fromBlocks base columns rows corner).map (fun value => (value : ℚ)) := by
    ext left right
    cases left <;> cases right <;> rfl
  rw [mapBlocks] at lifted
  change (base.map fun value => (value : ℚ)).det * _ = _ at lifted
  simp only [← Int.cast_det] at lifted
  exact_mod_cast lifted

theorem divides (base : Matrix ι ι Int) (nonzero : base.det ≠ 0)
    (columns : Matrix ι (Fin 2) Int) (rows : Matrix (Fin 2) ι Int) (corner : Matrix (Fin 2) (Fin 2) Int) :
    base.det ∣ (border base columns rows corner 0 0).det * (border base columns rows corner 1 1).det -
      (border base columns rows corner 1 0).det * (border base columns rows corner 0 1).det :=
  ⟨(Matrix.fromBlocks base columns rows corner).det, (int_identity base nonzero columns rows corner).symm⟩

end ProjectEulerSolutions.BorderedDeterminants
