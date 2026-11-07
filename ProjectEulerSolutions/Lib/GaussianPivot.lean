import Mathlib.LinearAlgebra.Matrix.NonsingularInverse

/-! Forward elimination always finds a pivot in a nonsingular matrix. -/
namespace ProjectEulerSolutions.GaussianPivot
open scoped BigOperators
set_option Elab.async false

private theorem fixed_prefix {n : Nat} (permutation : Equiv.Perm (Fin n)) (count : Nat)
    (bounded : ∀ index : Fin n, index.val < count → (permutation index).val ≤ index.val)
    (index : Fin n) (inside : index.val < count) : permutation index = index := by
  by_cases same : permutation index = index
  · exact same
  · have smaller : (permutation index).val < index.val := by
      have le := bounded index inside
      have different : (permutation index).val ≠ index.val := by intro equal; exact same (Fin.ext equal)
      omega
    exact permutation.injective (fixed_prefix permutation count bounded (permutation index) (by omega))
termination_by index.val

/-- Earlier columns are zero below their diagonals; the next active column must contain a nonzero entry. -/
theorem exists_pivot {n : Nat} (matrix : Matrix (Fin n) (Fin n) ℚ) (nonsingular : matrix.det ≠ 0)
    (column : Fin n)
    (eliminated : ∀ row previous : Fin n, previous.val < column.val → previous.val < row.val → matrix row previous = 0) :
    ∃ row : Fin n, column.val ≤ row.val ∧ matrix row column ≠ 0 := by
  classical
  rw [Matrix.det_apply'] at nonsingular
  obtain ⟨permutation, _, nonzero⟩ := Finset.exists_ne_zero_of_sum_ne_zero nonsingular
  have product : (∏ index, matrix (permutation index) index) ≠ 0 := (mul_ne_zero_iff.mp nonzero).2
  have entries : ∀ index, matrix (permutation index) index ≠ 0 := by
    intro index
    exact (Finset.prod_ne_zero_iff.mp product) index (Finset.mem_univ index)
  have fixed : ∀ index : Fin n, index.val < column.val → permutation index = index :=
    fixed_prefix permutation column.val (by
      intro index inside
      by_contra wrong
      exact entries index (eliminated (permutation index) index inside (by omega)))
  refine ⟨permutation column, ?_, entries column⟩
  by_contra wrong
  have earlier := fixed (permutation column) (by omega)
  have same := permutation.injective earlier
  have equal := congrArg Fin.val same
  omega

theorem exists_pivot_of_injective {n : Nat} (matrix : Matrix (Fin n) (Fin n) ℚ)
    (injective : Function.Injective matrix.mulVec) (column : Fin n)
    (eliminated : ∀ row previous : Fin n, previous.val < column.val → previous.val < row.val → matrix row previous = 0) :
    ∃ row : Fin n, column.val ≤ row.val ∧ matrix row column ≠ 0 :=
  exists_pivot matrix ((Matrix.isUnit_iff_isUnit_det matrix).mp (Matrix.mulVec_injective_iff_isUnit.mp injective)).ne_zero
    column eliminated

end ProjectEulerSolutions.GaussianPivot
