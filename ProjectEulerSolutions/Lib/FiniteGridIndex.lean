import ProjectEulerSolutions.Lib.FiniteGridGraph

/-! Row-major array indices for finite rectangular grids. -/
namespace ProjectEulerSolutions.FiniteGridIndex
open FiniteGridGraph

def index {height width : Nat} (v : Vertex height width) : Nat := v.1.val * width + v.2.val

theorem index_lt {height width : Nat} (v : Vertex height width) : index v < height * width := by
  calc
    index v < v.1.val * width + width := Nat.add_lt_add_left v.2.isLt _
    _ = (v.1.val + 1) * width := by simp [Nat.add_mul]
    _ ≤ height * width := Nat.mul_le_mul_right width (Nat.succ_le_of_lt v.1.isLt)

theorem index_div {height width : Nat} (v : Vertex height width) : index v / width = v.1.val := by
  have positive : 0 < width := Nat.zero_lt_of_lt v.2.isLt
  simp [index, Nat.add_div, Nat.div_eq_of_lt v.2.isLt, positive, Nat.mod_lt _ positive]

theorem index_mod {height width : Nat} (v : Vertex height width) : index v % width = v.2.val := by
  simp [index, Nat.add_mod, Nat.mod_eq_of_lt v.2.isLt]

theorem index_injective (height width : Nat) :
    Function.Injective (@index height width) := by
  intro v u equal
  apply Prod.ext <;> apply Fin.ext
  · simpa only [index_div] using congrArg (fun i => i / width) equal
  · simpa only [index_mod] using congrArg (fun i => i % width) equal

def fromIndex {height width : Nat} (i : Nat) (inside : i < height * width) : Vertex height width :=
  let positive : 0 < width := Nat.pos_of_ne_zero (by intro zero; simp [zero] at inside)
  (⟨i / width, (Nat.div_lt_iff_lt_mul positive).mpr inside⟩, ⟨i % width, Nat.mod_lt i positive⟩)

theorem index_fromIndex {height width : Nat} (i : Nat) (inside : i < height * width) :
    index (fromIndex i inside) = i := by
  exact Nat.div_add_mod' i width

theorem fromIndex_index {height width : Nat} (v : Vertex height width) :
    fromIndex (index v) (index_lt v) = v := by
  apply index_injective height width
  exact index_fromIndex _ _

theorem encode_fromIndex {height width : Nat} (i : Nat) (inside : i < height * width) :
    encode (fromIndex i inside) = (((i / width : Nat) : Int), ((i % width : Nat) : Int)) := rfl

theorem index_of_encode {height width : Nat} (v : Vertex height width) :
    (encode v).1.toNat * width + (encode v).2.toNat = index v := by
  simp [encode, index]

end ProjectEulerSolutions.FiniteGridIndex
