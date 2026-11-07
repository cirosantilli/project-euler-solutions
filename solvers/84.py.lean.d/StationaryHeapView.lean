import solvers.«84.py.lean.d».StationaryInvariant
import ProjectEulerSolutions.Lib.MutableIntegerMatrix

namespace solvers.«84».stationary_weights
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false

def rowAddresses (base : Mutable.Heap) (row : Fin 120) : Nat :=
  base.length + 3 * row.val + 3 + if row = 119 then 1 else 0

theorem rowAddresses_distinct (base : Mutable.Heap) : Function.Injective (rowAddresses base) := by
  intro first second same
  apply Fin.ext
  unfold rowAddresses at same
  split_ifs at same <;> omega

theorem rowAddresses_after (base : Mutable.Heap) (row : Fin 120) : base.length < rowAddresses base row := by
  unfold rowAddresses
  split_ifs <;> omega

theorem prepared_references (base : Mutable.Heap) (sides : Nat) :
    rowReferences base 119 ++ [.ref ((matrixHeap base sides 119).length + 3)] =
      List.ofFn (fun row => Mutable.Value.ref (rowAddresses base row)) := by
  apply List.ext_getElem
  · simp only [List.length_append, rowReferences, List.length_map, List.length_range, List.length_singleton, List.length_ofFn]
  · intro index left right
    have inside : index < 120 := by simpa only [List.length_ofFn] using right
    rw [List.getElem_ofFn]
    by_cases last : index = 119
    · subst index
      simp [rowReferences, rowAddresses, matrix_length]
    · have below : index < 119 := by omega
      have different : (⟨index, inside⟩ : Fin 120) ≠ 119 := by intro equal; exact last (congrArg Fin.val equal)
      rw [List.getElem_append_left (show index < (rowReferences base 119).length by simpa [rowReferences] using below)]
      simp [rowReferences, rowAddresses, different]

theorem augmentedRow_ofFn (sides : Nat) (positive : 0 < sides) (row : Fin 120) :
    TriangleBufferCalls.encode (augmentedRow sides row) =
      List.ofFn (fun column => Mutable.Value.scalar (.int ((Monopoly.initialElimination sides positive).matrix row column))) := by
  apply List.ext_getElem
  · simp only [TriangleBufferCalls.encode, List.length_map, augmentedRow_length, List.length_ofFn]
  · intro index left right
    have inside : index < 121 := by simpa only [List.length_ofFn] using right
    have read := initial_entry sides positive row ⟨index, inside⟩
    have entry := (List.getElem?_eq_some_iff.mp read).2
    simpa only [TriangleBufferCalls.encode, List.getElem_map, List.getElem_ofFn] using
      congrArg (fun value => Mutable.Value.scalar (.int value)) entry

/-- The source allocates distinct private rows, so subsequent writes cannot change another matrix row. -/
theorem prepared_mutable_view (base : Mutable.Heap) (sides : Nat) (positive : 0 < sides) :
    MutableIntegerMatrix.View (Monopoly.initialElimination sides positive).matrix
      (preparedHeap base sides) base.length (rowAddresses base) := by
  refine ⟨?_, ?_, rowAddresses_distinct base, fun row => Nat.ne_of_gt (rowAddresses_after base row)⟩
  · rw [← prepared_references base sides]
    exact prepared_found base sides
  · intro row
    rw [← augmentedRow_ofFn sides positive row]
    by_cases last : row = 119
    · subst row
      rw [augmentedRow_normalization]
      simpa [rowAddresses, matrix_length] using prepared_normalization_found base sides
    · have below : row.val < 119 := by
        have different : row.val ≠ 119 := by intro equal; exact last (Fin.ext equal)
        have inside := row.isLt
        omega
      rw [augmentedRow_ordinary sides row last]
      simpa [rowAddresses, last] using prepared_row_found base sides row.val below

end solvers.«84».stationary_weights
