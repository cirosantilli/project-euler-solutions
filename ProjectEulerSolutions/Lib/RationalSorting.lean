import ProjectEulerSolutions.Lib.RationalHeap
import ProjectEulerSolutions.Lib.TupleSorting
import Mathlib.Data.List.Perm.Basic

/-! Sorting rational tuples orders their integer components and retains their shared references. -/
namespace ProjectEulerSolutions.RationalPair.Heap
open FormalSW.PyAst
set_option Elab.async false

theorem Rows.integer_tuples {heap : Mutable.Heap} {values : List Mutable.Value} {pairs : List Pair}
    (rows : Rows heap values pairs) :
    ∀ item ∈ values, ∃ address : Nat, ∃ xs : List Int,
      item = .ref address ∧ heap[address]? = some (.tuple (xs.map fun n => .scalar (.int n))) := by
  induction rows with
  | nil => simp
  | @cons item pair items pairs row rows ih =>
    intro candidate member
    rcases List.mem_cons.mp member with rfl | later
    · obtain ⟨address, rfl, found⟩ := row
      exact ⟨address, [pair.1, pair.2], rfl, found⟩
    · exact ih candidate later

/-- Tuple-key sorting succeeds for every rational-pair list, including duplicate and signed components. -/
theorem Rows.sort {heap : Mutable.Heap} {values : List Mutable.Value} {pairs : List Pair}
    (rows : Rows heap values pairs) :
    ∃ sortedValues sortedPairs,
      HeapSort.sort heap values = some (.ok sortedValues) ∧
      Rows heap sortedValues sortedPairs ∧ sortedPairs.Perm pairs ∧ sortedValues.Perm values := by
  have sorted := TupleSorting.sortWithReverse heap false values rows.integer_tuples
  have permutation := TupleSorting.sorted_perm heap false values
  obtain ⟨sortedPairs, sortedRows, pairPermutation⟩ := List.perm_comp_forall₂ permutation rows
  exact ⟨TupleSorting.sorted heap false values, sortedPairs, sorted, sortedRows, pairPermutation, permutation⟩

end ProjectEulerSolutions.RationalPair.Heap
