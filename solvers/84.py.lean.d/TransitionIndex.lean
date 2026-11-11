import solvers.«84.py.lean.d».TransitionBuild

namespace solvers.«84».transitions
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false

theorem state_blocks_length (count first : Nat) :
    ((List.range' first count).flatMap positionStates).length = count * 3 := by
  induction count generalizing first with
  | zero => simp
  | succ count ih =>
      simp only [List.range'_succ, List.flatMap_cons, List.length_append, ih]
      simp [positionStates]
      omega

theorem state_blocks_at (count first index : Nat) (bound : index < count * 3) :
    ((List.range' first count).flatMap positionStates)[index]? = some (first + index / 3, index % 3) := by
  induction count generalizing first index with
  | zero => omega
  | succ count ih =>
      simp only [List.range'_succ, List.flatMap_cons]
      by_cases small : index < 3
      · rw [List.getElem?_append_left (by simpa [positionStates] using small)]
        have quotient : index / 3 = 0 := Nat.div_eq_of_lt small
        simp [positionStates, small, quotient, Nat.mod_eq_of_lt small]
      · rw [List.getElem?_append_right (by simp [positionStates]; omega)]
        simp only [positionStates, List.length_map, List.length_range']
        rw [ih (first + 1) (index - 3) (by omega)]
        congr 2 <;> omega

theorem allStates_length : allStates.length = 120 := state_blocks_length 40 0

/-- Row indices are exactly the Python encoding `3 * position + doubles`. -/
theorem allStates_at (index : Nat) (bound : index < 120) : allStates[index]? = some (index / 3, index % 3) := by
  simpa only [Nat.zero_add] using state_blocks_at 40 0 index bound

theorem View.at {sides : Nat} {heap : Mutable.Heap} {value : Mutable.Value}
    (view : View sides heap value) (index : Nat) (bound : index < 120) :
    ∃ (root : Nat) (items : List Mutable.Value) (row : Row),
      value = .ref root ∧ heap[root]? = some (.list items) ∧ items.length = 120 ∧
      items[index]? = some (.ref row.address) ∧ row.View sides heap (index / 3, index % 3) := by
  obtain ⟨root, rows, returned, present, views⟩ := view
  have length : rows.length = 120 := views.length_eq.symm.trans allStates_length
  have inside : index < rows.length := by omega
  have related := views.get (i := index) (by rw [allStates_length]; exact bound) inside
  simp only [List.get_eq_getElem] at related
  have selected := (List.getElem?_eq_some_iff.mp (allStates_at index bound)).2
  rw [selected] at related
  exact ⟨root, references rows, rows[index], returned, present,
    by simp [references, length], by simp [references, List.getElem?_eq_getElem inside], related⟩

end solvers.«84».transitions
