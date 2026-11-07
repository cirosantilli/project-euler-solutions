import solvers.«84.py.lean.d».TransitionIndex
import ProjectEulerSolutions.Lib.MonopolyEncoded

namespace solvers.«84».transitions
open FormalSW.PyAst HeapCalls ProjectEulerSolutions NaturalDictionary
set_option Elab.async false

theorem View.append {sides : Nat} {heap : Mutable.Heap} {value : Mutable.Value}
    (view : View sides heap value) (extra : Mutable.Heap) : View sides (heap ++ extra) value := by
  obtain ⟨root, rows, returned, found, views⟩ := view
  exact ⟨root, rows, returned, HeapCollection.prefix_found _ _ _ _ found, views.append extra⟩

theorem row_result (sides : Nat) (source : Fin 120) (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (value : Mutable.Value) (view : View sides heap value)
    (getTransitions : env "transitions" = some value) (getSource : env "source" = some (number source.val)) :
    ∃ row : Row, Evaluates program env (.subscript (.name "transitions") (.name "source")) heap (.ref row.address) heap globals ∧
      row.View sides heap (source.val / 3, source.val % 3) := by
  obtain ⟨root, items, row, rfl, found, length, selected, rowView⟩ := view.at source.val source.isLt
  refine ⟨row, ?_, rowView⟩
  apply Evaluates.subscript (by intro a b c; simp) (Evaluates.lookup getTransitions) (Evaluates.lookup getSource)
  simp [Mutable.getItem, Mutable.index, found, number, Except.map, length,
    not_lt.mpr (Int.natCast_nonneg source.val), not_le.mpr source.isLt,
    (List.getElem?_eq_some_iff.mp selected).2]

/-- Reading a sparse row, including a missing key, returns the exact finite-state matrix entry. -/
theorem get_result (sides : Nat) (source target : Fin 120) (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (value : Mutable.Value) (view : View sides heap value)
    (getTransitions : env "transitions" = some value) (getSource : env "source" = some (number source.val))
    (getTarget : env "target" = some (number target.val)) :
    Evaluates program env
      (.call (.attribute (.subscript (.name "transitions") (.name "source")) "get") [.name "target", .constant (.int 0)])
      heap (.scalar (.int (Monopoly.encodedWeight sides source target))) heap globals := by
  obtain ⟨row, read, found, counts⟩ := row_result sides source heap env globals value view getTransitions getSource
  have count : ((lookup row.entries target.val).getD 0 : Int) = Monopoly.encodedWeight sides source target := counts target.val
  rw [← count]
  apply Evaluates.dictionary_get (by intro name same; cases same) read found
    (.cons (Evaluates.lookup getTarget) (.cons Evaluates.constant .nil))
  simp only [HeapCalls.primitive, Mutable.dictionaryGet_scalar heap row.address (encode row.entries) found, lookup_encode]
  cases lookup row.entries target.val <;> rfl

end solvers.«84».transitions
