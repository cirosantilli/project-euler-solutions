import ProjectEulerSolutions.Lib.RationalHeap
import ProjectEulerSolutions.Lib.HeapPreservesOutside

/-! Rational operand tuples survive writes to a memo dictionary and result collector. -/
namespace ProjectEulerSolutions.RationalPair.Heap
open FormalSW.PyAst
set_option Elab.async false

theorem Row.preserved_outside {memo collector : Nat} {before after : Mutable.Heap} {item : Mutable.Value} {pair : Pair}
    (row : Row before item pair) (frame : PreservesOutside [memo, collector] before after)
    (dictionary : List (HashDict.Entry Mutable.Value)) (keys : List (HashKey × Mutable.Value))
    (memoFound : (before[memo]?).bind Mutable.dictionaryEntries = some dictionary)
    (setFound : (before[collector]?).bind Mutable.keyEntries = some keys) : Row after item pair := by
  obtain ⟨address, rfl, found⟩ := row
  exact ⟨address, rfl, frame.ordinary dictionary keys memoFound setFound address _ found rfl rfl⟩

theorem Rows.preserved_outside {memo collector : Nat} {before after : Mutable.Heap} {items : List Mutable.Value} {pairs : List Pair}
    (rows : Rows before items pairs) (frame : PreservesOutside [memo, collector] before after)
    (dictionary : List (HashDict.Entry Mutable.Value)) (keys : List (HashKey × Mutable.Value))
    (memoFound : (before[memo]?).bind Mutable.dictionaryEntries = some dictionary)
    (setFound : (before[collector]?).bind Mutable.keyEntries = some keys) : Rows after items pairs :=
  rows.imp (fun _ _ row => row.preserved_outside frame dictionary keys memoFound setFound)

end ProjectEulerSolutions.RationalPair.Heap
