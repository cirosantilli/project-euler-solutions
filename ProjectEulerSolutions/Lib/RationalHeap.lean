import ProjectEulerSolutions.Lib.RationalPair
import ProjectEulerSolutions.Lib.HeapCollection
import Mathlib.Data.List.Forall2

/-! Heap representations of rational pairs and lists of rational results. -/
namespace ProjectEulerSolutions.RationalPair.Heap
open FormalSW.PyAst
set_option Elab.async false

abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
abbrev object (pair : Pair) : Mutable.Object := .tuple [number pair.1, number pair.2]

def Row (heap : Mutable.Heap) (value : Mutable.Value) (pair : Pair) : Prop :=
  ∃ address, value = .ref address ∧ heap[address]? = some (object pair)

abbrev Rows (heap : Mutable.Heap) (values : List Mutable.Value) (pairs : List Pair) :=
  List.Forall₂ (Row heap) values pairs

def ListFound (heap : Mutable.Heap) (address : Nat) (pairs : List Pair) : Prop :=
  ∃ values, heap[address]? = some (.list values) ∧ Rows heap values pairs

theorem Row.append {heap : Mutable.Heap} {value : Mutable.Value} {pair : Pair}
    (found : Row heap value pair) (extra : Mutable.Heap) : Row (heap ++ extra) value pair := by
  obtain ⟨address, rfl, present⟩ := found
  exact ⟨address, rfl, ProjectEulerSolutions.HeapCollection.prefix_found heap extra address _ present⟩

theorem Rows.append {heap : Mutable.Heap} {values : List Mutable.Value} {pairs : List Pair}
    (found : Rows heap values pairs) (extra : Mutable.Heap) : Rows (heap ++ extra) values pairs :=
  found.imp (fun _ _ row => row.append extra)

theorem ListFound.append {heap : Mutable.Heap} {address : Nat} {pairs : List Pair}
    (found : ListFound heap address pairs) (extra : Mutable.Heap) : ListFound (heap ++ extra) address pairs := by
  obtain ⟨values, present, rows⟩ := found
  exact ⟨values, ProjectEulerSolutions.HeapCollection.prefix_found heap extra address _ present, rows.append extra⟩

/-- A rational tuple cannot alias the list cell used to collect results. -/
theorem Row.collector {base : Mutable.Heap} {old : List Mutable.Value} {garbage : Mutable.Heap}
    {value : Mutable.Value} {pair : Pair}
    (row : Row (ProjectEulerSolutions.HeapCollection.working base old garbage) value pair)
    (replacement : List Mutable.Value) (extra : Mutable.Heap) :
    Row (ProjectEulerSolutions.HeapCollection.working base replacement (garbage ++ extra)) value pair := by
  obtain ⟨address, rfl, present⟩ := row
  have different : base.length ≠ address := by
    rintro rfl
    simp [ProjectEulerSolutions.HeapCollection.found] at present
  exact ⟨address, rfl, ProjectEulerSolutions.HeapCollection.other_found
    base old replacement garbage extra address (object pair) different present⟩

theorem Rows.collector {base : Mutable.Heap} {old : List Mutable.Value} {garbage : Mutable.Heap}
    {values : List Mutable.Value} {pairs : List Pair}
    (rows : Rows (ProjectEulerSolutions.HeapCollection.working base old garbage) values pairs)
    (replacement : List Mutable.Value) (extra : Mutable.Heap) :
    Rows (ProjectEulerSolutions.HeapCollection.working base replacement (garbage ++ extra)) values pairs :=
  rows.imp (fun _ _ row => row.collector replacement extra)

/-- Recursive result lists survive writes to a distinct collector. -/
theorem ListFound.collector {base : Mutable.Heap} {old : List Mutable.Value} {garbage : Mutable.Heap}
    {address : Nat} {pairs : List Pair}
    (list : ListFound (ProjectEulerSolutions.HeapCollection.working base old garbage) address pairs)
    (different : base.length ≠ address) (replacement : List Mutable.Value) (extra : Mutable.Heap) :
    ListFound (ProjectEulerSolutions.HeapCollection.working base replacement (garbage ++ extra)) address pairs := by
  obtain ⟨values, present, rows⟩ := list
  exact ⟨values, ProjectEulerSolutions.HeapCollection.other_found
    base old replacement garbage extra address (.list values) different present, rows.collector replacement extra⟩

end ProjectEulerSolutions.RationalPair.Heap
