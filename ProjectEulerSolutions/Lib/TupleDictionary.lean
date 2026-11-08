import ProjectEulerSolutions.Lib.TuplePool
import lang.py.FormalSW.KeyDictionaryCalls

/-! Source dictionary displays retain cached tuple keys and Python's insertion order. -/
namespace ProjectEulerSolutions.TupleDictionary
open FormalSW.PyAst
set_option Elab.async false

structure Row where
  address : Nat
  keyAddress : Nat
  key : HashKey
  value : Mutable.Value

def Row.View (row : Row) (heap : Mutable.Heap) : Prop :=
  heap[row.address]? = some (.tuple [.ref row.keyAddress, row.value]) ∧
    TuplePool.View heap row.key (.ref row.keyAddress)

def Row.entry (row : Row) : HashDict.Entry Mutable.Value := (row.key, .ref row.keyAddress, row.value)
def insert (entries : List (HashDict.Entry Mutable.Value)) (row : Row) : List (HashDict.Entry Mutable.Value) :=
  HashDict.insert entries row.key (.ref row.keyAddress) row.value

theorem Row.View.frame {row : Row} {before after : Mutable.Heap} (view : row.View before)
    (kept : ∀ (address : Nat) (items : List Mutable.Value),
      before[address]? = some (.tuple items) → after[address]? = some (.tuple items)) : row.View after :=
  ⟨kept row.address _ view.1, TuplePool.View.frame before after kept row.key _ view.2⟩

theorem Row.View.append {row : Row} {heap : Mutable.Heap} (view : row.View heap) (extra : Mutable.Heap) : row.View (heap ++ extra) :=
  view.frame (fun _ _ found => (List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1).trans found)

theorem tuple_frame (heap : Mutable.Heap) (root : Nat) (entries replacement : List (HashDict.Entry Mutable.Value))
    (found : heap[root]? = some (.keyDict entries)) (address : Nat) (items : List Mutable.Value)
    (tupleFound : heap[address]? = some (.tuple items)) :
    (heap.set root (.keyDict replacement))[address]? = some (.tuple items) := by
  have different : root ≠ address := by
    intro same
    rw [same, tupleFound] at found
    cases found
  exact (List.getElem?_set_ne different).trans tupleFound

theorem insert_row (heap : Mutable.Heap) (root : Nat) (entries : List (HashDict.Entry Mutable.Value))
    (found : heap[root]? = some (.keyDict entries)) (row : Row) (view : row.View heap) :
    DictionaryConstructor.insertRow root (.ref row.address) heap =
      some (.ok (), heap.set root (.keyDict (insert entries row))) := by
  simp [DictionaryConstructor.insertRow, DictionaryConstructor.iterable, view.1, Mutable.sequence,
    Mutable.setItem, found, view.2.hash, insert, Except.map]

theorem insert_first (heap : Mutable.Heap) (root : Nat) (found : heap[root]? = some (.dict []))
    (row : Row) (view : row.View heap) :
    DictionaryConstructor.insertRow root (.ref row.address) heap =
      some (.ok (), heap.set root (.keyDict [row.entry])) := by
  simp [DictionaryConstructor.insertRow, DictionaryConstructor.iterable, view.1, Mutable.sequence,
    Mutable.setItem, found, view.2.hash, Mutable.promoteDictionary, HashDict.insert, Row.entry, Except.map]

/-- Duplicate keys replace only the value, retaining their first reference and position. -/
theorem insert_rows (rows : List Row) (heap : Mutable.Heap) (root : Nat) (entries : List (HashDict.Entry Mutable.Value))
    (found : heap[root]? = some (.keyDict entries)) (views : ∀ row ∈ rows, row.View heap) :
    DictionaryConstructor.insertRows root (rows.map fun row => .ref row.address) heap =
      some (.ok (), heap.set root (.keyDict (rows.foldl insert entries))) := by
  induction rows generalizing heap entries with
  | nil =>
      obtain ⟨inside, same⟩ := List.getElem?_eq_some_iff.mp found
      simp only [List.map_nil, List.foldl_nil, DictionaryConstructor.insertRows]
      rw [← same, List.set_getElem_self inside]
      rfl
  | cons row rows ih =>
      have first := insert_row heap root entries found row (views row (by simp))
      have rest := ih (heap.set root (.keyDict (insert entries row))) (insert entries row)
        (List.getElem?_set_self (List.getElem?_eq_some_iff.mp found).1)
        (fun current member => (views current (by simp [member])).frame (tuple_frame heap root entries _ found))
      simp [List.map_cons, List.foldl_cons, DictionaryConstructor.insertRows, first, rest]

/-- The runtime constructor reads the supplied pair objects and allocates just its result dictionary. -/
theorem construct (heap : Mutable.Heap) (source : Nat) (head : Row) (tail : List Row)
    (found : heap[source]? = some (.list ((head :: tail).map fun row => .ref row.address)))
    (views : ∀ row ∈ head :: tail, row.View heap) :
    DictionaryConstructor.construct [.ref source] heap =
      some (.ok (.ref heap.length), heap ++ [.keyDict (tail.foldl insert [head.entry])]) := by
  have first := insert_first (heap ++ [.dict []]) heap.length (by simp) head ((views head (by simp)).append _)
  have remaining := insert_rows tail (heap ++ [.keyDict [head.entry]]) heap.length [head.entry] (by simp)
    (fun row member => (views row (by simp [member])).append _)
  have sequence := (List.getElem?_append_left (l₂ := [Mutable.Object.dict []]) (List.getElem?_eq_some_iff.mp found).1).trans found
  simp [DictionaryConstructor.construct, found, DictionaryConstructor.fromIterable, Mutable.allocateDict,
    DictionaryConstructor.iterable, sequence, Mutable.sequence, DictionaryConstructor.insertRows, first, remaining, Except.map]

end ProjectEulerSolutions.TupleDictionary
