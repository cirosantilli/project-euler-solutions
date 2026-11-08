import solvers.«84.py.lean.d».LandingModel
import ProjectEulerSolutions.Lib.MemoCollection

namespace solvers.«84».landings
open FormalSW.PyAst ProjectEulerSolutions resolve_landing
set_option Elab.async false

structure Dictionary where
  address : Nat
  entries : List weights.Entry

def Dictionary.View (dictionary : Dictionary) (heap : Mutable.Heap) (square : Nat) : Prop :=
  heap[dictionary.address]? = some (.keyDict (weights.encode dictionary.entries)) ∧
    weights.Valid heap dictionary.entries ∧ weights.Distinct dictionary.entries ∧ weights.Positive dictionary.entries ∧
    ∀ landing, (weights.lookup dictionary.entries landing).getD 0 = weights.total (modelData square) landing

theorem Dictionary.View.frame {dictionary : Dictionary} {before after : Mutable.Heap} {square : Nat}
    (view : dictionary.View before square)
    (tuples : ∀ (address : Nat) (items : List Mutable.Value), before[address]? = some (.tuple items) → after[address]? = some (.tuple items))
    (dicts : ∀ (address : Nat) (entries : List (HashDict.Entry Mutable.Value)), before[address]? = some (.keyDict entries) → after[address]? = some (.keyDict entries)) :
    dictionary.View after square := by
  refine ⟨dicts _ _ view.1, ?_, view.2.2⟩
  intro entry member
  exact TuplePool.View.frame before after tuples _ _ (view.2.1 entry member)

theorem Dictionary.View.append {dictionary : Dictionary} {heap : Mutable.Heap} {square : Nat}
    (view : dictionary.View heap square) (extra : Mutable.Heap) : dictionary.View (heap ++ extra) square :=
  view.frame (fun _ _ found => HeapCollection.prefix_found _ _ _ _ found)
    (fun _ _ found => HeapCollection.prefix_found _ _ _ _ found)

def references (dictionaries : List Dictionary) : List Mutable.Value :=
  dictionaries.map fun dictionary => .ref dictionary.address

def Views (heap : Mutable.Heap) (squares : List Nat) (dictionaries : List Dictionary) : Prop :=
  List.Forall₂ (fun square dictionary => dictionary.View heap square) squares dictionaries

theorem Views.frame {before after : Mutable.Heap} {squares : List Nat} {dictionaries : List Dictionary}
    (views : Views before squares dictionaries)
    (tuples : ∀ (address : Nat) (items : List Mutable.Value), before[address]? = some (.tuple items) → after[address]? = some (.tuple items))
    (dicts : ∀ (address : Nat) (entries : List (HashDict.Entry Mutable.Value)), before[address]? = some (.keyDict entries) → after[address]? = some (.keyDict entries)) :
    Views after squares dictionaries := by
  induction views with
  | nil => exact .nil
  | cons head tail ih => exact .cons (head.frame tuples dicts) ih

theorem Views.append {heap : Mutable.Heap} {squares : List Nat} {dictionaries : List Dictionary}
    (views : Views heap squares dictionaries) (extra : Mutable.Heap) : Views (heap ++ extra) squares dictionaries :=
  views.frame (fun _ _ found => HeapCollection.prefix_found _ _ _ _ found)
    (fun _ _ found => HeapCollection.prefix_found _ _ _ _ found)

theorem Views.store {heap : Mutable.Heap} {squares : List Nat} {dictionaries : List Dictionary}
    (views : Views heap squares dictionaries) (root : Nat) (old replacement : List Mutable.Value)
    (found : heap[root]? = some (.list old)) :
    Views (heap.set root (.list replacement)) squares dictionaries :=
  views.frame (MemoCollection.tuple_frame heap root old replacement found)
    (MemoCollection.dictionary_frame heap root old replacement found)

theorem Views.snoc {heap : Mutable.Heap} {squares : List Nat} {dictionaries : List Dictionary}
    (views : Views heap squares dictionaries) (square : Nat) (dictionary : Dictionary)
    (view : dictionary.View heap square) : Views heap (squares ++ [square]) (dictionaries ++ [dictionary]) := by
  induction views with
  | nil => exact .cons view .nil
  | cons head tail ih => exact .cons head ih

theorem dictionary_result (square : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals) :
    HeapMemoCalls.ReturnsRel program pool.spec "resolve_landing" [.scalar (.int square)] heap
      (fun value after => ∃ (dictionary : Dictionary) (extra : Mutable.Heap), value = .ref dictionary.address ∧ after = heap ++ extra ∧ dictionary.View after square) globals := by
  apply (resolve_landing.result square heap globals).consequence
  rintro value after ⟨root, entries, extra, returned, same, _, found, valid, distinct, positive, counts⟩
  exact ⟨⟨root, entries⟩, extra, returned, same, found, valid, distinct, positive, counts⟩

def Post (base : Mutable.Heap) (squares : List Nat) (value : Mutable.Value) (after : Mutable.Heap) : Prop :=
  ∃ dictionaries garbage, value = .ref base.length ∧
    after = HeapCollection.working base (references dictionaries) garbage ∧ Views after squares dictionaries

end solvers.«84».landings
