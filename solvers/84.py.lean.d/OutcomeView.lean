import solvers.«84.py.lean.d».TransitionIteration

namespace solvers.«84».outcomes
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false

def TupleView (heap : Mutable.Heap) (outcome : Outcome) (value : Mutable.Value) : Prop :=
  ∃ address, value = .ref address ∧ heap[address]? = some (.tuple outcome.items)

def View (heap : Mutable.Heap) (root : Nat) (values : List Outcome) : Prop :=
  ∃ items, heap[root]? = some (.list items) ∧ List.Forall₂ (TupleView heap) values items

theorem heap_view (base : Mutable.Heap) (values : List Outcome) : View (heap base values) base.length values := by
  refine ⟨references base values, found base values, ?_⟩
  apply List.forall₂_of_length_eq_of_get (by simp [references])
  intro index inside referenceInside
  refine ⟨base.length + 1 + index, ?_, ?_⟩
  · simp [references]
  · exact tuple_found base values index values[index] (List.getElem?_eq_getElem inside)

theorem View.frame {before after : Mutable.Heap} {source root : Nat} {values : List Outcome}
    (view : View before source values) (entries : NaturalDictionary.Entries)
    (found : NaturalDictionary.Found before root entries) (frame : transition_row.items.HeapFrame before after root) :
    View after source values := by
  obtain ⟨items, present, represented⟩ := view
  have different : source ≠ root := by
    intro same
    subst source
    change before[root]? = some (.dict (NaturalDictionary.encode entries)) at found
    rw [present] at found
    cases found
  refine ⟨items, frame _ _ different present, ?_⟩
  clear present
  induction represented with
  | nil => exact .nil
  | cons head tail ih =>
      obtain ⟨address, rfl, tupleFound⟩ := head
      exact .cons ⟨address, rfl, frame.tuples entries found address _ tupleFound⟩ ih

theorem View.at {heap : Mutable.Heap} {source : Nat} {values : List Outcome}
    (view : View heap source values) (position : Nat) (inside : position < values.length) :
    ∃ (items : List Mutable.Value) (address : Nat), heap[source]? = some (.list items) ∧ items.length = values.length ∧
      items[position]? = some (.ref address) ∧ heap[address]? = some (.tuple values[position].items) := by
  obtain ⟨items, present, represented⟩ := view
  have itemInside : position < items.length := by rw [← represented.length_eq]; exact inside
  obtain ⟨address, same, tupleFound⟩ := represented.get inside itemInside
  refine ⟨items, address, present, represented.length_eq.symm, ?_, tupleFound⟩
  rw [List.getElem?_eq_getElem itemInside]
  exact congrArg some same

end solvers.«84».outcomes
