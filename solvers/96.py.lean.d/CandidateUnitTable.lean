import solvers.«96.py.lean.d».CandidateUnitDigits

namespace solvers.«96».reduce_cands
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku
set_option Elab.async false
noncomputable section

structure UnitEntry where
  address : Nat
  cells : Fin 9 → Cell
  related : ∀ i j, i ≠ j → Peer (cells i) (cells j)

structure UnitTable where
  root : Nat
  units : List UnitEntry

namespace UnitTable

def references (units : List UnitEntry) : List Mutable.Value := units.map fun unit => .ref unit.address

structure View (table : UnitTable) (heap : Mutable.Heap) : Prop where
  root : heap[table.root]? = some (.list (references table.units))
  units : ∀ unit ∈ table.units, heap[unit.address]? = some (.list (peerLoop.encoded ((List.finRange 9).map unit.cells)))

structure Separate (table : UnitTable) (address : Nat) : Prop where
  root : address ≠ table.root
  units : ∀ unit ∈ table.units, address ≠ unit.address

theorem separation (table : UnitTable) (address : Nat) (heap : Mutable.Heap) (candidates : Candidates)
    (view : table.View heap) (found : reduce_cands.View address heap candidates) : table.Separate address := by
  constructor
  · intro same
    have objects := view.root
    rw [← same, found] at objects
    have lists : values candidates = references table.units := by simpa using objects
    cases units : table.units with
    | nil =>
        have size := congrArg List.length lists
        simp [values, references, units] at size
    | cons unit rest =>
        have head := congrArg (fun items : List Mutable.Value => items[0]?) lists
        change (values candidates)[0]? = (references table.units)[0]? at head
        have candidateHead : (values candidates)[0]? = some (number (Masks.encode (candidates (Encoding.cell 0)))) := rfl
        rw [candidateHead] at head
        simp [references, units] at head
  · intro unit member same
    have objects := view.units unit member
    rw [← same, found] at objects
    have lists : values candidates = peerLoop.encoded ((List.finRange 9).map unit.cells) := by simpa using objects
    have size := congrArg List.length lists
    simp [values, peerLoop.encoded] at size

theorem preserve (table : UnitTable) (address : Nat) (before after : Mutable.Heap)
    (view : table.View before) (separate : table.Separate address)
    (frame : ∀ index < before.length, address ≠ index → after[index]? = before[index]?) : table.View after :=
  ⟨(frame table.root (List.getElem?_eq_some_iff.mp view.root).1 separate.root).trans view.root,
    fun unit member => (frame unit.address (List.getElem?_eq_some_iff.mp (view.units unit member)).1
      (separate.units unit member)).trans (view.units unit member)⟩

end UnitTable
end
end solvers.«96».reduce_cands
