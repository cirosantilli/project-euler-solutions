import ProjectEulerSolutions.Lib.HeapCollection

/-! Shared tuple cells in a mutable list of digit suffixes and bit masks. -/
namespace ProjectEulerSolutions.MaskedStates
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

set_option Elab.async false

abbrev State := List Nat × Nat
def number (n : Nat) : Mutable.Value := .scalar (.int n)
def numbers (items : List Nat) := items.map number

structure Cell where
  state : State
  digitsAddress : Nat
  address : Nat

def reference (cell : Cell) : Mutable.Value := .ref cell.address
def references (cells : List Cell) := cells.map reference

def Stored (heap : Mutable.Heap) (cell : Cell) : Prop :=
  heap[cell.digitsAddress]? = some (.tuple (numbers cell.state.1)) ∧
    heap[cell.address]? = some (.tuple [.ref cell.digitsAddress, number cell.state.2])

/-- All retained tuples follow the collector, so appending to the list cannot overwrite them. -/
structure View (heap : Mutable.Heap) (collector : Nat) (cells : List Cell) : Prop where
  stored : ∀ cell ∈ cells, Stored heap cell
  afterDigits : ∀ cell ∈ cells, collector < cell.digitsAddress
  afterCell : ∀ cell ∈ cells, collector < cell.address

theorem Stored.append {heap : Mutable.Heap} {cell : Cell} (stored : Stored heap cell)
    (extra : Mutable.Heap) : Stored (heap ++ extra) cell :=
  ⟨HeapCollection.prefix_found heap extra _ _ stored.1,
    HeapCollection.prefix_found heap extra _ _ stored.2⟩

theorem View.append {heap : Mutable.Heap} {collector : Nat} {cells : List Cell}
    (view : View heap collector cells) (extra : Mutable.Heap) : View (heap ++ extra) collector cells :=
  ⟨fun cell member => (view.stored cell member).append extra, view.afterDigits, view.afterCell⟩

theorem View.set_collector {heap : Mutable.Heap} {collector : Nat} {cells : List Cell}
    (view : View heap collector cells) (items : List Mutable.Value) :
    View (heap.set collector (.list items)) collector cells := by
  refine ⟨?_, view.afterDigits, view.afterCell⟩
  intro cell member
  obtain ⟨digits, pair⟩ := view.stored cell member
  constructor
  · rw [List.getElem?_set_ne (by have := view.afterDigits cell member; omega)]
    exact digits
  · rw [List.getElem?_set_ne (by have := view.afterCell cell member; omega)]
    exact pair

/-- Appending a source tuple preserves the original heap prefix and all earlier tuple identities. -/
theorem append {globals : Mutable.Globals} (program : Module) (env : HeapCalls.Env)
    (base garbage extra : Mutable.Heap) (cells : List Cell) (view : View (HeapCollection.working base (references cells) garbage) base.length cells)
    (name : String) (getList : env name = some (.ref base.length))
    (digitsExpression usedExpression : Expr) (state : State) (digitsAddress : Nat)
    (digitsResult : Evaluates (globals := globals) program env digitsExpression
      (HeapCollection.working base (references cells) garbage) (.ref digitsAddress)
      (HeapCollection.working base (references cells) garbage ++ extra))
    (digitsFound : (HeapCollection.working base (references cells) garbage ++ extra)[digitsAddress]? =
      some (.tuple (numbers state.1)))
    (digitsAfter : base.length < digitsAddress)
    (usedResult : Evaluates (globals := globals) program env usedExpression
      (HeapCollection.working base (references cells) garbage ++ extra) (number state.2)
      (HeapCollection.working base (references cells) garbage ++ extra)) :
    let address := (HeapCollection.working base (references cells) garbage ++ extra).length
    let cell : Cell := ⟨state, digitsAddress, address⟩
    let finalGarbage := garbage ++ extra ++ [.tuple [.ref digitsAddress, number state.2]]
    BlockContinues (globals := globals) program env
        [.expr (.call (.attribute (.name name) "append") [.tuple [digitsExpression, usedExpression]])]
        (HeapCollection.working base (references cells) garbage) env
        (HeapCollection.working base (references (cells ++ [cell])) finalGarbage) ∧
      View (HeapCollection.working base (references (cells ++ [cell])) finalGarbage) base.length (cells ++ [cell]) := by
  dsimp only
  let current := HeapCollection.working base (references cells) garbage
  let afterDigits := current ++ extra
  let address := afterDigits.length
  let cell : Cell := ⟨state, digitsAddress, address⟩
  let item := Mutable.Object.tuple [.ref digitsAddress, number state.2]
  let beforeStore := afterDigits ++ [item]
  have pair : Evaluates (globals := globals) program env (.tuple [digitsExpression, usedExpression]) current
      (.ref address) beforeStore :=
    Evaluates.tuple (.cons digitsResult (.cons usedResult .nil)) (by simp)
  have beforeView : View beforeStore base.length cells := (view.append extra).append [item]
  have freshView : View beforeStore base.length [cell] := by
    constructor
    · intro selected member
      obtain rfl := List.mem_singleton.mp member
      exact ⟨HeapCollection.prefix_found afterDigits [item] _ _ digitsFound, by simp [beforeStore, cell, address, item]⟩
    · simpa only [List.mem_singleton, forall_eq, cell] using digitsAfter
    · intro selected member
      obtain rfl := List.mem_singleton.mp member
      simp only [cell, address, afterDigits, current, HeapCollection.working, List.length_append, List.length_cons]
      omega
  have combined : View beforeStore base.length (cells ++ [cell]) :=
    ⟨fun selected member => (List.mem_append.mp member).elim (beforeView.stored selected) (freshView.stored selected),
      fun selected member => (List.mem_append.mp member).elim (beforeView.afterDigits selected) (freshView.afterDigits selected),
      fun selected member => (List.mem_append.mp member).elim (beforeView.afterCell selected) (freshView.afterCell selected)⟩
  have storedHeap : beforeStore.set base.length (.list (references cells ++ [.ref address])) =
      HeapCollection.working base (references (cells ++ [cell])) (garbage ++ extra ++ [item]) := by
    simp [beforeStore, afterDigits, current, HeapCollection.append, HeapCollection.store, references, reference, cell]
  refine ⟨?_, ?_⟩
  · have unchanged : beforeStore[base.length]? = some (.list (references cells)) := by
      simpa only [beforeStore, afterDigits, List.append_assoc] using
        HeapCollection.prefix_found current (extra ++ [item]) base.length _
          (HeapCollection.found base (references cells) garbage)
    have executed := BlockContinues.discard (Evaluates.append (Evaluates.lookup getList)
      (HeapCollection.found base (references cells) garbage) pair unchanged)
    rw [storedHeap] at executed
    exact executed
  · have kept := combined.set_collector (references cells ++ [.ref address])
    rw [storedHeap] at kept
    exact kept

end ProjectEulerSolutions.MaskedStates
