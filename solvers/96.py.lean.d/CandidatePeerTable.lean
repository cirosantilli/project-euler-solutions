import solvers.«96.py.lean.d».CandidatePeerLoop

namespace solvers.«96».reduce_cands
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku
set_option Elab.async false
noncomputable section

structure PeerTable where
  root : Nat
  address : Cell → Nat
  cells : Cell → List Cell
  related : ∀ cell other, other ∈ cells cell → Peer cell other

namespace PeerTable

def references (table : PeerTable) : List Mutable.Value :=
  List.ofFn fun slot : Fin 81 => .ref (table.address (Encoding.cell slot))

structure View (table : PeerTable) (heap : Mutable.Heap) : Prop where
  root : heap[table.root]? = some (.list table.references)
  sets : ∀ cell, heap[table.address cell]? = some (.set (peerLoop.keys (table.cells cell)))

structure Separate (table : PeerTable) (address queue : Nat) : Prop where
  rootCandidates : address ≠ table.root
  rootQueue : queue ≠ table.root
  setCandidates : ∀ cell, address ≠ table.address cell
  setQueue : ∀ cell, queue ≠ table.address cell

theorem preserve (table : PeerTable) (address queue : Nat) (before after : Mutable.Heap)
    (view : table.View before) (separate : table.Separate address queue)
    (frame : peerLoop.Frame address queue before after) : table.View after :=
  ⟨(frame.outside table.root separate.rootCandidates separate.rootQueue).trans view.root,
    fun cell => (frame.outside (table.address cell) (separate.setCandidates cell) (separate.setQueue cell)).trans (view.sets cell)⟩

theorem read (table : PeerTable) (heap : Mutable.Heap) (cell : Cell) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (view : table.View heap) (getPeers : env "PEERS" = some (.ref table.root))
    (getCell : env "i" = some (number (Encoding.index cell).val)) :
    Evaluates program env (.subscript (.name "PEERS") (.name "i")) heap (.ref (table.address cell)) heap globals := by
  apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getPeers) (Evaluates.lookup getCell)
  have size : table.references.length = 81 := by simp [references]
  have selected : table.references[(Encoding.index cell).val]? = some (.ref (table.address cell)) := by
    simp only [references, List.getElem?_ofFn, dif_pos (Encoding.index cell).isLt]
    change some (Mutable.Value.ref (table.address (Encoding.cell (Encoding.index cell)))) = _
    rw [Encoding.cell_index]
  obtain ⟨_, selected⟩ := List.getElem_of_getElem? selected
  simp [Mutable.getItem, view.root, Mutable.index, size,
    not_lt.mpr (Int.natCast_nonneg (Encoding.index cell).val), not_le.mpr (Encoding.index cell).isLt,
    selected, Except.map]

end PeerTable
end
end solvers.«96».reduce_cands
