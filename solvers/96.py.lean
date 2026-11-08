import solvers.«96.py.lean.d».SolveResult
import data.«project-euler-statements».data.py.project_euler_statements.«p96.py.lean.d».Naive

namespace solvers.«96».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.Sudoku reduce_cands
set_option Elab.async false
noncomputable section

theorem «Returns» (text : String) (solutions : List Board) (input : Sudoku.Text.Input text solutions)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (peers : PeerTable) (units : UnitTable) (complete : units.Complete)
    (peerView : peers.View heap) (unitView : units.View heap) (getAll : globals.lookup "ALL" = some (number 511))
    (getPeers : globals.lookup "PEERS" = some (.ref peers.root)) (getUnits : globals.lookup "UNITS" = some (.ref units.root)) :
    (SetCalls.call program "solve" [.scalar (.string text)] heap globals).Returns := by
  intro choices
  obtain ⟨extra, _, executed⟩ := result choices text solutions input heap globals peers units complete peerView unitView
    getAll getPeers getUnits 0 heap.length []
  exact ⟨number ((solutions.map Sudoku.Text.score).sum), globals, heap ++ extra, heap.length, executed.finishes (.done rfl)⟩

theorem «Supported» (text : String) (solutions : List Board) (input : Sudoku.Text.Input text solutions)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (peers : PeerTable) (units : UnitTable) (complete : units.Complete)
    (peerView : peers.View heap) (unitView : units.View heap) (getAll : globals.lookup "ALL" = some (number 511))
    (getPeers : globals.lookup "PEERS" = some (.ref peers.root)) (getUnits : globals.lookup "UNITS" = some (.ref units.root)) :
    (SetCalls.call program "solve" [.scalar (.string text)] heap globals).Supported := by
  intro choices
  exact («Returns» text solutions input heap globals peers units complete peerView unitView getAll getPeers getUnits choices).supported

/-- Every optimized set schedule agrees with the complete reference, preserving caller aliases, cycles, and globals. -/
theorem «Equiv(naive)» (text : String) (solutions : List Board) (input : Sudoku.Text.Input text solutions)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (valid : heap.Valid)
    (globalsWithin : ∀ entry ∈ globals, entry.2.edge.Within heap.length)
    (peers : PeerTable) (units : UnitTable) (complete : units.Complete)
    (peerView : peers.View heap) (unitView : units.View heap) (getAll : globals.lookup "ALL" = some (number 511))
    (getPeers : globals.lookup "PEERS" = some (.ref peers.root)) (getUnits : globals.lookup "UNITS" = some (.ref units.root)) :
    (SetCalls.call program "solve" [.scalar (.string text)] heap globals).Equiv
      (SetCalls.call P96Ast.module "naive" [.scalar (.string text)] heap globals) := by
  intro leftChoices rightChoices
  obtain ⟨leftExtra, _, optimized⟩ := result leftChoices text solutions input heap globals peers units complete peerView unitView
    getAll getPeers getUnits 0 heap.length []
  obtain ⟨rightExtra, reference⟩ := data.«project-euler-statements».data.py.project_euler_statements.p96.naive.result
    text solutions input heap globals
  apply Mutable.Call.equiv_of_finishes
    (a := (SetCalls.call program "solve" [.scalar (.string text)] heap globals).run leftChoices)
    (b := (SetCalls.call P96Ast.module "naive" [.scalar (.string text)] heap globals).run rightChoices)
    (optimized.finishes (.done rfl)) (SetCalls.finishes_of_heap reference.finishes rightChoices 0)
  exact Mutable.Completion.equivalent_allocations heap leftExtra rightExtra
    (number ((solutions.map Sudoku.Text.score).sum)) globals valid trivial globalsWithin

/-- Starting from the actual initialized module needs no supplied peer or unit tables. -/
theorem initialized_returns (text : String) (solutions : List Board) (input : Sudoku.Text.Input text solutions)
    (caller : Mutable.Heap) :
    (SetCalls.call program "solve" [.scalar (.string text)] (initialization.context caller).heap
      (initialization.globals caller)).Returns :=
  «Returns» text solutions input (initialization.context caller).heap (initialization.globals caller)
    (initialization.context caller).peers (build_peers_units.table caller.length) (initialization.complete caller)
    (initialization.context caller).peerView (initialization.context caller).unitView
    (initialization.get_all caller) (initialization.get_peers caller) (initialization.get_units caller)

theorem initialized_equivalence (text : String) (solutions : List Board) (input : Sudoku.Text.Input text solutions)
    (caller : Mutable.Heap) (valid : (initialization.context caller).heap.Valid) :
    (SetCalls.call program "solve" [.scalar (.string text)] (initialization.context caller).heap
      (initialization.globals caller)).Equiv
    (SetCalls.call P96Ast.module "naive" [.scalar (.string text)] (initialization.context caller).heap
      (initialization.globals caller)) := by
  apply «Equiv(naive)» text solutions input (initialization.context caller).heap (initialization.globals caller) valid
    _ (initialization.context caller).peers (build_peers_units.table caller.length) (initialization.complete caller)
    (initialization.context caller).peerView (initialization.context caller).unitView
    (initialization.get_all caller) (initialization.get_peers caller) (initialization.get_units caller)
  have peers := (List.getElem?_eq_some_iff.mp (initialization.context caller).peerView.root).1
  have units := (List.getElem?_eq_some_iff.mp (initialization.context caller).unitView.root).1
  simp only [initialization.globals, List.mem_cons, List.not_mem_nil, or_false]
  intro entry member
  rcases member with rfl | rfl | rfl
  · exact units
  · exact peers
  · trivial

end
end solvers.«96».solve
