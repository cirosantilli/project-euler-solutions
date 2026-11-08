import solvers.«96.py.lean.d».SolveLoop
import solvers.«96.py.lean.d».Initialization

namespace solvers.«96».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.Sudoku reduce_cands
set_option Elab.async false
noncomputable section

/-- The full optimized call terminates with the sum of the unique completions under every set schedule. -/
theorem result (choices : ChoiceSchedule) (text : String) (solutions : List Board)
    (input : Sudoku.Text.Input text solutions) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (peers : PeerTable) (units : UnitTable) (complete : units.Complete)
    (peerView : peers.View heap) (unitView : units.View heap)
    (getAll : globals.lookup "ALL" = some (number 511))
    (getPeers : globals.lookup "PEERS" = some (.ref peers.root)) (getUnits : globals.lookup "UNITS" = some (.ref units.root))
    (used anchors : Nat) (stack : List HeapCalls.Frame) :
    ∃ extra finalUsed, SetCalls.Reaches choices program anchors
      ⟨⟨.operation (.call "solve") [.scalar (.string text)], stack, heap⟩, used⟩
      ⟨⟨.value (number ((solutions.map Sudoku.Text.score).sum)), stack, heap ++ extra⟩, finalUsed⟩ globals := by
  let entry := functionEnv definition.1 definition.2 [.scalar (.string text)] globals
  let ready := setLocal entry "total" (number 0)
  let base := parse_puzzles.normalizedHeap text heap
  obtain ⟨boards, correct, parsed⟩ := parse_puzzles.result text solutions input heap globals
  let parsedHeap := parse_puzzles.work base boards
  have parsedView : RecordsView base.length boards parsedHeap := RecordsView.parsed base boards
  have parsedFrame : ∀ index < heap.length, parsedHeap[index]? = heap[index]? := by
    intro index inside
    exact (parse_puzzles.frame base boards index (by simp [base, parse_puzzles.normalizedHeap]; omega)).trans
      (List.getElem?_append_left inside)
  have parsedSize : heap.length ≤ parsedHeap.length := by
    simp [parsedHeap, parse_puzzles.length, base, parse_puzzles.normalizedHeap]
    omega
  have getText : ready "text" = some (.scalar (.string text)) := by cases globals <;> rfl
  have entered : SetCalls.Reaches choices program anchors
      ⟨⟨.operation (.call "solve") [.scalar (.string text)], stack, heap⟩, used⟩
      ⟨⟨.block [loop, finish] ready, stack, heap⟩, used⟩ globals := by
    apply SetCalls.Reaches.of_heap
    apply HeapCalls.Reaches.next (following := ⟨.block definition.2 entry, stack, heap⟩) rfl
    rw [source]
    exact BlockContinues.assign Evaluates.constant [loop, finish] anchors stack
  have parsing : Evaluates program ready (.call (.name "parse_puzzles") [.name "text"])
      heap (.ref base.length) parsedHeap globals :=
    Evaluates.call (.cons (Evaluates.lookup getText) .nil) parsed
  have iterated : SetCalls.Reaches choices program anchors
      ⟨⟨.block [loop, finish] ready, stack, heap⟩, used⟩
      ⟨⟨.block [cursor base.length 0, finish] ready, stack, parsedHeap⟩, used⟩ globals := by
    apply SetCalls.Reaches.of_heap
    apply HeapCalls.Reaches.next (following :=
      ⟨.expr (.call (.name "parse_puzzles") [.name "text"]) ready, .sequence "grid" body [] [finish] ready :: stack, heap⟩) rfl
    apply (parsing anchors _).trans
    exact .next (by simp [step, run, Mutable.sequence, parsedView.collector, sequenceSource, cursor]) .refl
  obtain ⟨after, finalUsed, _, frame, executed⟩ := loop_result choices globals peers units complete getAll getPeers getUnits
    correct boards base.length 0 (by simp) parsedHeap parsedView
    (solve_cands.search.peer_frame peers heap parsedHeap peerView parsedFrame)
    (solve_cands.search.unit_frame units heap parsedHeap unitView parsedFrame) ready 0
    (by simp [ready, setLocal]) used anchors stack
  have retained : after.take heap.length = heap := by
    apply List.ext_getElem?
    intro index
    by_cases inside : index < heap.length
    · simp only [List.getElem?_take, if_pos inside]
      exact (frame index (lt_of_lt_of_le inside parsedSize)).trans (parsedFrame index inside)
    · rw [List.getElem?_eq_none (by simp; omega), List.getElem?_eq_none (by omega)]
  have same : after = heap ++ after.drop heap.length := by
    simpa only [retained] using (List.take_append_drop heap.length after).symm
  refine ⟨after.drop heap.length, finalUsed, ?_⟩
  have full := entered.trans (iterated.trans executed)
  rw [same] at full
  simpa only [Int.zero_add] using full

/-- The module's actual initialization supplies every table and global needed by the full solver. -/
theorem initialized_result (choices : ChoiceSchedule) (text : String) (solutions : List Board)
    (input : Sudoku.Text.Input text solutions) (caller : Mutable.Heap) (used anchors : Nat) (stack : List HeapCalls.Frame) :
    ∃ extra finalUsed, SetCalls.Reaches choices program anchors
      ⟨⟨.operation (.call "solve") [.scalar (.string text)], stack, (initialization.context caller).heap⟩, used⟩
      ⟨⟨.value (number ((solutions.map Sudoku.Text.score).sum)), stack, (initialization.context caller).heap ++ extra⟩, finalUsed⟩
      (initialization.globals caller) :=
  result choices text solutions input (initialization.context caller).heap (initialization.globals caller)
    (initialization.context caller).peers (build_peers_units.table caller.length) (initialization.complete caller)
    (initialization.context caller).peerView (initialization.context caller).unitView
    (initialization.get_all caller) (initialization.get_peers caller) (initialization.get_units caller) used anchors stack

end
end solvers.«96».solve
