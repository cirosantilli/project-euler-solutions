import solvers.«96.py.lean.d».CandidateSearch
import ProjectEulerSolutions.Lib.SudokuCompletion
import ProjectEulerSolutions.Lib.SudokuText

namespace solvers.«96»
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.Sudoku reduce_cands
set_option Elab.async false
noncomputable section

/-- A uniquely solvable input forces every permitted search execution to return that board. -/
theorem solve_cands.unique_result (choices : ChoiceSchedule) (globals : Mutable.Globals)
    (peers : PeerTable) (units : UnitTable) (complete : units.Complete)
    (board solved : Board) (completed : Completion board solved)
    (unique : ∀ final, Completion board final → final = solved)
    (address : Nat) (heap : Mutable.Heap) (found : View address heap (initialCandidates board))
    (peerView : peers.View heap) (unitView : units.View heap)
    (getPeers : globals.lookup "PEERS" = some (.ref peers.root)) (getUnits : globals.lookup "UNITS" = some (.ref units.root))
    (used anchors : Nat) (stack : List HeapCalls.Frame) :
    ∃ after finalUsed output, heap.length ≤ after.length ∧
      (∀ index < heap.length, address ≠ index → after[index]? = heap[index]?) ∧
      View output after (fun cell => {assignment solved cell}) ∧
      SetCalls.Reaches choices program anchors
        ⟨⟨.operation (.call "solve_cands") [.ref address], stack, heap⟩, used⟩
        ⟨⟨.value (.ref output), stack, after⟩, finalUsed⟩ globals := by
  obtain ⟨after, finalUsed, answer, size, frame, correct, executed⟩ :=
    search.result choices globals peers units complete address (initialCandidates board) heap found peerView unitView
      getPeers getUnits (initial_nonempty board) used anchors stack
  cases answer with
  | none => exact False.elim (correct ⟨assignment solved, completed_assignment board solved completed⟩)
  | some answer =>
      have same := unique_assignment board solved unique answer.assignment correct.1 correct.2.1
      exact ⟨after, finalUsed, answer.address, size, frame, by simpa only [same] using correct.2.2, executed⟩

/-- The three source reads recover the completed board's decimal score from singleton masks. -/
theorem top_left_number.completed (solved : Board) (full : Full solved) (heap : Mutable.Heap)
    (address : Nat) (globals : Mutable.Globals) (found : View address heap (fun cell => {assignment solved cell})) :
    ReturnsValue program "top_left_number" [.ref address] heap (number (Sudoku.Text.score solved)) heap globals := by
  let digits := assignment solved
  let masks : List Mutable.Value := List.ofFn fun slot : Fin 81 => number (2 ^ (digits (Encoding.cell slot)).val)
  have encoded : values (fun cell => {assignment solved cell}) = masks := by
    simp only [values, CandidateArray.encode, List.map_ofFn, Masks.singleton]
    rfl
  have first : masks = number (2 ^ (digits (0, 0)).val) :: number (2 ^ (digits (0, 1)).val) ::
      number (2 ^ (digits (0, 2)).val) :: masks.drop 3 := by
    rfl
  have cells (cell : Cell) : Values.encode (solved cell) = (digits cell).val + 1 := by
    rw [← congrFun (assignment_full solved full) cell]
    rfl
  have stored : heap[address]? = some (.list masks) := by simpa only [View, encoded] using found
  rw [first] at stored
  have executed := result (digits (0, 0)).val (digits (0, 1)).val (digits (0, 2)).val (masks.drop 3) heap address globals stored
  simpa only [Sudoku.Text.score, cells] using executed

end
end solvers.«96»
