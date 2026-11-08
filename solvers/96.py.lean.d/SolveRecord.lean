import solvers.«96.py.lean.d».Grid
import solvers.«96.py.lean.d».UniqueSolution
import solvers.«96.py.lean.d».Parser

namespace solvers.«96».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.Sudoku reduce_cands
set_option Elab.async false
noncomputable section

abbrev definition := (findFunctionIn program.body "solve").getD ([], [])
abbrev rejected : Stmt := .ifThen (.builtinCall .identity [.name "solved", .constant .none])
  [.raise (some (.call (.name "ValueError") [.constant (.string "Unsolvable puzzle encountered")]))] []
abbrev accumulate : Stmt := .augAssign (.name "total") .add (.call (.name "top_left_number") [.name "solved"])
abbrev body : List Stmt :=
  [.assign [.name "cands"] (.call (.name "grid_to_cands") [.name "grid"]),
   .assign [.name "solved"] (.call (.name "solve_cands") [.name "cands"]), rejected, accumulate]
abbrev loop : Stmt := .forIn (.name "grid") (.call (.name "parse_puzzles") [.name "text"]) body []
abbrev finish : Stmt := .return (.name "total")

theorem source : definition.2 = [.assign [.name "total"] (.constant (.int 0)), loop, finish] := rfl

/-- One source iteration initializes candidates, searches, and adds the unique completion's score. -/
theorem record (choices : ChoiceSchedule) (globals : Mutable.Globals)
    (peers : PeerTable) (units : UnitTable) (complete : units.Complete)
    (board solved : Board) (correct : parse_puzzles.Correct board solved)
    (heap : Mutable.Heap) (address : Nat) (grid : heap[address]? = some (.list (parse_puzzles.rows board)))
    (peerView : peers.View heap) (unitView : units.View heap)
    (getAll : globals.lookup "ALL" = some (number 511))
    (getPeers : globals.lookup "PEERS" = some (.ref peers.root)) (getUnits : globals.lookup "UNITS" = some (.ref units.root))
    (env : HeapCalls.Env) (total : Int) (getGrid : env "grid" = some (.ref address))
    (getTotal : env "total" = some (number total))
    (following : List Stmt) (used anchors : Nat) (stack : List HeapCalls.Frame) :
    ∃ after finalUsed finalEnv, heap.length ≤ after.length ∧
      (∀ index < heap.length, after[index]? = heap[index]?) ∧
      finalEnv "total" = some (number (total + Sudoku.Text.score solved)) ∧
      SetCalls.Reaches choices program anchors
        ⟨⟨.block (body ++ following) env, stack, heap⟩, used⟩
        ⟨⟨.block following finalEnv, stack, after⟩, finalUsed⟩ globals := by
  let prepared := grid_to_cands.work heap board 81
  let ready := setLocal env "cands" (.ref (heap.length + 1))
  let rest := rejected :: accumulate :: following
  let continuation := HeapCalls.Frame.assign "solved" rest ready :: stack
  have allocated := BlockContinues.assign (name := "cands")
    (Evaluates.call (.cons (Evaluates.lookup getGrid) .nil) (grid_to_cands.result board heap address globals grid getAll))
  have initialized : SetCalls.Reaches choices program anchors
      ⟨⟨.block (body ++ following) env, stack, heap⟩, used⟩
      ⟨⟨.block (.assign [.name "solved"] (.call (.name "solve_cands") [.name "cands"]) :: rest) ready, stack, prepared⟩, used⟩ globals :=
    SetCalls.Reaches.of_heap (allocated _ anchors stack) choices used
  have invocation : SetCalls.Reaches choices program anchors
      ⟨⟨.block (.assign [.name "solved"] (.call (.name "solve_cands") [.name "cands"]) :: rest) ready, stack, prepared⟩, used⟩
      ⟨⟨.operation (.call "solve_cands") [.ref (heap.length + 1)], continuation, prepared⟩, used⟩ globals := by
    apply SetCalls.Reaches.of_heap
    apply HeapCalls.Reaches.next rfl
    apply HeapCalls.Reaches.next rfl
    exact Evaluates.arguments_reach (.cons (Evaluates.lookup (by simp [ready, setLocal])) .nil)
      (.call "solve_cands") [] anchors continuation
  have preparedFrame : ∀ index < heap.length, prepared[index]? = heap[index]? :=
    fun _ inside => List.getElem?_append_left inside
  have found : View (heap.length + 1) prepared (initialCandidates board) := by
    simpa only [View, values, CandidateArray.initial] using grid_to_cands.found heap board 81
  obtain ⟨after, finalUsed, output, size, frame, finalView, called⟩ :=
    solve_cands.unique_result choices globals peers units complete board solved correct.1 correct.2
      (heap.length + 1) prepared found
      (solve_cands.search.peer_frame peers heap prepared peerView preparedFrame)
      (solve_cands.search.unit_frame units heap prepared unitView preparedFrame)
      getPeers getUnits used anchors continuation
  let withSolved := setLocal ready "solved" (.ref output)
  let finalEnv := setLocal withSolved "total" (number (total + Sudoku.Text.score solved))
  have getSolved : withSolved "solved" = some (.ref output) := by simp [withSolved, setLocal]
  have accepted : BlockContinues program withSolved [rejected] after withSolved after globals := by
    apply BlockContinues.branch (flag := false)
      (Evaluates.builtin (.cons (Evaluates.lookup getSolved) (.cons Evaluates.constant .nil)) rfl) rfl .nil
  have accumulated : BlockContinues program withSolved [accumulate] after finalEnv after globals := by
    apply BlockContinues.augAssign (by simpa [withSolved, ready, setLocal] using getTotal)
      (Evaluates.call (.cons (Evaluates.lookup getSolved) .nil)
        (top_left_number.completed solved correct.1.2.1 after output globals finalView))
    simp [inplace, Mutable.binary, scalarBinary, number]
  have bound : SetCalls.Reaches choices program anchors
      ⟨⟨.value (.ref output), continuation, after⟩, finalUsed⟩
      ⟨⟨.block rest withSolved, stack, after⟩, finalUsed⟩ globals :=
    SetCalls.Reaches.of_heap (.next rfl .refl) choices finalUsed
  refine ⟨after, finalUsed, finalEnv, ?_, ?_, by simp [finalEnv, setLocal], ?_⟩
  · exact (show heap.length ≤ prepared.length by simp [prepared, grid_to_cands.work]).trans size
  · intro index inside
    exact (frame index (by simp [prepared, grid_to_cands.work]; omega) (by omega)).trans (preparedFrame index inside)
  · exact initialized.trans (invocation.trans (called.trans (bound.trans
      (SetCalls.Reaches.of_heap ((accepted.append accumulated) following anchors stack) choices finalUsed))))

end
end solvers.«96».solve
