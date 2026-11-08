import solvers.«96.py.lean.d».SolveRecord

namespace solvers.«96».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.Sudoku reduce_cands
set_option Elab.async false
noncomputable section

structure RecordsView (root : Nat) (boards : List Board) (heap : Mutable.Heap) : Prop where
  collector : heap[root]? = some (.list (parse_puzzles.references root boards.length))
  grid : ∀ index board, boards[index]? = some board →
    heap[root + 1 + index]? = some (.list (parse_puzzles.rows board))

theorem RecordsView.parsed (base : Mutable.Heap) (boards : List Board) :
    RecordsView base.length boards (parse_puzzles.work base boards) := by
  refine ⟨parse_puzzles.root base boards, ?_⟩
  intro index board selected
  have inside := (List.getElem?_eq_some_iff.mp selected).1
  have found := parse_puzzles.grid base boards ⟨index, inside⟩
  have same := (List.getElem?_eq_some_iff.mp selected).2
  change (parse_puzzles.work base boards)[base.length + 1 + index]? = some (.list (parse_puzzles.rows boards[index])) at found
  simpa only [same] using found

theorem RecordsView.frame {root : Nat} {boards : List Board} {before after : Mutable.Heap}
    (view : RecordsView root boards before) (kept : ∀ index < before.length, after[index]? = before[index]?) :
    RecordsView root boards after := by
  refine ⟨(kept root (List.getElem?_eq_some_iff.mp view.collector).1).trans view.collector, ?_⟩
  intro index board selected
  exact (kept (root + 1 + index) (List.getElem?_eq_some_iff.mp (view.grid index board selected)).1).trans (view.grid index board selected)

abbrev cursor (root position : Nat) : Stmt := .sequenceStepLoop "grid" (.ref root) position body []

/-- The actual record loop sums all unique solutions while preserving the parsed grids and module tables. -/
theorem loop_result (choices : ChoiceSchedule) (globals : Mutable.Globals)
    (peers : PeerTable) (units : UnitTable) (complete : units.Complete)
    (getAll : globals.lookup "ALL" = some (number 511))
    (getPeers : globals.lookup "PEERS" = some (.ref peers.root)) (getUnits : globals.lookup "UNITS" = some (.ref units.root))
    {remaining solutions : List Board} (correct : List.Forall₂ parse_puzzles.Correct remaining solutions)
    (boards : List Board) (root position : Nat) (suffix : boards.drop position = remaining)
    (heap : Mutable.Heap) (view : RecordsView root boards heap) (peerView : peers.View heap) (unitView : units.View heap)
    (env : HeapCalls.Env) (total : Int) (getTotal : env "total" = some (number total))
    (used anchors : Nat) (stack : List HeapCalls.Frame) :
    ∃ after finalUsed, heap.length ≤ after.length ∧
      (∀ index < heap.length, after[index]? = heap[index]?) ∧
      SetCalls.Reaches choices program anchors
        ⟨⟨.block [cursor root position, finish] env, stack, heap⟩, used⟩
        ⟨⟨.value (number (total + (solutions.map Sudoku.Text.score).sum)), stack, after⟩, finalUsed⟩ globals := by
  induction correct generalizing position heap env total used with
  | nil =>
      have outside : boards.length ≤ position := by
        have size := congrArg List.length suffix
        simp only [List.length_drop, List.length_nil] at size
        omega
      have finished : (parse_puzzles.references root boards.length)[position]? = none :=
        List.getElem?_eq_none (by simpa [parse_puzzles.references] using outside)
      refine ⟨heap, used, le_rfl, fun _ _ => rfl, ?_⟩
      simp only [List.map_nil, List.sum_nil, Int.add_zero]
      apply SetCalls.Reaches.of_heap
      apply BlockReturns.sequence_done (by simp [sequenceValue, Mutable.sequence, view.collector]) finished
      exact BlockReturns.return_value (Evaluates.lookup getTotal)
  | @cons board solved remaining solutions first tail ih =>
      have selected : boards[position]? = some board := by
        simpa [suffix] using (List.head?_drop (l := boards) (i := position)).symm
      have inside := (List.getElem?_eq_some_iff.mp selected).1
      have item : (parse_puzzles.references root boards.length)[position]? = some (.ref (root + 1 + position)) := by
        simp [parse_puzzles.references, List.getElem?_map, List.getElem?_range inside]
      let ready := setLocal env "grid" (.ref (root + 1 + position))
      let following := [cursor root (position + 1), finish]
      have entered : SetCalls.Reaches choices program anchors
          ⟨⟨.block [cursor root position, finish] env, stack, heap⟩, used⟩
          ⟨⟨.block (body ++ following) ready, stack, heap⟩, used⟩ globals := by
        apply SetCalls.Reaches.of_heap
        exact .next (by simp [step, cursor, run, sequenceValue, Mutable.sequence, view.collector, item,
          Calls.loopContinuation, show Calls.loopControl body = false from rfl, ready, following]) .refl
      obtain ⟨middle, middleUsed, nextEnv, size, frame, nextTotal, executed⟩ :=
        record choices globals peers units complete board solved first heap (root + 1 + position)
          (view.grid position board selected) peerView unitView getAll getPeers getUnits ready total
          (by simp [ready, setLocal]) (by simpa [ready, setLocal] using getTotal) following used anchors stack
      have nextSuffix : boards.drop (position + 1) = remaining := by simp [List.drop_add_one_eq_tail_drop, suffix]
      obtain ⟨after, finalUsed, finalSize, finalFrame, continued⟩ := ih (position + 1) nextSuffix middle
        (view.frame frame) (solve_cands.search.peer_frame peers heap middle peerView frame)
        (solve_cands.search.unit_frame units heap middle unitView frame) nextEnv (total + Sudoku.Text.score solved)
        nextTotal middleUsed
      refine ⟨after, finalUsed, size.trans finalSize,
        fun index inside => (finalFrame index (lt_of_lt_of_le inside size)).trans (frame index inside), ?_⟩
      simpa only [List.map_cons, List.sum_cons, Int.add_assoc] using entered.trans (executed.trans continued)

end
end solvers.«96».solve
