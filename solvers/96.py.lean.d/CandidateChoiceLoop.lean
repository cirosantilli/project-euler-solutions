import solvers.«96.py.lean.d».CandidateChoice

namespace solvers.«96».solve_cands.choice
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku reduce_cands
set_option Elab.async false
noncomputable section

theorem iteration (address : Nat) (candidates : Candidates) (slot : Fin 81) (best : Option (Fin 81))
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals) (found : View address heap candidates)
    (getCandidates : env "cands" = some (.ref address)) (getIndex : env "i" = some (number slot.val))
    (known : State candidates heap env best) :
    ∃ updated extra, BlockContinues program env body heap updated (heap ++ extra) globals ∧
      State candidates (heap ++ extra) updated (Choice.consider candidates best slot) ∧
      updated "cands" = env "cands" ∧ extra.length ≤ 1 := by
  by_cases multiple : 1 < Choice.count candidates slot
  · obtain ⟨updated, executed, state, preserved⟩ :=
      eligible address candidates slot best env heap globals found getCandidates getIndex known multiple
    refine ⟨updated, [.tuple (pair candidates slot)], ?_, state, preserved, by simp⟩
    exact BlockContinues.branch (flag := true) (filtered address candidates slot env heap globals found getCandidates getIndex)
      (by simp [multiple, Mutable.truth, scalarTruth]) executed
  · refine ⟨env, [], ?_, ?_, rfl, by simp⟩
    · simpa only [List.append_nil] using BlockContinues.branch (flag := false)
        (filtered address candidates slot env heap globals found getCandidates getIndex)
        (by simp [multiple, Mutable.truth, scalarTruth]) (BlockContinues.nil)
    · simpa only [List.append_nil, Choice.consider, if_neg multiple] using known

/-- The lazy generator visits each row-major slot once, retaining its minimum eligible tuple. -/
theorem loop (length first : Nat) (remaining : first + length = 81) (address : Nat) (candidates : Candidates)
    (best : Option (Fin 81)) (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (found : View address heap candidates) (getCandidates : env "cands" = some (.ref address))
    (known : State candidates heap env best) (minimal : Choice.Best candidates ((List.finRange 81).take first) best) :
    ∃ updated extra finalBest, BlockContinues program env [.rangeStepLoop "i" first 81 1 body []]
        heap updated (heap ++ extra) globals ∧
      State candidates (heap ++ extra) updated finalBest ∧
      Choice.Best candidates (List.finRange 81) finalBest ∧
      updated "cands" = some (.ref address) ∧ extra.length ≤ length := by
  induction length generalizing first best env heap with
  | zero =>
      have atEnd : first = 81 := by omega
      refine ⟨env, [], best, ?_, by simpa using known, ?_, getCandidates, by simp⟩
      · simpa only [List.append_nil] using
          (BlockContinues.range_done (module := program) (env := env) (heap := heap) (globals := globals)
            (name := "i") (next := first) (stop := 81) (stride := 1) (body := body) (otherwise := [])
            (by simp [Calls.rangeContinues, atEnd]) .nil)
      · simpa [atEnd] using minimal
  | succ length ih =>
      let slot : Fin 81 := ⟨first, by omega⟩
      let bound := setLocal env "i" (number first)
      have boundCandidates : bound "cands" = some (.ref address) := by simpa [bound, setLocal] using getCandidates
      obtain ⟨afterCell, cellExtra, cellRun, cellState, cellCandidates, cellSize⟩ :=
        iteration address candidates slot best bound heap globals found boundCandidates rfl (known.bind_index first)
      have afterView : View address (heap ++ cellExtra) candidates := by
        exact (List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1).trans found
      have nextMinimal : Choice.Best candidates ((List.finRange 81).take (first + 1))
          (Choice.consider candidates best slot) := by
        have prefixEq : (List.finRange 81).take (first + 1) = (List.finRange 81).take first ++ [slot] := by
          rw [List.take_succ_eq_append_getElem (by simp; omega)]
          simp [slot]
        rw [prefixEq]
        exact Choice.step candidates _ best slot minimal
      obtain ⟨updated, restExtra, finalBest, rest, finalState, finalMinimal, finalCandidates, restSize⟩ :=
        ih (first + 1) (by omega) (Choice.consider candidates best slot) afterCell (heap ++ cellExtra)
          afterView (cellCandidates.trans boundCandidates) cellState nextMinimal
      refine ⟨updated, cellExtra ++ restExtra, finalBest, ?_, ?_, finalMinimal, finalCandidates, ?_⟩
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa only [Calls.loopContinuation, show Calls.loopControl body = false from rfl,
          Bool.false_eq_true, if_false, Int.natCast_add, Int.natCast_one, List.append_assoc] using cellRun.append rest
      · simpa only [List.append_assoc] using finalState
      · simp only [List.length_append]
        omega

/-- An unsolved, nonempty candidate array gives min a nonempty stream and an eligible minimizing cell. -/
theorem minimum (address : Nat) (candidates : Candidates) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (found : View address heap candidates) (getCandidates : env "cands" = some (.ref address))
    (nonempty : ∀ cell, candidates cell ≠ ∅) (unsolved : ¬ ∀ cell, (candidates cell).card = 1) :
    ∃ slot extra tupleAddress, Evaluates program env expression heap (.ref tupleAddress) (heap ++ extra) globals ∧
      (heap ++ extra)[tupleAddress]? = some (.tuple (pair candidates slot)) ∧
      1 < Choice.count candidates slot ∧
      (∀ other, 1 < Choice.count candidates other → Choice.priority candidates slot ≤ Choice.priority candidates other) ∧
      extra.length ≤ 81 := by
  let initial := extremumEnv (comprehensionTargetsEnv env generators) (.scalar .none)
  have initialCandidates : initial "cands" = some (.ref address) := by
    simpa [initial, extremumEnv, setLocal, extremumBest, extremumSeen, comprehensionTargetsEnv, generators, targetLocals]
      using getCandidates
  have initialState : State candidates heap initial none := by
    refine ⟨?_, trivial⟩
    simp [initial, extremumEnv, setLocal, extremumBest, extremumSeen]
  obtain ⟨updated, extra, best, ran, state, minimal, _, size⟩ :=
    loop 81 0 rfl address candidates none initial heap globals found initialCandidates initialState (Choice.initial candidates)
  obtain ⟨slot, selected, multiple, least⟩ := Choice.finish candidates best minimal nonempty unsolved
  subst best
  obtain ⟨tupleAddress, getBest, tupleFound⟩ := state.2
  refine ⟨slot, extra, tupleAddress, ?_, tupleFound, multiple, least, size⟩
  apply Evaluates.extremum_generator_range (maximum := false) (fallback := none)
    (name := "i") (filters := [filter]) (generators := [])
    (middle := heap) (afterDefault := heap) (.cons Evaluates.constant .nil) rfl Evaluates.constant
  have returned : BlockReturns program updated (extremumFinish false) (heap ++ extra) (.ref tupleAddress) (heap ++ extra) globals := by
    apply BlockReturns.branch (flag := true) (Evaluates.lookup state.1) rfl
    exact BlockReturns.return_value (Evaluates.lookup getBest)
  exact ran.returns returned

/-- The source tuple unpack binds idx to the minimum-remaining-values cell, leaving the candidate array intact. -/
theorem execute (address : Nat) (candidates : Candidates) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (found : View address heap candidates) (getCandidates : env "cands" = some (.ref address))
    (nonempty : ∀ cell, candidates cell ≠ ∅) (unsolved : ¬ ∀ cell, (candidates cell).card = 1) :
    ∃ slot extra, BlockContinues program env [statement] heap
        (assignNames env ["_", "idx"] (pair candidates slot)) (heap ++ extra) globals ∧
      1 < Choice.count candidates slot ∧
      (∀ other, 1 < Choice.count candidates other → Choice.priority candidates slot ≤ Choice.priority candidates other) ∧
      View address (heap ++ extra) candidates ∧ extra.length ≤ 81 := by
  obtain ⟨slot, extra, tupleAddress, evaluated, tupleFound, multiple, least, size⟩ :=
    minimum address candidates env heap globals found getCandidates nonempty unsolved
  refine ⟨slot, extra, ?_, multiple, least, ?_, size⟩
  · apply BlockContinues.unpack (names := ["_", "idx"]) (items := pair candidates slot) rfl evaluated
      (by simp [Mutable.sequence, tupleFound]) rfl
  · exact (List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1).trans found

end
end solvers.«96».solve_cands.choice
