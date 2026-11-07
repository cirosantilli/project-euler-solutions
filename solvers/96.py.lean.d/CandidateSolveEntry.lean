import solvers.«96.py.lean.d».CandidateSolved

namespace solvers.«96».solve_cands.entry
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku reduce_cands
set_option Elab.async false
noncomputable section

abbrev reject : Stmt := .ifThen (.unaryOp .not (.call (.name "reduce_cands") [.name "cands"]))
  [.return (.constant .none)] []

theorem source : solved.definition.2.take 2 = [reject, solved.statement] := rfl

def outcome (choices : ChoiceSchedule) (globals : Mutable.Globals) (units : UnitTable) (address : Nat)
    (before : Candidates) (result : Option Propagation.State) (start : SetCalls.State) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (used : Nat) (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) : Prop :=
  match result with
  | none => (¬ ∃ assignment, Legal assignment ∧ Admits before assignment) ∧
      SetCalls.Reaches choices program anchors start ⟨⟨.value (.scalar .none), stack, heap⟩, used⟩ globals
  | some after => View address heap after.candidates ∧ propagationLoop.Finished units before after ∧
      SetCalls.Reaches choices program anchors start
        ⟨⟨if ∀ cell, (after.candidates cell).card = 1 then .value (.ref address) else .block following env, stack, heap⟩, used⟩ globals

/-- Propagation failures return None; solved candidates return directly; other candidates reach search. -/
theorem execute (choices : ChoiceSchedule) (globals : Mutable.Globals) (peers : PeerTable) (units : UnitTable)
    (address : Nat) (candidates : Candidates) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (found : View address heap candidates) (peerView : peers.View heap) (unitView : units.View heap)
    (getCandidates : env "cands" = some (.ref address))
    (getPeers : globals.lookup "PEERS" = some (.ref peers.root)) (getUnits : globals.lookup "UNITS" = some (.ref units.root))
    (nonempty : ∀ cell, candidates cell ≠ ∅) (following : List Stmt) (used anchors : Nat) (stack : List HeapCalls.Frame) :
    ∃ finalHeap finalUsed result, heap.length < finalHeap.length ∧
      (∀ index < heap.length, address ≠ index → finalHeap[index]? = heap[index]?) ∧
      outcome choices globals units address candidates result
        ⟨⟨.block (solved.definition.2.take 2 ++ following) env, stack, heap⟩, used⟩
        env finalHeap finalUsed following anchors stack := by
  let continuation : List HeapCalls.Frame :=
    .unary .not :: .branch [.return (.constant .none)] [] (solved.statement :: following) env :: stack
  have entered : SetCalls.Reaches choices program anchors
      ⟨⟨.block (solved.definition.2.take 2 ++ following) env, stack, heap⟩, used⟩
      ⟨⟨.operation (.call "reduce_cands") [.ref address], continuation, heap⟩, used⟩ globals := by
    rw [source]
    apply SetCalls.Reaches.of_heap
    apply HeapCalls.Reaches.next rfl
    apply HeapCalls.Reaches.next rfl
    apply HeapCalls.Reaches.next rfl
    exact Evaluates.arguments_reach (.cons (Evaluates.lookup getCandidates) .nil) (.call "reduce_cands") [] anchors continuation
  obtain ⟨finalHeap, finalUsed, result, length, frame, conclusion, reduced⟩ :=
    reduce_cands.result choices globals peers units address candidates heap found peerView unitView getPeers getUnits nonempty
      used anchors continuation
  refine ⟨finalHeap, finalUsed, result, length, frame, ?_⟩
  cases result with
  | none =>
      refine ⟨conclusion, (entered.trans reduced).trans ?_⟩
      apply SetCalls.Reaches.of_heap
      apply HeapCalls.Reaches.next rfl
      apply HeapCalls.Reaches.next rfl
      exact BlockReturns.return_value (module := program) (env := env) (globals := globals)
        (heap := finalHeap) (rest := solved.statement :: following) Evaluates.constant anchors stack
  | some after =>
      refine ⟨conclusion.1, conclusion.2, ?_⟩
      have returned : SetCalls.Reaches choices program anchors
          ⟨⟨.value (.scalar (.bool true)), continuation, finalHeap⟩, finalUsed⟩
          ⟨⟨.block (solved.statement :: following) env, stack, finalHeap⟩, finalUsed⟩ globals :=
        SetCalls.Reaches.of_heap (.next rfl (.next rfl .refl)) choices finalUsed
      have tested := (entered.trans reduced).trans returned
      by_cases single : ∀ cell, (after.candidates cell).card = 1
      · rw [if_pos single]
        exact tested.trans (SetCalls.Reaches.of_heap
          (solved.accept address after.candidates env finalHeap globals conclusion.1 getCandidates single following anchors stack)
          choices finalUsed)
      · rw [if_neg single]
        exact tested.trans (SetCalls.Reaches.of_heap
          (solved.continue_search address after.candidates env finalHeap globals conclusion.1 getCandidates single following anchors stack)
          choices finalUsed)

end
end solvers.«96».solve_cands.entry
