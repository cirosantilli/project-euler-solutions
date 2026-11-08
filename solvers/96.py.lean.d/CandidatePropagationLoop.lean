import solvers.«96.py.lean.d».CandidatePropagationBody

namespace solvers.«96».reduce_cands.propagationLoop
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku
set_option Elab.async false
noncomputable section

theorem enter (choices : ChoiceSchedule) (globals : Mutable.Globals) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (changed : env "changed" = some (.scalar (.bool true)))
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    SetCalls.Reaches choices program anchors
      ⟨⟨.block (statement :: following) env, stack, heap⟩, used⟩
      ⟨⟨.block (outerBody ++ statement :: following) env, stack, heap⟩, used⟩ globals := by
  apply SetCalls.Reaches.of_heap
  apply HeapCalls.Reaches.next (following := ⟨.expr (.name "changed") env,
    .branch (outerBody ++ [Calls.loopContinuation outerBody statement]) [] following env :: stack, heap⟩) rfl
  apply (Evaluates.lookup (globals := globals) (module := program) (heap := heap) changed anchors _).trans
  exact .next (by simp [step, run, Mutable.truth, scalarTruth, resume]) .refl

theorem finish (choices : ChoiceSchedule) (globals : Mutable.Globals) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (changed : env "changed" = some (.scalar (.bool false)))
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    SetCalls.Reaches choices program anchors
      ⟨⟨.block (statement :: following) env, stack, heap⟩, used⟩
      ⟨⟨.block following env, stack, heap⟩, used⟩ globals := by
  apply SetCalls.Reaches.of_heap
  apply HeapCalls.Reaches.next (following := ⟨.expr (.name "changed") env,
    .branch (outerBody ++ [Calls.loopContinuation outerBody statement]) [] following env :: stack, heap⟩) rfl
  apply (Evaluates.lookup (globals := globals) (module := program) (heap := heap) changed anchors _).trans
  exact .next (by simp [step, run, Mutable.truth, scalarTruth]) .refl

structure Finished (before : Candidates) (after : Propagation.State) : Prop where
  subset : ∀ cell, after.candidates cell ⊆ before cell
  nonempty : Propagation.Nonempty after
  empty : after.queued = []
  unchanged : after.changed = false
  equivalent : ∀ assignment, Legal assignment → (Admits after.candidates assignment ↔ Admits before assignment)

def outcome (choices : ChoiceSchedule) (globals : Mutable.Globals) (address : Nat)
    (before : Candidates) (result : Option Propagation.State) (start : SetCalls.State) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (used : Nat) (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) : Prop :=
  match result with
  | none => (¬ ∃ assignment, Legal assignment ∧ Admits before assignment) ∧
      SetCalls.Reaches choices program anchors start ⟨⟨.value (.scalar (.bool false)), stack, heap⟩, used⟩ globals
  | some after => View address heap after.candidates ∧ onlyPlace.Locals address after env ∧ Finished before after ∧
      SetCalls.Reaches choices program anchors start ⟨⟨.block following env, stack, heap⟩, used⟩ globals

/-- Every further outer pass consumes candidates, regardless of peer-set iteration order. -/
theorem run (choices : ChoiceSchedule) (globals : Mutable.Globals) (peers : PeerTable) (units : UnitTable)
    (address : Nat) (candidates : Candidates) (heap : Mutable.Heap) (env : HeapCalls.Env) (used : Nat)
    (found : View address heap candidates) (peerView : peers.View heap) (unitView : units.View heap)
    (getCandidates : env "cands" = some (.ref address)) (getPeers : env "PEERS" = some (.ref peers.root))
    (getUnits : env "UNITS" = some (.ref units.root)) (getChanged : env "changed" = some (.scalar (.bool true)))
    (nonempty : ∀ cell, candidates cell ≠ ∅)
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) :
    ∃ finalHeap finalEnv finalUsed result, heap.length < finalHeap.length ∧
      (∀ index < heap.length, address ≠ index → finalHeap[index]? = heap[index]?) ∧
      (∀ name, name ∉ written → finalEnv name = env name) ∧
      outcome choices globals address candidates result ⟨⟨.block (statement :: following) env, stack, heap⟩, used⟩
        finalEnv finalHeap finalUsed following anchors stack := by
  have entry := enter choices globals heap env getChanged following anchors stack used
  obtain ⟨middleHeap, middleEnv, middleUsed, result, length, _, frame, names, executed⟩ :=
    body choices globals peers units address candidates heap env used found peerView unitView getCandidates getPeers getUnits nonempty
      (statement :: following) anchors stack
  cases result with
  | none => exact ⟨middleHeap, middleEnv, middleUsed, none, length, frame, names, executed.1, entry.trans executed.2⟩
  | some middle =>
      rcases executed with ⟨middleView, middleLocals, success, path⟩
      cases changed : middle.changed with
      | false =>
          exact ⟨middleHeap, middleEnv, middleUsed, some middle, length, frame, names, middleView, middleLocals,
            ⟨success.progress.subset, success.nonempty, success.empty, changed, success.equivalent⟩,
            entry.trans (path.trans (finish choices globals middleHeap middleEnv (by simpa [changed] using middleLocals.changed)
              following anchors stack middleUsed))⟩
      | true =>
          have smaller : weight middle.candidates < weight candidates :=
            (success.progress.changed changed).resolve_left (by simp [singletons.initial])
          obtain ⟨finalHeap, finalEnv, finalUsed, finalResult, finalLength, finalFrame, finalNames, finished⟩ :=
            run choices globals peers units address middle.candidates middleHeap middleEnv middleUsed middleView
              (peers_preserved peers address candidates heap middleHeap found peerView frame)
              (units.preserve address heap middleHeap unitView (units.separation address heap candidates unitView found) frame)
              middleLocals.candidates ((names "PEERS" (by decide)).trans getPeers) ((names "UNITS" (by decide)).trans getUnits)
              (by simpa [changed] using middleLocals.changed) success.nonempty following anchors stack
          refine ⟨finalHeap, finalEnv, finalUsed, finalResult, by omega, ?_, ?_, ?_⟩
          · intro index inside different
            exact (finalFrame index (by omega) different).trans (frame index inside different)
          · intro name outside
            exact (finalNames name outside).trans (names name outside)
          · cases finalResult with
            | none =>
                refine ⟨?_, entry.trans (path.trans finished.2)⟩
                rintro ⟨assignment, legal, admitted⟩
                exact finished.1 ⟨assignment, legal, (success.equivalent assignment legal).mpr admitted⟩
            | some after =>
                rcases finished with ⟨finalView, finalLocals, finalSuccess, finalPath⟩
                exact ⟨finalView, finalLocals,
                  ⟨fun cell => (finalSuccess.subset cell).trans (success.progress.subset cell), finalSuccess.nonempty,
                    finalSuccess.empty, finalSuccess.unchanged,
                    fun assignment legal => (finalSuccess.equivalent assignment legal).trans (success.equivalent assignment legal)⟩,
                  entry.trans (path.trans finalPath)⟩
termination_by weight candidates
decreasing_by exact smaller

end
end solvers.«96».reduce_cands.propagationLoop
