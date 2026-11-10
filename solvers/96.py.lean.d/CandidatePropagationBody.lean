import solvers.«96.py.lean.d».CandidateSingletons
import solvers.«96.py.lean.d».CandidateUnits

namespace solvers.«96».reduce_cands.propagationLoop
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku
set_option Elab.async false
noncomputable section

abbrev statement : Stmt := .whileDo (.name "changed") outerBody []
abbrev written : List String := ["queue", "changed", "i", "val", "p", "unit", "k", "dbit", "places"]

theorem source : definition.2.drop 2 = [statement, .return (.constant (.bool true))] := rfl
theorem body_source : outerBody = outerBody.take 3 ++ [onlyPlace.statement] := rfl
theorem resume : Calls.loopContinuation outerBody statement = statement := rfl

structure Success (units : UnitTable) (before : Candidates) (after : Propagation.State) : Prop where
  progress : Propagation.Progress (singletons.initial before) after
  nonempty : Propagation.Nonempty after
  empty : after.queued = []
  equivalent : ∀ assignment, Legal assignment → (Admits after.candidates assignment ↔ Admits before assignment)
  checked : after.changed = false → units.Covered after.candidates

def bodyOutcome (choices : ChoiceSchedule) (globals : Mutable.Globals) (units : UnitTable) (address : Nat)
    (before : Candidates) (result : Option Propagation.State) (start : SetCalls.State) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (used : Nat) (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) : Prop :=
  match result with
  | none => (¬ ∃ assignment, Legal assignment ∧ Admits before assignment) ∧
      SetCalls.Reaches choices program anchors start ⟨⟨.value (.scalar (.bool false)), stack, heap⟩, used⟩ globals
  | some after => View address heap after.candidates ∧ onlyPlace.Locals address after env ∧ Success units before after ∧
      SetCalls.Reaches choices program anchors start ⟨⟨.block following env, stack, heap⟩, used⟩ globals

/-- Both propagation phases, with one fresh queue and at most nine fresh lists per unit. -/
theorem body (choices : ChoiceSchedule) (globals : Mutable.Globals) (peers : PeerTable) (units : UnitTable)
    (address : Nat) (candidates : Candidates) (heap : Mutable.Heap) (env : HeapCalls.Env) (used : Nat)
    (found : View address heap candidates) (peerView : peers.View heap) (unitView : units.View heap)
    (getCandidates : env "cands" = some (.ref address)) (getPeers : env "PEERS" = some (.ref peers.root))
    (getUnits : env "UNITS" = some (.ref units.root)) (nonempty : ∀ cell, candidates cell ≠ ∅)
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) :
    ∃ finalHeap finalEnv finalUsed result, heap.length < finalHeap.length ∧
      finalHeap.length ≤ heap.length + 1 + 9 * units.units.length ∧
      (∀ index < heap.length, address ≠ index → finalHeap[index]? = heap[index]?) ∧
      (∀ name, name ∉ written → finalEnv name = env name) ∧
      bodyOutcome choices globals units address candidates result ⟨⟨.block (outerBody ++ following) env, stack, heap⟩, used⟩
        finalEnv finalHeap finalUsed following anchors stack := by
  obtain ⟨middleHeap, middleEnv, middleUsed, result, length, frame, names, executed⟩ :=
    singletons.execute choices globals peers address candidates heap env used found peerView getCandidates getPeers nonempty
      (onlyPlace.statement :: following) anchors stack
  have start : outerBody ++ following = outerBody.take 3 ++ onlyPlace.statement :: following := by rw [body_source, List.append_assoc]; rfl
  have firstNames : ∀ name, name ∉ written → middleEnv name = env name := by
    intro name outside
    apply names name
    intro member
    apply outside
    simp only [written, List.mem_cons] at member ⊢
    tauto
  cases result with
  | none =>
      refine ⟨middleHeap, middleEnv, middleUsed, none, by omega, by omega, frame, firstNames, executed.1, ?_⟩
      simpa only [start] using executed.2
  | some middle =>
      rcases executed with ⟨middleView, middleLocals, progress, middleNonempty, _, empty, equivalent, path⟩
      have unitKept := units.preserve address heap middleHeap unitView (units.separation address heap candidates unitView found) frame
      have lookup := (names "UNITS" (by decide)).trans getUnits
      obtain ⟨finalHeap, finalEnv, finalResult, lower, upper, finalFrame, finalNames, conclusion, checked, finished⟩ :=
        onlyPlace.phase units address middle middleHeap middleEnv globals middleView.candidates
          ⟨middleLocals.candidates, middleLocals.changed⟩ unitKept lookup middleNonempty
      refine ⟨finalHeap, finalEnv, middleUsed, finalResult, by omega, by omega, ?_, ?_, ?_⟩
      · intro index inside different
        exact (finalFrame index (by omega) different).trans (frame index inside different)
      · intro name outside
        have away : name ∉ ["unit", "k", "dbit", "places", "changed"] := by
          intro member
          apply outside
          simp only [written, List.mem_cons] at member ⊢
          tauto
        exact (finalNames name away).trans (firstNames name outside)
      · cases finalResult with
        | none =>
            refine ⟨?_, ?_⟩
            · rintro ⟨assignment, legal, admitted⟩
              exact conclusion ⟨assignment, legal, (equivalent assignment legal).mpr admitted⟩
            · rw [start]
              exact path.trans (SetCalls.Reaches.of_heap (finished following anchors stack) choices middleUsed)
        | some after =>
            refine ⟨finished.1, finished.2.1,
              ⟨progress.trans conclusion.progress, conclusion.nonempty, conclusion.queued.trans empty,
                fun assignment legal => (conclusion.equivalent assignment legal).trans (equivalent assignment legal), checked⟩, ?_⟩
            rw [start]
            exact path.trans (SetCalls.Reaches.of_heap (finished.2.2 following anchors stack) choices middleUsed)

theorem peers_preserved (peers : PeerTable) (address : Nat) (candidates : Candidates) (before after : Mutable.Heap)
    (found : View address before candidates) (view : peers.View before)
    (frame : ∀ index < before.length, address ≠ index → after[index]? = before[index]?) : peers.View after := by
  have separate := (singletons.separation peers address before candidates view found).2
  exact ⟨(frame peers.root (List.getElem?_eq_some_iff.mp view.root).1 separate.rootCandidates).trans view.root,
    fun cell => (frame (peers.address cell) (List.getElem?_eq_some_iff.mp (view.sets cell)).1
      (separate.setCandidates cell)).trans (view.sets cell)⟩

end
end solvers.«96».reduce_cands.propagationLoop
