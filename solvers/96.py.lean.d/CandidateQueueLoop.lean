import solvers.«96.py.lean.d».CandidateQueueStep

namespace solvers.«96».reduce_cands.queueLoop
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku
set_option Elab.async false
noncomputable section

def outcome (choices : ChoiceSchedule) (globals : Mutable.Globals) (table : PeerTable) (address queue : Nat)
    (before : Propagation.State) (result : Option Propagation.State) (start : SetCalls.State) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (used : Nat) (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) : Prop :=
  match result with
  | none => (¬ ∃ assignment, Legal assignment ∧ Admits before.candidates assignment) ∧
      SetCalls.Reaches choices program anchors start ⟨⟨.value (.scalar (.bool false)), stack, heap⟩, used⟩ globals
  | some after => peerLoop.View address queue heap after ∧ Locals table address queue after env ∧
      Propagation.Progress before after ∧ Propagation.Nonempty after ∧ Propagation.Singles after ∧ after.queued = [] ∧
      (∀ assignment, Legal assignment → (Admits after.candidates assignment ↔ Admits before.candidates assignment)) ∧
      SetCalls.Reaches choices program anchors start ⟨⟨.block following env, stack, heap⟩, used⟩ globals

/-- The live queue terminates for every peer-set choice schedule, including newly appended singletons. -/
theorem run (choices : ChoiceSchedule) (globals : Mutable.Globals) (table : PeerTable) (address queue : Nat)
    (state : Propagation.State) (heap : Mutable.Heap) (env : HeapCalls.Env) (used : Nat)
    (view : peerLoop.View address queue heap state) (locals : Locals table address queue state env)
    (tableView : table.View heap) (separate : table.Separate address queue) (different : address ≠ queue)
    (nonempty : Propagation.Nonempty state) (singles : Propagation.Singles state)
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) :
    ∃ finalHeap finalEnv finalUsed result, peerLoop.Frame address queue heap finalHeap ∧
      (∀ name, name ∉ ["i", "val", "p", "changed"] → finalEnv name = env name) ∧
      outcome choices globals table address queue state result
        ⟨⟨.block (statement :: following) env, stack, heap⟩, used⟩
        finalEnv finalHeap finalUsed following anchors stack := by
  by_cases empty : state.queued = []
  · exact ⟨heap, env, used, some state, .refl _ _ _, fun _ _ => rfl, view, locals, .refl _, nonempty, singles, empty,
      fun _ _ => Iff.rfl, finish choices globals table address queue state heap env view locals empty following anchors stack used⟩
  · let remaining := state.queued.dropLast
    let cell := state.queued.getLast empty
    have queued : state.queued = remaining ++ [cell] := (List.dropLast_append_getLast empty).symm
    have atCell := singles cell (by rw [queued]; simp)
    obtain ⟨digit, single⟩ := Finset.card_eq_one.mp atCell
    let beforePeers := Propagation.dequeued state remaining
    have popFrame := pop_frame address queue heap remaining
    have popView := pop_view address queue heap state remaining view different
    have tableAfterPop := table.preserve address queue heap (popped heap queue remaining) tableView separate popFrame
    have taken := pop_result table address queue state cell digit remaining heap env globals view locals different queued single
    have entry : SetCalls.Reaches choices program anchors
        ⟨⟨.block (statement :: following) env, stack, heap⟩, used⟩
        ⟨⟨.block (peerLoop.statement :: statement :: following) (ready env cell digit), stack, popped heap queue remaining⟩, used⟩ globals := by
      apply (enter choices globals table address queue state cell remaining heap env view locals queued following anchors stack used).trans
      simpa only [source_body, List.append_assoc, List.singleton_append] using
        SetCalls.Reaches.of_heap (taken (peerLoop.statement :: statement :: following) anchors stack) choices used
    have read := table.read (popped heap queue remaining) cell (ready env cell digit) globals tableAfterPop
      (by simpa [ready, setLocal] using locals.peers) (by simp [ready, setLocal])
    obtain ⟨peerHeap, peerEnv, peerUsed, peerFrame, peerEnvFrame, completed⟩ :=
      peerLoop.execute choices globals address queue (table.address cell) (table.cells cell) beforePeers digit
        (popped heap queue remaining) (ready env cell digit) used popView
        (ready_locals table address queue state cell digit remaining env locals) different
        (separate.setCandidates cell) (separate.setQueue cell) (tableAfterPop.sets cell) read
        (statement :: following) anchors stack
    have frame := popFrame.trans peerFrame
    have names : ∀ name, name ∉ ["i", "val", "p", "changed"] → peerEnv name = env name := by
      intro name outside
      have away : name ≠ "i" ∧ name ≠ "val" ∧ name ≠ "p" ∧ name ≠ "changed" := by simpa using outside
      rw [peerEnvFrame name (by simp [away.2.2.1, away.2.2.2])]
      simp [ready, setLocal, away.1, away.2.1]
    have related : ∀ other ∈ SetCalls.selectionOrder choices used (table.cells cell), Peer cell other := by
      intro other member
      exact table.related cell other ((SetCalls.selectionOrder_perm choices used (table.cells cell)).mem_iff.mp member)
    cases result : Propagation.peers beforePeers digit (SetCalls.selectionOrder choices used (table.cells cell)) with
    | none =>
        simp only [result, peerLoop.outcome] at completed
        exact ⟨peerHeap, peerEnv, peerUsed, none, frame, names,
          Propagation.peers_impossible beforePeers cell digit _ single related result,
          entry.trans completed⟩
    | some middle =>
        simp only [result, peerLoop.outcome] at completed
        obtain ⟨progress, middleNonempty, middleSingles, _, equivalent⟩ := Propagation.peers_some beforePeers middle cell digit _
          single related nonempty (Propagation.dequeued_singles state remaining cell queued singles) result
        have middleLocals : Locals table address queue middle peerEnv :=
          ⟨completed.2.1.candidates, completed.2.1.queue, completed.2.1.changed,
            (names "PEERS" (by simp)).trans locals.peers⟩
        have smaller : Propagation.potential middle < Propagation.potential state :=
          progress.potential.trans_lt (Propagation.dequeued_potential state remaining cell queued)
        have beforeLoop := entry.trans completed.2.2
        obtain ⟨finalHeap, finalEnv, finalUsed, finalResult, finalFrame, envFrame, executed⟩ :=
          run choices globals table address queue middle peerHeap peerEnv peerUsed completed.1 middleLocals
            (table.preserve address queue heap peerHeap tableView separate frame) separate different
            middleNonempty middleSingles following anchors stack
        refine ⟨finalHeap, finalEnv, finalUsed, finalResult, frame.trans finalFrame,
          fun name outside => (envFrame name outside).trans (names name outside), ?_⟩
        cases finalResult with
        | none =>
            rcases executed with ⟨impossible, path⟩
            refine ⟨?_, beforeLoop.trans path⟩
            rintro ⟨assignment, legal, admitted⟩
            exact impossible ⟨assignment, legal, (equivalent assignment legal).mpr admitted⟩
        | some after =>
            rcases executed with ⟨finalView, finalLocals, finalProgress, finalNonempty, finalSingles, finalEmpty, finalEquivalent, path⟩
            exact ⟨finalView, finalLocals, ((Propagation.dequeued_progress state remaining cell queued).trans progress).trans finalProgress,
              finalNonempty, finalSingles, finalEmpty,
              fun assignment legal => (finalEquivalent assignment legal).trans (equivalent assignment legal), beforeLoop.trans path⟩
termination_by Propagation.potential state
decreasing_by exact smaller

end
end solvers.«96».reduce_cands.queueLoop
