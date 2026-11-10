import solvers.«96.py.lean.d».CandidatePropagationLoop

namespace solvers.«96».reduce_cands
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku
set_option Elab.async false
noncomputable section

theorem full_source : definition.2 =
    [.expr (.constant (.string "Constraint propagation: propagate singles + 'only place for digit in a unit'.")),
     .assign [.name "changed"] (.constant (.bool true)), propagationLoop.statement, .return (.constant (.bool true))] := rfl

theorem executable_source : executableBody definition.2 =
    [.assign [.name "changed"] (.constant (.bool true)), propagationLoop.statement, .return (.constant (.bool true))] := rfl

/-- The full reduce_cands call terminates for every peer-set schedule and preserves legal completions. -/
theorem result (choices : ChoiceSchedule) (globals : Mutable.Globals) (peers : PeerTable) (units : UnitTable)
    (address : Nat) (candidates : Candidates) (heap : Mutable.Heap)
    (found : View address heap candidates) (peerView : peers.View heap) (unitView : units.View heap)
    (getPeers : globals.lookup "PEERS" = some (.ref peers.root)) (getUnits : globals.lookup "UNITS" = some (.ref units.root))
    (nonempty : ∀ cell, candidates cell ≠ ∅) (used anchors : Nat) (stack : List HeapCalls.Frame) :
    ∃ (finalHeap : Mutable.Heap) (finalUsed : Nat) (result : Option Propagation.State), heap.length < finalHeap.length ∧
      (∀ index < heap.length, address ≠ index → finalHeap[index]? = heap[index]?) ∧
      (match result with
       | none => ¬ ∃ assignment, Legal assignment ∧ Admits candidates assignment
       | some after => View address finalHeap after.candidates ∧ propagationLoop.Finished units candidates after) ∧
      SetCalls.Reaches choices program anchors
        ⟨⟨.operation (.call "reduce_cands") [.ref address], stack, heap⟩, used⟩
        ⟨⟨.value (.scalar (.bool result.isSome)), stack, finalHeap⟩, finalUsed⟩ globals := by
  let entry := functionEnv definition.1 definition.2 [.ref address] globals
  let ready := setLocal entry "changed" (.scalar (.bool true))
  have candidatesLookup : ready "cands" = some (.ref address) := by cases globals <;> rfl
  have peersLookup : ready "PEERS" = some (.ref peers.root) := by
    cases globals with
    | nil => simp at getPeers
    | cons first rest =>
        have notLocal : "PEERS" ∉ blockLocals definition.2 := by
          rw [full_source]
          simp [blockLocals, assignedLocals, targetLocals, outerBody, definition, program, Solvers.P96Ast.module, findFunctionIn]
        simpa [ready, setLocal, entry, functionEnv, show "PEERS" ∉ definition.1.map Arg.name from by decide,
          notLocal] using getPeers
  have unitsLookup : ready "UNITS" = some (.ref units.root) := by
    cases globals with
    | nil => simp at getUnits
    | cons first rest =>
        have notLocal : "UNITS" ∉ blockLocals definition.2 := by
          rw [full_source]
          simp [blockLocals, assignedLocals, targetLocals, outerBody, definition, program, Solvers.P96Ast.module, findFunctionIn]
        simpa [ready, setLocal, entry, functionEnv, show "UNITS" ∉ definition.1.map Arg.name from by decide,
          notLocal] using getUnits
  have entered : SetCalls.Reaches choices program anchors
      ⟨⟨.operation (.call "reduce_cands") [.ref address], stack, heap⟩, used⟩
      ⟨⟨.block [propagationLoop.statement, .return (.constant (.bool true))] ready, stack, heap⟩, used⟩ globals := by
    apply SetCalls.Reaches.of_heap
    apply HeapCalls.Reaches.next (following := ⟨.block (executableBody definition.2) entry, stack, heap⟩)
    · rfl
    · rw [executable_source]
      exact BlockContinues.assign (module := program) (globals := globals) Evaluates.constant
        [propagationLoop.statement, .return (.constant (.bool true))] anchors stack
  obtain ⟨finalHeap, finalEnv, finalUsed, result, length, frame, _, executed⟩ :=
    propagationLoop.run choices globals peers units address candidates heap ready used found peerView unitView
      candidatesLookup peersLookup unitsLookup (by simp [ready, setLocal]) nonempty
      [.return (.constant (.bool true))] anchors stack
  refine ⟨finalHeap, finalUsed, result, length, frame, ?_, ?_⟩
  · cases result with
    | none => exact executed.1
    | some after => exact ⟨executed.1, executed.2.2.1⟩
  · cases result with
    | none => exact entered.trans executed.2
    | some after =>
        exact entered.trans (executed.2.2.2.trans (SetCalls.Reaches.of_heap
          (BlockReturns.return_value (module := program) (env := finalEnv) (globals := globals)
            (heap := finalHeap) (rest := []) Evaluates.constant anchors stack) choices finalUsed))

theorem «Returns» (globals : Mutable.Globals) (peers : PeerTable) (units : UnitTable)
    (address : Nat) (candidates : Candidates) (heap : Mutable.Heap)
    (found : View address heap candidates) (peerView : peers.View heap) (unitView : units.View heap)
    (getPeers : globals.lookup "PEERS" = some (.ref peers.root)) (getUnits : globals.lookup "UNITS" = some (.ref units.root))
    (nonempty : ∀ cell, candidates cell ≠ ∅) :
    (SetCalls.call program "reduce_cands" [.ref address] heap globals).Returns := by
  intro choices
  obtain ⟨finalHeap, finalUsed, result, _, _, _, path⟩ :=
    result choices globals peers units address candidates heap found peerView unitView getPeers getUnits nonempty 0 heap.length []
  exact ⟨.scalar (.bool result.isSome), globals, finalHeap, heap.length,
    path.finishes (.done rfl)⟩

end
end solvers.«96».reduce_cands
