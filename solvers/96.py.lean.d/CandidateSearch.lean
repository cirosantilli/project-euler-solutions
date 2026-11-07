import solvers.«96.py.lean.d».CandidateSearchLoop

namespace solvers.«96».solve_cands.search
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku reduce_cands
set_option Elab.async false
noncomputable section

/-- The complete recursive solver terminates, finds a legal completion when one exists,
and preserves every caller object except its candidate array. Peer-set iteration is arbitrary. -/
theorem result (choices : ChoiceSchedule) (globals : Mutable.Globals) (peers : PeerTable) (units : UnitTable)
    (complete : units.Complete) (address : Nat) (candidates : Candidates) (heap : Mutable.Heap)
    (found : View address heap candidates) (peerView : peers.View heap) (unitView : units.View heap)
    (getPeers : globals.lookup "PEERS" = some (.ref peers.root)) (getUnits : globals.lookup "UNITS" = some (.ref units.root))
    (nonempty : ∀ cell, candidates cell ≠ ∅) : Executes choices globals address candidates heap := by
  intro used anchors stack
  let env := functionEnv solved.definition.1 solved.definition.2 [.ref address] globals
  have getCandidates : env "cands" = some (.ref address) := by cases globals <;> rfl
  have entered : SetCalls.Reaches choices program anchors
      ⟨⟨.operation (.call "solve_cands") [.ref address], stack, heap⟩, used⟩
      ⟨⟨.block (solved.definition.2.take 2 ++ solved.definition.2.drop 2) env, stack, heap⟩, used⟩ globals :=
    SetCalls.Reaches.of_heap (.next rfl .refl) choices used
  obtain ⟨afterHeap, afterUsed, propagated, length, frame, propagatedResult⟩ :=
    entry.execute choices globals peers units address candidates heap env found peerView unitView getCandidates
      getPeers getUnits nonempty (solved.definition.2.drop 2) used anchors stack
  cases propagated with
  | none => exact ⟨afterHeap, afterUsed, none, length.le, frame, propagatedResult.1, entered.trans propagatedResult.2⟩
  | some after =>
      rcases propagatedResult with ⟨afterView, finished, reached⟩
      by_cases single : ∀ cell, (after.candidates cell).card = 1
      · obtain ⟨assignment, legal, admitted, assigned, _⟩ := finished.solved complete single
        have equal : after.candidates = fun cell => {assignment cell} := funext assigned
        refine ⟨afterHeap, afterUsed, some ⟨address, assignment⟩, length.le, frame,
          ⟨legal, admitted, by simpa only [equal] using afterView⟩, ?_⟩
        exact entered.trans (by simpa only [if_pos single] using reached)
      · have afterPeers := propagationLoop.peers_preserved peers address candidates heap afterHeap found peerView frame
        have afterUnits := units.preserve address heap afterHeap unitView (units.separation address heap candidates unitView found) frame
        obtain ⟨slot, extra, selected, multiple, _, selectedView, _⟩ :=
          choice.execute address after.candidates env afterHeap globals afterView getCandidates finished.nonempty single
        let chosen := assignNames env ["_", "idx"] (choice.pair after.candidates slot)
        let ready := setLocal chosen "mask" (number (Masks.encode (after.candidates (Encoding.cell slot))))
        let selectedHeap := afterHeap ++ extra
        have chosenCandidates : chosen "cands" = some (.ref address) := by
          simpa [chosen, assignNames, setLocal] using getCandidates
        have chosenIndex : chosen "idx" = some (number (Encoding.index (Encoding.cell slot)).val) := by
          simp [chosen, assignNames, setLocal, Encoding.index_cell]
        have readyLocals : Locals address (Encoding.cell slot) (after.candidates (Encoding.cell slot)) ready := by
          constructor
          · simpa [ready, setLocal] using chosenCandidates
          · simpa [ready, setLocal] using chosenIndex
          · simp [ready, setLocal]
        have initialized := bits.start address after.candidates (Encoding.cell slot) chosen selectedHeap globals
          selectedView chosenCandidates chosenIndex
        have prepared : SetCalls.Reaches choices program anchors
            ⟨⟨.block (solved.definition.2.drop 2) env, stack, afterHeap⟩, afterUsed⟩
            ⟨⟨.block [statement, .return (.constant .none)] ready, stack, selectedHeap⟩, afterUsed⟩ globals :=
          SetCalls.Reaches.of_heap ((selected.append initialized) [statement, .return (.constant .none)] anchors stack) choices afterUsed
        have selectedFrame : ∀ index < afterHeap.length, selectedHeap[index]? = afterHeap[index]? :=
          fun _ inside => List.getElem?_append_left inside
        have afterWeight : weight after.candidates ≤ weight candidates :=
          Finset.sum_le_sum (fun cell _ => Finset.card_le_card (finished.subset cell))
        have recurse : ∀ digit ∈ after.candidates (Encoding.cell slot), ∀ childAddress childHeap,
            View childAddress childHeap (Function.update after.candidates (Encoding.cell slot) {digit}) →
            peers.View childHeap → units.View childHeap →
            Executes choices globals childAddress (Function.update after.candidates (Encoding.cell slot) {digit}) childHeap := by
          intro digit _ childAddress childHeap childView childPeers childUnits
          have smaller : weight (Function.update after.candidates (Encoding.cell slot) {digit}) < weight candidates :=
            (branch_weight_lt after.candidates (Encoding.cell slot) digit multiple).trans_le afterWeight
          exact result choices globals peers units complete childAddress
            (Function.update after.candidates (Encoding.cell slot) {digit}) childHeap childView childPeers childUnits
            getPeers getUnits (clone.nonempty after.candidates (Encoding.cell slot) digit finished.nonempty)
        obtain ⟨finalHeap, finalUsed, answer, finalLength, finalFrame, correct, searched⟩ :=
          loop choices globals peers units address after.candidates (Encoding.cell slot)
            (after.candidates (Encoding.cell slot)) selectedHeap ready afterUsed selectedView
            (peer_frame peers afterHeap selectedHeap afterPeers selectedFrame)
            (unit_frame units afterHeap selectedHeap afterUnits selectedFrame) readyLocals (fun _ present => present)
            recurse anchors stack
        have selectedLength : afterHeap.length ≤ selectedHeap.length := by simp [selectedHeap]
        refine ⟨finalHeap, finalUsed, answer, length.le.trans (selectedLength.trans finalLength), ?_, ?_, ?_⟩
        · intro index inside different
          exact (finalFrame index (lt_of_lt_of_le (inside.trans length) selectedLength)).trans
            ((selectedFrame index (inside.trans length)).trans (frame index inside different))
        · cases answer with
          | none =>
              rintro ⟨assignment, legal, admitted⟩
              have afterAdmitted := (finished.equivalent assignment legal).mpr admitted
              exact correct ⟨assignment, legal, afterAdmitted, afterAdmitted (Encoding.cell slot)⟩
          | some answer =>
              exact ⟨correct.1, (finished.equivalent answer.assignment correct.1).mp correct.2.1, correct.2.2⟩
        · simp only [if_neg single] at reached
          exact entered.trans (reached.trans (prepared.trans searched))
termination_by weight candidates
decreasing_by exact smaller

theorem «Returns» (globals : Mutable.Globals) (peers : PeerTable) (units : UnitTable) (complete : units.Complete)
    (address : Nat) (candidates : Candidates) (heap : Mutable.Heap)
    (found : View address heap candidates) (peerView : peers.View heap) (unitView : units.View heap)
    (getPeers : globals.lookup "PEERS" = some (.ref peers.root)) (getUnits : globals.lookup "UNITS" = some (.ref units.root))
    (nonempty : ∀ cell, candidates cell ≠ ∅) :
    (SetCalls.call program "solve_cands" [.ref address] heap globals).Returns := by
  intro choices
  obtain ⟨finalHeap, finalUsed, answer, _, _, _, path⟩ :=
    result choices globals peers units complete address candidates heap found peerView unitView getPeers getUnits nonempty
      0 heap.length []
  exact ⟨value answer, globals, finalHeap, heap.length, path.finishes (.done rfl)⟩

end
end solvers.«96».solve_cands.search
