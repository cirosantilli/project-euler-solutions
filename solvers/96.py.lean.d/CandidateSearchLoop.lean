import solvers.«96.py.lean.d».CandidateSearchState

namespace solvers.«96».solve_cands.search
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku reduce_cands
set_option Elab.async false
noncomputable section

def Remaining (candidates : Candidates) (cell : Cell) (digits : Finset Digit)
    (heap : Mutable.Heap) : Option Answer → Prop
  | none => ¬ ∃ assignment, Legal assignment ∧ Admits candidates assignment ∧ assignment cell ∈ digits
  | some answer => Correct candidates heap (some answer)

/-- Every failed candidate is removed exactly once; a successful recursive branch returns immediately. -/
theorem loop (choices : ChoiceSchedule) (globals : Mutable.Globals) (peers : PeerTable) (units : UnitTable)
    (address : Nat) (candidates : Candidates) (cell : Cell) (digits : Finset Digit)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (used : Nat)
    (found : View address heap candidates) (peerView : peers.View heap) (unitView : units.View heap)
    (locals : Locals address cell digits env) (included : digits ⊆ candidates cell)
    (recurse : ∀ digit ∈ candidates cell, ∀ childAddress childHeap,
      View childAddress childHeap (Function.update candidates cell {digit}) →
      peers.View childHeap → units.View childHeap →
      Executes choices globals childAddress (Function.update candidates cell {digit}) childHeap)
    (anchors : Nat) (stack : List HeapCalls.Frame) :
    ∃ finalHeap finalUsed answer, heap.length ≤ finalHeap.length ∧
      (∀ index < heap.length, finalHeap[index]? = heap[index]?) ∧
      Remaining candidates cell digits finalHeap answer ∧
      SetCalls.Reaches choices program anchors
        ⟨⟨.block [statement, .return (.constant .none)] env, stack, heap⟩, used⟩
        ⟨⟨.value (value answer), stack, finalHeap⟩, finalUsed⟩ globals := by
  by_cases empty : digits = ∅
  · subst digits
    exact ⟨heap, used, none, le_rfl, fun _ _ => rfl, by simp [Remaining],
      finish choices globals address cell heap env locals used anchors stack⟩
  · obtain ⟨digit, member, _, selected⟩ := bits.choose digits empty env heap globals locals.mask
    let bitEnv := setLocal env "b" (number (2 ^ digit.val))
    let ready := setLocal bitEnv "nxt" (.ref heap.length)
    let copied := clone.work heap candidates cell digit
    have bitLocals : Locals address cell digits bitEnv := locals.set_other "b" _ (by decide) (by decide) (by decide)
    have readyLocals : Locals address cell digits ready := bitLocals.set_other "nxt" _ (by decide) (by decide) (by decide)
    have bitFound : bitEnv "b" = some (number (2 ^ digit.val)) := by simp [bitEnv, setLocal]
    have copiedRun := clone.execute address candidates cell digit bitEnv heap globals found
      bitLocals.candidates bitLocals.index bitFound
    have prepared : SetCalls.Reaches choices program anchors
        ⟨⟨.block (bits.body ++ [statement, .return (.constant .none)]) env, stack, heap⟩, used⟩
        ⟨⟨.block (bits.body.drop 3 ++ [statement, .return (.constant .none)]) ready, stack, copied⟩, used⟩ globals := by
      apply SetCalls.Reaches.of_heap
      exact (selected.append copiedRun) (bits.body.drop 3 ++ [statement, .return (.constant .none)]) anchors stack
    let following := [accept, bits.advance, statement, .return (.constant .none)]
    let continuation := HeapCalls.Frame.assign "res" following ready :: stack
    have invocation : SetCalls.Reaches choices program anchors
        ⟨⟨.block (bits.body.drop 3 ++ [statement, .return (.constant .none)]) ready, stack, copied⟩, used⟩
        ⟨⟨.operation (.call "solve_cands") [.ref heap.length], continuation, copied⟩, used⟩ globals := by
      apply SetCalls.Reaches.of_heap
      apply HeapCalls.Reaches.next rfl
      apply HeapCalls.Reaches.next rfl
      exact Evaluates.arguments_reach (.cons (Evaluates.lookup (by simp [ready, setLocal])) .nil)
        (.call "solve_cands") [] anchors continuation
    have copyFrame : ∀ index < heap.length, copied[index]? = heap[index]? := clone.frame heap candidates cell digit
    obtain ⟨afterHeap, afterUsed, answer, afterLength, afterFrame, correct, called⟩ :=
      recurse digit (included member) heap.length copied (clone.view heap candidates cell digit)
        (peer_frame peers heap copied peerView copyFrame) (unit_frame units heap copied unitView copyFrame)
        used anchors continuation
    let afterEnv := setLocal ready "res" (value answer)
    have originalLength : heap.length ≤ afterHeap.length :=
      (show heap.length ≤ copied.length by simp [copied, clone.work]).trans afterLength
    have originalFrame : ∀ index < heap.length, afterHeap[index]? = heap[index]? := by
      intro index inside
      exact (afterFrame index (by simp [copied, clone.work]; omega) (by omega)).trans (copyFrame index inside)
    have bound : SetCalls.Reaches choices program anchors
        ⟨⟨.value (value answer), continuation, afterHeap⟩, afterUsed⟩
        ⟨⟨.block following afterEnv, stack, afterHeap⟩, afterUsed⟩ globals :=
      SetCalls.Reaches.of_heap (.next rfl .refl) choices afterUsed
    have reached := (enter choices globals address cell digits empty heap env locals
      [.return (.constant .none)] used anchors stack).trans (prepared.trans (invocation.trans (called.trans bound)))
    cases answer with
    | some answer =>
        refine ⟨afterHeap, afterUsed, some answer, originalLength, originalFrame, ?_, ?_⟩
        · exact ⟨correct.1, (branch_complete candidates answer.assignment cell).mpr
            ⟨digit, included member, correct.2.1⟩, correct.2.2⟩
        · exact reached.trans (SetCalls.Reaches.of_heap
            (accept_solution answer afterEnv afterHeap globals (by simp [afterEnv, setLocal, value])
              [bits.advance, statement, .return (.constant .none)] anchors stack) choices afterUsed)
    | none =>
        have afterLocals : Locals address cell digits afterEnv := readyLocals.set_other "res" _ (by decide) (by decide) (by decide)
        have afterBit : afterEnv "b" = some (number (2 ^ digit.val)) := by simp [afterEnv, ready, bitEnv, setLocal]
        let nextEnv := setLocal afterEnv "mask" (number (Masks.encode (digits.erase digit)))
        have nextLocals : Locals address cell (digits.erase digit) nextEnv := by
          constructor
          · simpa [nextEnv, setLocal] using afterLocals.candidates
          · simpa [nextEnv, setLocal] using afterLocals.index
          · simp [nextEnv, setLocal]
        have continued : SetCalls.Reaches choices program anchors
            ⟨⟨.block following afterEnv, stack, afterHeap⟩, afterUsed⟩
            ⟨⟨.block [statement, .return (.constant .none)] nextEnv, stack, afterHeap⟩, afterUsed⟩ globals := by
          apply SetCalls.Reaches.of_heap
          exact ((reject_branch afterEnv afterHeap globals (by simp [afterEnv, setLocal, value])).append
            (bits.next digits digit member afterEnv afterHeap globals afterLocals.mask afterBit))
            [statement, .return (.constant .none)] anchors stack
        obtain ⟨finalHeap, finalUsed, finalAnswer, finalLength, finalFrame, finalCorrect, finished⟩ :=
          loop choices globals peers units address candidates cell (digits.erase digit) afterHeap nextEnv afterUsed
            ((originalFrame address (List.getElem?_eq_some_iff.mp found).1).trans found)
            (peer_frame peers heap afterHeap peerView originalFrame) (unit_frame units heap afterHeap unitView originalFrame)
            nextLocals (fun _ present => included (Finset.mem_of_mem_erase present)) recurse anchors stack
        refine ⟨finalHeap, finalUsed, finalAnswer, originalLength.trans finalLength,
          fun index inside => (finalFrame index (lt_of_lt_of_le inside originalLength)).trans (originalFrame index inside),
          ?_, reached.trans (continued.trans finished)⟩
        cases finalAnswer with
        | some answer => exact finalCorrect
        | none =>
            rintro ⟨assignment, legal, admitted, present⟩
            by_cases same : assignment cell = digit
            · apply correct
              refine ⟨assignment, legal, ?_⟩
              intro current
              by_cases atCell : current = cell
              · subst current
                simp [same]
              · simpa [Function.update_of_ne atCell] using admitted current
            · exact finalCorrect ⟨assignment, legal, admitted, Finset.mem_erase.mpr ⟨same, present⟩⟩
termination_by digits.card
decreasing_by exact Finset.card_erase_lt_of_mem member

end
end solvers.«96».solve_cands.search
