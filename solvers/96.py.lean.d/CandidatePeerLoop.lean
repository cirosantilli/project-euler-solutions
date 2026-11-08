import solvers.«96.py.lean.d».CandidateQueue
import ProjectEulerSolutions.Lib.SudokuPropagation
import lang.py.FormalSW.SetChoiceCalls

namespace solvers.«96».reduce_cands.peerLoop
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku
set_option Elab.async false
noncomputable section

abbrev atom (cell : Cell) : Constant := .int (Encoding.index cell).val
abbrev encoded (cells : List Cell) : List Mutable.Value := cells.map fun cell => .scalar (atom cell)
abbrev keys (cells : List Cell) : List Constant := cells.map atom

structure View (address queue : Nat) (heap : Mutable.Heap) (state : Propagation.State) : Prop where
  candidates : reduce_cands.View address heap state.candidates
  queued : heap[queue]? = some (.list (encoded state.queued))

structure Locals (address queue : Nat) (digit : Digit) (state : Propagation.State) (env : HeapCalls.Env) : Prop where
  candidates : env "cands" = some (.ref address)
  queue : env "queue" = some (.ref queue)
  digit : env "val" = some (number (2 ^ digit.val))
  changed : env "changed" = some (.scalar (.bool state.changed))

structure Frame (address queue : Nat) (before after : Mutable.Heap) : Prop where
  length : after.length = before.length
  outside : ∀ index, address ≠ index → queue ≠ index → after[index]? = before[index]?

theorem Frame.refl (address queue : Nat) (heap : Mutable.Heap) : Frame address queue heap heap :=
  ⟨rfl, fun _ _ _ => rfl⟩

theorem Frame.trans {address queue : Nat} {a b c : Mutable.Heap}
    (first : Frame address queue a b) (second : Frame address queue b c) : Frame address queue a c :=
  ⟨second.length.trans first.length, fun index a b => (second.outside index a b).trans (first.outside index a b)⟩

theorem changed_view (address queue : Nat) (heap : Mutable.Heap) (state : Propagation.State) (other : Cell) (digit : Digit)
    (view : View address queue heap state) (different : address ≠ queue) :
    View address queue (peer.finish address queue heap state.candidates other digit (encoded state.queued))
      (Propagation.reduced state other digit) := by
  constructor
  · exact peer.finish_view address queue heap state.candidates other digit _ view.candidates different
  · have updated := peer.finish_queue address queue heap state.candidates other digit _ view.queued different
    by_cases single : ((state.candidates other).erase digit).card = 1 <;>
      simpa [Propagation.reduced, encoded, single] using updated

theorem changed_frame (address queue : Nat) (heap : Mutable.Heap) (state : Propagation.State) (other : Cell) (digit : Digit) :
    Frame address queue heap (peer.finish address queue heap state.candidates other digit (encoded state.queued)) := by
  constructor
  · unfold peer.finish
    split <;> simp [write]
  · exact fun outside a b => peer.finish_frame address queue outside heap state.candidates other digit _ a b

def outcome (choices : ChoiceSchedule) (globals : Mutable.Globals) (address queue : Nat) (digit : Digit)
    (result : Option Propagation.State) (start : SetCalls.State) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (used : Nat) (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) : Prop :=
  match result with
  | none => SetCalls.Reaches choices program anchors start ⟨⟨.value (.scalar (.bool false)), stack, heap⟩, used⟩ globals
  | some after => View address queue heap after ∧ Locals address queue digit after env ∧
      SetCalls.Reaches choices program anchors start ⟨⟨.block following env, stack, heap⟩, used⟩ globals

abbrev loop (set : Nat) (original remaining : List Cell) : Stmt :=
  .setStepLoop "p" set (keys original) (keys remaining) peerBody []

theorem loop_control (set : Nat) (original remaining : List Cell) :
    Calls.loopContinuation peerBody (loop set original remaining) = loop set original remaining := rfl

theorem run (choices : ChoiceSchedule) (globals : Mutable.Globals) (address queue set : Nat)
    (original remaining : List Cell) (state : Propagation.State) (digit : Digit)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (used : Nat)
    (view : View address queue heap state) (locals : Locals address queue digit state env)
    (different : address ≠ queue) (notCandidates : address ≠ set) (notQueue : queue ≠ set)
    (found : heap[set]? = some (.set (keys original)))
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) :
    ∃ finalHeap finalEnv finalUsed, Frame address queue heap finalHeap ∧
      (∀ name, name ∉ ["p", "changed"] → finalEnv name = env name) ∧
      outcome choices globals address queue digit
        (Propagation.peers state digit (SetCalls.selectionOrder choices used remaining))
        ⟨⟨.block (loop set original remaining :: following) env, stack, heap⟩, used⟩
        finalEnv finalHeap finalUsed following anchors stack := by
  cases remaining with
  | nil =>
      simp only [SetCalls.selectionOrder, Propagation.peers, outcome]
      refine ⟨heap, env, used, .refl _ _ _, fun _ _ => rfl, view, locals, .next ?_ .refl⟩
      simp [keys, SetCalls.step, HeapCalls.step, SetCalls.iteratorStep, found]
  | cons head tail =>
      let index := choices used % (head :: tail).length
      have bound : index < (head :: tail).length := Nat.mod_lt _ (by simp)
      let other := (head :: tail)[index]
      let rest := (head :: tail).eraseIdx index
      let ready := setLocal env "p" (.scalar (atom other))
      have nextLocals : peer.Locals ready address queue other digit := by
        exact ⟨by simpa [ready, setLocal] using locals.candidates, by simp [ready, setLocal, atom, number],
          by simpa [ready, setLocal] using locals.digit, by simpa [ready, setLocal] using locals.queue⟩
      have entered : SetCalls.Reaches choices program anchors
          ⟨⟨.block (loop set original (head :: tail) :: following) env, stack, heap⟩, used⟩
          ⟨⟨.block (peerBody ++ [loop set original rest] ++ following) ready, stack, heap⟩, used + 1⟩ globals := by
        apply SetCalls.Reaches.next (following := ⟨⟨.block (peerBody ++ [loop set original rest] ++ following) ready, stack, heap⟩, used + 1⟩) _ .refl
        have selected : (keys (head :: tail))[index]? = some (atom other) := by
          simp only [keys, List.getElem?_map, List.getElem?_eq_getElem bound]
          rfl
        have erased : (keys (head :: tail)).eraseIdx index = keys rest := List.eraseIdx_map atom _ index
        have selected' : (atom head :: tail.map atom)[choices used % (tail.length + 1)]? = some (atom other) := selected
        have erased' : (atom head :: tail.map atom).eraseIdx (choices used % (tail.length + 1)) = keys rest := erased
        simp [loop, SetCalls.step, HeapCalls.step, SetCalls.iteratorStep, found, keys,
          selected', erased', ready, Calls.loopContinuation, show Calls.loopControl peerBody = false from rfl]
      have order : SetCalls.selectionOrder choices used (head :: tail) =
          other :: SetCalls.selectionOrder choices (used + 1) rest := by rw [SetCalls.selectionOrder]
      by_cases present : digit ∈ state.candidates other
      · by_cases empty : (state.candidates other).erase digit = ∅
        · refine ⟨write address heap state.candidates other ((state.candidates other).erase digit), ready, used + 1,
            ⟨by simp [write], fun outside a _ => write_frame address outside heap state.candidates other _ a⟩,
            ?_, ?_⟩
          · intro name outside
            have notP : name ≠ "p" := fun same => outside (by simp [same])
            simp [ready, setLocal, notP]
          · simp only [order, Propagation.peers, Propagation.remove, if_pos present, if_pos empty, Option.bind_none, outcome]
            exact entered.trans (SetCalls.Reaches.of_heap
              (peer.conflict_result address queue heap state.candidates other digit ready globals view.candidates nextLocals present empty
                (loop set original rest :: following) anchors stack) choices (used + 1))
        · let changedEnv := setLocal ready "changed" (.scalar (.bool true))
          let changedHeap := peer.finish address queue heap state.candidates other digit (encoded state.queued)
          have changedLocals : Locals address queue digit (Propagation.reduced state other digit) changedEnv := by
            constructor
            · simpa [changedEnv, ready, setLocal] using locals.candidates
            · simpa [changedEnv, ready, setLocal] using locals.queue
            · simpa [changedEnv, ready, setLocal] using locals.digit
            · simp [changedEnv, Propagation.reduced, setLocal]
          have frame := changed_frame address queue heap state other digit
          obtain ⟨finalHeap, finalEnv, finalUsed, finalFrame, envFrame, completed⟩ :=
            run choices globals address queue set original rest (Propagation.reduced state other digit) digit changedHeap changedEnv (used + 1)
              (changed_view address queue heap state other digit view different) changedLocals different notCandidates notQueue
              ((frame.outside set notCandidates notQueue).trans found) following anchors stack
          refine ⟨finalHeap, finalEnv, finalUsed, frame.trans finalFrame, ?_, ?_⟩
          · intro name outside
            rw [envFrame name outside]
            have away : name ≠ "p" ∧ name ≠ "changed" := by simpa using outside
            simp [changedEnv, ready, setLocal, away.1, away.2]
          · have executed := peer.changed_result address queue heap state.candidates other digit (encoded state.queued)
              ready globals view.candidates view.queued different nextLocals present empty
            have path := entered.trans (SetCalls.Reaches.of_heap (executed (loop set original rest :: following) anchors stack) choices (used + 1))
            simp only [order, Propagation.peers, Propagation.remove, if_pos present, if_neg empty, Option.bind_some]
            cases result : Propagation.peers (Propagation.reduced state other digit) digit (SetCalls.selectionOrder choices (used + 1) rest) with
            | none =>
                simp only [result, outcome] at completed ⊢
                exact path.trans completed
            | some after =>
                simp only [result, outcome] at completed ⊢
                exact ⟨completed.1, completed.2.1, path.trans completed.2.2⟩
      · have keptLocals : Locals address queue digit state ready := by
          constructor
          · simpa [ready, setLocal] using locals.candidates
          · simpa [ready, setLocal] using locals.queue
          · simpa [ready, setLocal] using locals.digit
          · simpa [ready, setLocal] using locals.changed
        obtain ⟨finalHeap, finalEnv, finalUsed, frame, envFrame, completed⟩ :=
          run choices globals address queue set original rest state digit heap ready (used + 1) view keptLocals
            different notCandidates notQueue found following anchors stack
        refine ⟨finalHeap, finalEnv, finalUsed, frame, ?_, ?_⟩
        · intro name outside
          rw [envFrame name outside]
          have notP : name ≠ "p" := fun same => outside (by simp [same])
          simp [ready, setLocal, notP]
        · have executed := peer.skip address queue heap state.candidates other digit ready globals view.candidates nextLocals present
          have path := entered.trans (SetCalls.Reaches.of_heap (executed (loop set original rest :: following) anchors stack) choices (used + 1))
          simp only [order, Propagation.peers, Propagation.remove, if_neg present, Option.bind_some]
          cases result : Propagation.peers state digit (SetCalls.selectionOrder choices (used + 1) rest) with
          | none =>
              simp only [result, outcome] at completed ⊢
              exact path.trans completed
          | some after =>
              simp only [result, outcome] at completed ⊢
              exact ⟨completed.1, completed.2.1, path.trans completed.2.2⟩
termination_by remaining.length
decreasing_by
  all_goals
    rw [List.length_eraseIdx_of_lt (Nat.mod_lt _ (by simp))]
    simp only [List.length_cons]
    omega

abbrev statement : Stmt := .forIn (.name "p") (.subscript (.name "PEERS") (.name "i")) peerBody []

theorem source : queueBody.drop 2 = [statement] := rfl

theorem execute (choices : ChoiceSchedule) (globals : Mutable.Globals) (address queue set : Nat)
    (original : List Cell) (state : Propagation.State) (digit : Digit)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (used : Nat)
    (view : View address queue heap state) (locals : Locals address queue digit state env)
    (different : address ≠ queue) (notCandidates : address ≠ set) (notQueue : queue ≠ set)
    (found : heap[set]? = some (.set (keys original)))
    (read : Evaluates program env (.subscript (.name "PEERS") (.name "i")) heap (.ref set) heap globals)
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) :
    ∃ finalHeap finalEnv finalUsed, Frame address queue heap finalHeap ∧
      (∀ name, name ∉ ["p", "changed"] → finalEnv name = env name) ∧
      outcome choices globals address queue digit
        (Propagation.peers state digit (SetCalls.selectionOrder choices used original))
        ⟨⟨.block (statement :: following) env, stack, heap⟩, used⟩
        finalEnv finalHeap finalUsed following anchors stack := by
  have entry : SetCalls.Reaches choices program anchors
      ⟨⟨.block (statement :: following) env, stack, heap⟩, used⟩
      ⟨⟨.block (loop set original original :: following) env, stack, heap⟩, used⟩ globals := by
    apply SetCalls.Reaches.next (following := ⟨⟨.expr (.subscript (.name "PEERS") (.name "i")) env,
      .sequence "p" peerBody [] following env :: stack, heap⟩, used⟩) rfl
    apply (SetCalls.Reaches.of_heap (read anchors _) choices used).trans
    exact .next (by simp [SetCalls.step, HeapCalls.step, HeapCalls.run, Mutable.sequence, found, SetCalls.iteratorStep, Mutable.unsupported]) .refl
  obtain ⟨finalHeap, finalEnv, finalUsed, frame, envFrame, result⟩ := run choices globals address queue set original original
    state digit heap env used view locals different notCandidates notQueue found following anchors stack
  refine ⟨finalHeap, finalEnv, finalUsed, frame, envFrame, ?_⟩
  cases done : Propagation.peers state digit (SetCalls.selectionOrder choices used original) with
  | none =>
      simp only [done, outcome] at result ⊢
      exact entry.trans result
  | some after =>
      simp only [done, outcome] at result ⊢
      exact ⟨result.1, result.2.1, entry.trans result.2.2⟩

end
end solvers.«96».reduce_cands.peerLoop
