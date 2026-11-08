import solvers.«96.py.lean.d».CandidateOnlyPlaceChecks

namespace solvers.«96».reduce_cands.onlyPlace
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.Sudoku
set_option Elab.async false
noncomputable section

structure Locals (address : Nat) (state : Propagation.State) (env : HeapCalls.Env) : Prop where
  candidates : env "cands" = some (.ref address)
  changed : env "changed" = some (.scalar (.bool state.changed))

def outcome (address : Nat) (result : Option Propagation.State) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (body : List Stmt) (finalEnv : HeapCalls.Env) (finalHeap : Mutable.Heap)
    (globals : Mutable.Globals) : Prop :=
  match result with
  | none => ∀ following, BlockReturns program env (body ++ following) heap (.scalar (.bool false)) finalHeap globals
  | some after => View address finalHeap after.candidates ∧ Locals address after finalEnv ∧
      BlockContinues program env body heap finalEnv finalHeap globals

theorem source_body : places.digitBody = places.digitBody.take 2 ++ [reject, choose] := rfl

/-- One actual digit iteration, including allocation, short-circuit tests, and the candidate write. -/
theorem execute (address unit : Nat) (state : Propagation.State) (cells : List Cell) (digit : Digit)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : View address heap state.candidates) (locals : Locals address state env)
    (unitFound : heap[unit]? = some (.list (peerLoop.encoded cells)))
    (getUnit : env "unit" = some (.ref unit)) (getDigit : env "k" = some (number digit.val)) :
    ∃ finalHeap finalEnv, finalHeap.length = heap.length + 1 ∧
      (∀ index < heap.length, address ≠ index → finalHeap[index]? = heap[index]?) ∧
      (∀ name, name ∉ ["dbit", "places", "changed"] → finalEnv name = env name) ∧
      outcome address (OnlyPlace.inspect state cells digit) env heap places.digitBody finalEnv finalHeap globals := by
  let selected := OnlyPlace.places state.candidates cells digit
  let work := places.work heap selected
  let bits := setLocal env "dbit" (number (2 ^ digit.val))
  let ready := setLocal bits "places" (.ref heap.length)
  have bitsCandidates : bits "cands" = some (.ref address) := by simpa [bits, setLocal] using locals.candidates
  have bitsDigit : bits "dbit" = some (number (2 ^ digit.val)) := by simp [bits, setLocal]
  have prepared : BlockContinues program env (places.digitBody.take 2) heap ready work globals := by
    exact (places.digit_result digit env heap globals getDigit).append
      (BlockContinues.assign (places.result address unit heap state.candidates digit cells bits globals found unitFound
        bitsCandidates (by simpa [bits, setLocal] using getUnit) bitsDigit))
  have view : View address work state.candidates := HeapCollection.prefix_found heap _ address _ found
  have located : work[heap.length]? = some (.list (peerLoop.encoded selected)) := HeapCollection.found heap _ []
  have candidates : ready "cands" = some (.ref address) := by simpa [ready, setLocal] using bitsCandidates
  have lookup : ready "places" = some (.ref heap.length) := by simp [ready, setLocal]
  have bit : ready "dbit" = some (number (2 ^ digit.val)) := by simpa [ready, setLocal] using bitsDigit
  have readyLocals : Locals address state ready := ⟨candidates, by simpa [ready, bits, setLocal] using locals.changed⟩
  have names : ∀ name, name ∉ ["dbit", "places", "changed"] → ready name = env name := by
    intro name outside
    simp only [List.mem_cons, not_or] at outside
    simp [ready, bits, setLocal, outside.1, outside.2.1]
  have length : work.length = heap.length + 1 := by simp [work, HeapCollection.working]
  have frame : ∀ index < heap.length, address ≠ index → work[index]? = heap[index]? := by
    intro index inside _
    exact List.getElem?_append_left inside
  cases placesEq : selected with
  | nil =>
      refine ⟨work, ready, length, frame, names, ?_⟩
      have result : OnlyPlace.inspect state cells digit = none := (OnlyPlace.inspect_none state cells digit).mpr placesEq
      simp only [outcome, result]
      intro following
      rw [source_body, List.append_assoc]
      apply prepared.returns
      apply BlockReturns.branch (flag := true) (empty_result heap.length work [] ready globals (by simpa [placesEq] using located) lookup) rfl
      exact BlockReturns.return_value Evaluates.constant
  | cons cell rest =>
      have nonempty : BlockContinues program ready [reject] work ready work globals :=
        BlockContinues.branch (flag := false)
          (empty_result heap.length work (cell :: rest) ready globals (by simpa [placesEq] using located) lookup) rfl .nil
      cases rest with
      | nil =>
          have one : work[heap.length]? = some (.list (peerLoop.encoded [cell])) := by simpa [placesEq] using located
          have tested := condition_single address heap.length work state.candidates cell digit ready globals view one candidates lookup bit
          by_cases same : state.candidates cell = {digit}
          · have result : OnlyPlace.inspect state cells digit = some state := by simp [OnlyPlace.inspect, show OnlyPlace.places state.candidates cells digit = _ from placesEq, same]
            refine ⟨work, ready, length, frame, names, ?_⟩
            simp only [outcome, result]
            refine ⟨view, readyLocals, ?_⟩
            rw [source_body]
            exact prepared.append (nonempty.append (BlockContinues.branch (flag := false) tested
              (by simp [Mutable.truth, scalarTruth, same]) .nil))
          · let final := write address work state.candidates cell {digit}
            let finalEnv := setLocal ready "changed" (.scalar (.bool true))
            have result : OnlyPlace.inspect state cells digit = some (OnlyPlace.force state cell digit) := by
              simp [OnlyPlace.inspect, show OnlyPlace.places state.candidates cells digit = _ from placesEq, same]
            refine ⟨final, finalEnv, ?_, ?_, ?_, ?_⟩
            · simpa [final, write] using length
            · intro index inside different
              exact (write_frame address index work state.candidates cell {digit} different).trans (frame index inside different)
            · intro name outside
              have away : name ≠ "changed" := by simpa using fun member => outside (by simp [member])
              simpa [finalEnv, setLocal, away] using names name outside
            · simp only [outcome, result]
              refine ⟨write_view address work state.candidates cell {digit} view, ⟨?_, ?_⟩, ?_⟩
              · simpa [finalEnv, setLocal] using candidates
              · simp [finalEnv, setLocal, OnlyPlace.force]
              · rw [source_body]
                exact prepared.append (nonempty.append (BlockContinues.branch (flag := true) tested
                  (by simp [Mutable.truth, scalarTruth, same])
                  (force_result address heap.length work state.candidates cell digit ready globals view one candidates lookup bit)))
      | cons other rest =>
          have result : OnlyPlace.inspect state cells digit = some state := by simp [OnlyPlace.inspect, show OnlyPlace.places state.candidates cells digit = _ from placesEq]
          refine ⟨work, ready, length, frame, names, ?_⟩
          simp only [outcome, result]
          refine ⟨view, readyLocals, ?_⟩
          rw [source_body]
          exact prepared.append (nonempty.append (BlockContinues.branch (flag := false)
            (condition_many heap.length work (cell :: other :: rest) ready globals (by simpa [placesEq] using located) lookup (by simp)) rfl .nil))

end
end solvers.«96».reduce_cands.onlyPlace
