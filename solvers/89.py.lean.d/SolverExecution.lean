import solvers.«89.py.lean.d».SolverState

namespace solvers.«89».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions RomanNumerals
set_option Elab.async false
set_option maxHeartbeats 1200000

/-- The line iterator preserves the input list while accumulating the exact signed savings. -/
theorem loop (lines remaining : List String) (valid : ∀ text ∈ remaining, ValidText text)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (symbols : SymbolTable heap globals)
    (tokens : EncodingTable heap globals) (address : Nat)
    (found : heap[address]? = some (.list (lines.map textValue))) (position : Nat)
    (suffix : lines.drop position = remaining) (total : Int) (env : HeapCalls.Env)
    (getTotal : env "saved" = some (.scalar (.int total))) :
    ∃ extra, BlockReturns program env
      [.sequenceStepLoop "s" (.ref address) position body [], code[2]'(by decide)] heap
      (.scalar (.int (total + (remaining.map saved).sum))) (heap ++ extra) globals := by
  induction remaining generalizing position total heap env with
  | nil =>
      have done : lines[position]? = none := List.getElem?_eq_none_iff.mpr (List.drop_eq_nil_iff.mp suffix)
      refine ⟨[], ?_⟩
      simp only [List.map_nil, List.sum_nil, Int.add_zero, List.append_nil]
      apply BlockReturns.sequence_done (items := lines.map textValue)
      · simp [sequenceValue, Mutable.sequence, found]
      · simp [done]
      · exact BlockReturns.return_value (Evaluates.lookup getTotal)
  | cons text rest ih =>
      have item : lines[position]? = some text := by
        have head := congrArg (fun xs : List String => xs[0]?) suffix
        simpa using head
      have tail : lines.drop (position + 1) = rest := by
        have dropped := congrArg (List.drop 1) suffix
        simpa [List.drop_drop, Nat.add_comm] using dropped
      let bound := setLocal env "s" (textValue text)
      obtain ⟨allocated, summed, iterated, nextTotal⟩ := iteration text (valid text (by simp)) total heap globals
        symbols tokens bound (by simp [bound, setLocal]) (by simpa [bound, setLocal] using getTotal)
      have preserved := HeapCollection.prefix_found heap allocated address _ found
      obtain ⟨extra, continued⟩ := ih (fun s member => valid s (by simp [member]))
        (heap ++ allocated) (symbols.append allocated) (tokens.append allocated) preserved
        (position + 1) tail (total + saved text) summed nextTotal
      refine ⟨allocated ++ extra, ?_⟩
      apply BlockReturns.sequence_step (items := lines.map textValue) (value := textValue text)
      · simp [sequenceValue, Mutable.sequence, found]
      · simp [item]
      · have restOfLoop := iterated.returns continued
        simpa only [Calls.loopContinuation, show Calls.loopControl body = false from rfl, if_false,
          List.map_cons, List.sum_cons, Int.add_assoc, List.append_assoc] using restOfLoop

/-- The complete optimized solver has the reference savings value for every valid input list. -/
theorem result (lines : List String) (valid : ∀ text ∈ lines, ValidText text)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (symbols : SymbolTable heap globals)
    (tokens : EncodingTable heap globals) (address : Nat)
    (found : heap[address]? = some (.list (lines.map textValue))) :
    ∃ extra, ReturnsValue program "solve" [.ref address] heap
      (.scalar (.int (RomanNumerals.savings lines))) (heap ++ extra) globals := by
  let args := ((findFunctionIn program.body "solve").getD ([], [])).1
  let source := ((findFunctionIn program.body "solve").getD ([], [])).2
  let entry := functionEnv args source [.ref address] globals
  let initialized := setLocal entry "saved" (.scalar (.int 0))
  have getInput : entry "lines" = some (.ref address) := by cases globals <;> rfl
  obtain ⟨extra, scanned⟩ := loop lines lines valid heap globals symbols tokens address found 0 rfl 0 initialized
    (by simp [initialized, setLocal])
  refine ⟨extra, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.for_sequence (items := lines.map textValue) (middle := heap)
    (by intros; simp) (Evaluates.lookup (by simpa [initialized, setLocal] using getInput))
  · simp [Mutable.sequence, found]
  · simpa only [Int.zero_add, RomanNumerals.savings, saved] using scanned

end solvers.«89».solve
