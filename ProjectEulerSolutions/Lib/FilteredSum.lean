import lang.py.FormalSW.SumCalls

/-! Pure filtered sums retain the original sequence and evaluate each predicate lazily. -/
namespace ProjectEulerSolutions.FilteredSum
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

set_option Elab.async false

variable {α : Type} {program : Module} {values : List α} {encode : α → Mutable.Value}
  {predicate : α → Bool} {number : α → Int} {source : SequenceSource} {heap : Mutable.Heap}
  {name : String} {element test : Expr} {position : Nat} {env : HeapCalls.Env} {initial : Int}

theorem loop (remaining : List α) (privateName : name ≠ sumAccumulator)
    (sequence : Mutable.sequence (sequenceValue source) heap = some (.ok (values.map encode), heap))
    (tested : ∀ env item, env name = some (encode item) →
      Evaluates program env test heap (.scalar (.bool (predicate item))) heap)
    (evaluated : ∀ env item, env name = some (encode item) → predicate item = true →
      Evaluates program env element heap (.scalar (.int (number item))) heap)
    (suffix : values.drop position = remaining)
    (accumulator : env sumAccumulator = some (.scalar (.int initial))) :
    BlockReturns program env
      [.sequenceStepLoop name source position (comprehensionFilters [test] (sumBody element [])) [],
        .return (.name sumAccumulator)] heap
      (.scalar (.int (remaining.foldl (fun total item =>
        if predicate item then total + number item else total) initial))) heap := by
  induction remaining generalizing position env initial with
  | nil =>
      apply BlockReturns.sequence_done sequence
      · have missing : values[position]? = none := by
          simpa [suffix] using (List.head?_drop (l := values) (i := position)).symm
        simp [missing]
      · exact BlockReturns.return_value (Evaluates.lookup accumulator)
  | cons head tail ih =>
      have item : values[position]? = some head := by
        simpa [suffix] using (List.head?_drop (l := values) (i := position)).symm
      let bound := setLocal env name (encode head)
      have saved : bound sumAccumulator = some (.scalar (.int initial)) := by
        simpa [bound, setLocal, Ne.symm privateName] using accumulator
      have headTest := tested bound head (by simp [bound, setLocal])
      apply BlockReturns.sequence_step (value := encode head) sequence (by simp [item])
      cases accepted : predicate head with
      | false =>
          simp only [List.foldl_cons, accepted, Bool.false_eq_true, if_false]
          apply BlockReturns.branch (by simpa only [accepted] using headTest) rfl
          have rest := ih (position := position+1) (env := bound) (initial := initial)
            (by simp [List.drop_add_one_eq_tail_drop, suffix]) saved
          simpa [comprehensionFilters, sumBody, Calls.loopContinuation, Calls.loopControl,
            Calls.loopControlStmt] using rest
      | true =>
          simp only [List.foldl_cons, accepted, if_true]
          apply BlockReturns.branch (by simpa only [accepted] using headTest) rfl
          apply BlockReturns.assign (value := .scalar (.int (initial+number head))) (middle := heap)
          · exact Evaluates.binary (Evaluates.lookup saved)
              (evaluated bound head (by simp [bound, setLocal]) accepted) rfl
          · have rest := ih (position := position+1)
              (env := setLocal bound sumAccumulator (.scalar (.int (initial+number head))))
              (initial := initial+number head) (by simp [List.drop_add_one_eq_tail_drop, suffix])
              (by simp [setLocal])
            simpa [comprehensionFilters, sumBody, Calls.loopContinuation, Calls.loopControl,
              Calls.loopControlStmt] using rest

end ProjectEulerSolutions.FilteredSum
