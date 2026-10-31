import solvers.«89.py.lean.d».EncoderState

namespace solvers.«89».int_to_min_roman
open FormalSW.PyAst HeapCalls ProjectEulerSolutions RomanNumerals
set_option Elab.async false
set_option maxHeartbeats 1800000

/-- The tuple table is read from the caller's heap; only the private result list is mutated. -/
theorem table_loop (base : Mutable.Heap) (tableAddress : Nat) (entries remaining : List Entry)
    (found : base[tableAddress]? = some (.tuple (entries.map Entry.reference)))
    (valid : ∀ entry ∈ remaining, entry.Valid base) (position : Nat)
    (suffix : entries.drop position = remaining) (n : Nat) (output : List String) (garbage : Mutable.Heap)
    (globals : Mutable.Globals) (env : HeapCalls.Env) (getN : env "x" = some (number n))
    (getOutput : env "res" = some (.ref base.length)) :
    ∃ finalOutput finalGarbage,
      BlockReturns program env
        [.unpackStepLoop ["val", "sym"] (.ref tableAddress) position loopCode [], code[3]'(by decide)]
        (working base output garbage)
        (textValue (String.ofList (output.flatMap String.toList ++ encode (remaining.map Entry.model) n)))
        (working base finalOutput finalGarbage) globals := by
  induction remaining generalizing position n output garbage env with
  | nil =>
      have done : entries[position]? = none := List.getElem?_eq_none_iff.mpr (List.drop_eq_nil_iff.mp suffix)
      refine ⟨output, garbage, ?_⟩
      simp only [List.map_nil, encode, List.append_nil]
      apply BlockReturns.unpack_done (items := entries.map Entry.reference)
      · have table := HeapCollection.prefix_found base (.list (output.map textValue) :: garbage) tableAddress _ found
        simp [sequenceValue, Mutable.sequence, working, HeapCollection.working, table]
      · simp [done]
      · exact finish base output garbage globals env getOutput
  | cons entry rest ih =>
      have current := valid entry (by simp)
      have item : entries[position]? = some entry := by
        have head := congrArg (fun xs : List Entry => xs[0]?) suffix
        simpa using head
      have tail : entries.drop (position + 1) = rest := by
        have dropped := congrArg (List.drop 1) suffix
        simpa [List.drop_drop, Nat.add_comm] using dropped
      let bound := assignNames env ["val", "sym"] [number entry.denomination, textValue entry.token]
      have boundN : bound "x" = some (number n) := by simpa [bound, assignNames, setLocal] using getN
      have boundOutput : bound "res" = some (.ref base.length) := by
        simpa [bound, assignNames, setLocal] using getOutput
      have boundD : bound "val" = some (number entry.denomination) := by simp [bound, assignNames, setLocal]
      have tested : Evaluates program bound (.compare (.name "x") [(.eq, .constant (.int 0))])
          (working base output garbage) (.scalar (.bool (decide (n = 0)))) (working base output garbage) globals := by
        apply Evaluates.compare (Evaluates.lookup boundN) Evaluates.constant
        simp [HeapCalls.compare, scalarCompare, evalCmpOp]
      have entered (answer : Mutable.Value) (finalHeap : Mutable.Heap)
          (executed : BlockReturns program bound
            (loopCode ++ [.loopResume (.unpackStepLoop ["val", "sym"] (.ref tableAddress) (position + 1) loopCode [])] ++
              [code[3]'(by decide)]) (working base output garbage) answer finalHeap globals) :
          BlockReturns program env
            [.unpackStepLoop ["val", "sym"] (.ref tableAddress) position loopCode [], code[3]'(by decide)]
            (working base output garbage) answer finalHeap globals := by
        apply BlockReturns.unpack_step (items := entries.map Entry.reference)
          (value := .ref entry.address) (values := [number entry.denomination, textValue entry.token])
        · have table := HeapCollection.prefix_found base (.list (output.map textValue) :: garbage) tableAddress _ found
          simp [sequenceValue, Mutable.sequence, working, HeapCollection.working, table]
        · simp [item, Entry.reference]
        · have pair := HeapCollection.prefix_found base (.list (output.map textValue) :: garbage) entry.address _ current.found
          simp [Mutable.sequence, working, HeapCollection.working, pair]
        · rfl
        · exact executed
      by_cases zero : n = 0
      · refine ⟨output, garbage, ?_⟩
        simp only [zero, encode_zero, List.append_nil]
        apply entered
        apply BlockReturns.branch (flag := true) tested (by simp only [zero]; rfl)
        apply BlockReturns.break_loop (by rfl)
        exact finish base output garbage globals bound boundOutput
      · let divided := dividedEnv bound n entry.denomination
        let nextGarbage := garbage ++ [quotientObject n entry.denomination]
        have dividedN : divided "x" = some (number (n % entry.denomination)) := by
          simp [divided, assignNames, setLocal]
        have dividedOutput : divided "res" = some (.ref base.length) := by
          simpa [divided, assignNames, setLocal] using boundOutput
        have dividedCount : divided "k" = some (number (n / entry.denomination)) := by
          simp [divided, assignNames, setLocal]
        have dividedToken : divided "sym" = some (textValue entry.token) := by
          simp [divided, bound, assignNames, setLocal]
        have division : BlockContinues program bound [loopCode[1]'(by decide)]
            (working base output garbage) divided (working base output nextGarbage) globals := by
          simpa only [HeapCollection.append] using
            divide n entry.denomination current.positive (working base output garbage) globals bound boundN boundD
        by_cases quotientZero : n / entry.denomination = 0
        · obtain ⟨finalOutput, finalGarbage, continued⟩ := ih (fun e mem => valid e (by simp [mem]))
            (position + 1) tail (n % entry.denomination) output nextGarbage divided dividedN dividedOutput
          refine ⟨finalOutput, finalGarbage, ?_⟩
          simp only [List.map_cons, encode, Entry.model, quotientZero, copies_zero, List.nil_append]
          apply entered
          apply BlockReturns.branch (flag := false) tested (by simp only [zero]; rfl)
          apply division.returns
          apply BlockReturns.branch (flag := false) (Evaluates.lookup dividedCount)
            (by simp [Mutable.truth, scalarTruth, quotientZero])
          exact BlockReturns.loop_resume continued
        · let chunk := String.join (List.replicate (n / entry.denomination) entry.token)
          obtain ⟨finalOutput, finalGarbage, continued⟩ := ih (fun e mem => valid e (by simp [mem]))
            (position + 1) tail (n % entry.denomination) (output ++ [chunk]) nextGarbage divided dividedN dividedOutput
          refine ⟨finalOutput, finalGarbage, ?_⟩
          apply entered
          apply BlockReturns.branch (flag := false) tested (by simp only [zero]; rfl)
          apply division.returns
          apply (emit entry.token (n / entry.denomination) (Nat.pos_of_ne_zero quotientZero) base output nextGarbage globals divided
            dividedCount dividedToken dividedOutput).returns
          apply BlockReturns.loop_resume
          simpa only [List.map_cons, Entry.model, encode, List.flatMap_append, List.flatMap_cons,
            List.flatMap_nil, List.append_nil, chunk, repeated_characters, List.append_assoc] using continued

end solvers.«89».int_to_min_roman
