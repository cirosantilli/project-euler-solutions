import solvers.«98.py.lean.d».PartnerLoop

namespace solvers.«98».square_search
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.AnagramSquares
open pattern_signature (program)
open mapped_word (number integerValue)
set_option Elab.async false

def preparationLocals : List String := ["char_to_digit", "digit_to_char", "ok", "$formalsw.unpack", "ch", "d", "sq1"]

theorem preparation_source : bijection.candidateBody.take 6 = bijection.candidateBody.take 4 ++
    [.ifThen (.unaryOp .not (.name "ok")) [.continueLoop] [],
     .assign [.name "sq1"] (.builtinCall .int [.name "sq_str"])] := rfl

/-- A candidate selected by the pattern cache builds both maps, passes the ok guard, and parses its first square. -/
theorem prepared (width : Nat) (word square : String) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (member : square ∈ candidates.values width word)
    (getWord : env "w1" = some (.scalar (.string word))) (getSquare : env "sq_str" = some (.scalar (.string square))) :
    ∃ finalEnv garbage, BlockContinues program env (bijection.candidateBody.take 6) heap finalEnv
      (bijection.work heap (alphabet [] word.toList) (alphabet [] square.toList) garbage) ∧
      finalEnv "char_to_digit" = some (.ref heap.length) ∧
      finalEnv "digit_to_char" = some (.ref (heap.length + 1)) ∧
      finalEnv "ok" = some (.scalar (.bool true)) ∧ finalEnv "sq1" = some (number (integerValue square)) ∧
      (∀ name, name ∉ preparationLocals → finalEnv name = env name) := by
  obtain ⟨checkedEnv, garbage, checked, getForward, getReverse, getOk, kept⟩ := bijection.result word square heap env
    (candidates.matched width word square member) getWord getSquare
  let current := bijection.work heap (alphabet [] word.toList) (alphabet [] square.toList) garbage
  have tested : Evaluates program checkedEnv (.unaryOp .not (.name "ok")) current (.scalar (.bool false)) current :=
    Evaluates.unary (Evaluates.lookup getOk) rfl
  have guard : BlockContinues program checkedEnv [.ifThen (.unaryOp .not (.name "ok")) [.continueLoop] []] current checkedEnv current :=
    BlockContinues.branch (flag := false) tested rfl BlockContinues.nil
  have parsed := mapped_word.integer_result width "sq_str" square current checkedEnv (List.mem_filter.mp member).1
    ((kept "sq_str" (by decide)).trans getSquare)
  refine ⟨setLocal checkedEnv "sq1" (number (integerValue square)), garbage, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [preparation_source]
    exact checked.append (guard.append (BlockContinues.assign parsed))
  · simpa [setLocal] using getForward
  · simpa [setLocal] using getReverse
  · simpa [setLocal] using getOk
  · simp [setLocal]
  · intro name absent
    have notFirst : name ≠ "sq1" := fun same => absent (by simp [preparationLocals, same])
    simp only [setLocal, if_neg notFirst]
    apply kept
    intro member
    exact absent (by simpa [preparationLocals] using List.mem_append_left ["sq1"] member)

end solvers.«98».square_search
