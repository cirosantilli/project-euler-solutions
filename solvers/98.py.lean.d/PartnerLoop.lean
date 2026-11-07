import solvers.«98.py.lean.d».Partner

namespace solvers.«98».mapped_word
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.AnagramSquares
open pattern_signature (program)
open data.«project-euler-statements».data.py.project_euler_statements.p98.mapping (table)
open word_patterns (text)
set_option Elab.async false

def scan (width : Nat) (word digits : List Char) (others : List String) (firstSquare best : Int) : Int :=
  others.foldl (fun current other => score width word digits other firstSquare current) best

def localNames : List String := ["w2", "num2", "sq2", "best"]

/-- The complete partner loop scans its finite list, including leading-zero skips, without changing cached objects. -/
theorem loop (width : Int) (word digits : List Char) (before remaining : List String) (firstSquare best : Int)
    (heap : Mutable.Heap) (partners mapping outer recordAddress : Nat) (env : HeapCalls.Env)
    (matched : labels [] word = labels [] digits)
    (valid : ∀ other ∈ remaining, other.toList ≠ [] ∧ ∀ letter ∈ other.toList, letter ∈ word)
    (foundPartners : heap[partners]? = some (.list ((before ++ remaining).map text)))
    (foundMapping : heap[mapping]? = some (.dict (table (alphabet [] word) (alphabet [] digits))))
    (stored : square_cache.Stored outer heap width recordAddress)
    (getMapping : env "char_to_digit" = some (.ref mapping)) (getRecord : env "sqinfo" = some (.ref recordAddress))
    (getFirst : env "sq1" = some (number firstSquare)) (getBest : env "best" = some (number best)) :
    ∃ finalEnv extra, BlockContinues program env
      [.sequenceStepLoop "w2" (.ref partners) before.length body []] heap finalEnv (heap ++ extra) ∧
      finalEnv "best" = some (number (scan width.toNat word digits remaining firstSquare best)) ∧
      (∀ name, name ∉ localNames → finalEnv name = env name) := by
  induction remaining generalizing before best heap env with
  | nil =>
      refine ⟨env, [], ?_, getBest, by intros; rfl⟩
      simp only [List.append_nil]
      apply BlockContinues.sequence_done (items := before.map text)
      · simp [Mutable.sequence, sequenceValue, foundPartners]
      · simp
      · exact BlockContinues.nil
  | cons other rest ih =>
      let ready := setLocal env "w2" (text other)
      let next := Stmt.sequenceStepLoop "w2" (.ref partners) (before.length + 1) body []
      obtain ⟨stepEnv, stepExtra, stepRan, stepAnswer, stepKept⟩ := result width word digits other firstSquare best
        heap mapping outer recordAddress ready next matched (valid other (by simp)).1 (valid other (by simp)).2
        foundMapping stored (by simpa [ready, setLocal] using getMapping) (by simp [ready, setLocal])
        (by simpa [ready, setLocal] using getRecord) (by simpa [ready, setLocal] using getFirst)
        (by simpa [ready, setLocal] using getBest)
      obtain ⟨finalEnv, extra, continued, answer, kept⟩ := ih (before ++ [other])
        (score width.toNat word digits other firstSquare best) (heap ++ stepExtra) stepEnv
        (fun item member => valid item (by simp [member]))
        (by simpa only [List.append_assoc, List.singleton_append] using HeapCollection.prefix_found heap stepExtra partners _ foundPartners)
        (HeapCollection.prefix_found heap stepExtra mapping _ foundMapping) (stored.append stepExtra)
        ((stepKept "char_to_digit" (by decide)).trans (by simpa [ready, setLocal] using getMapping))
        ((stepKept "sqinfo" (by decide)).trans (by simpa [ready, setLocal] using getRecord))
        ((stepKept "sq1" (by decide)).trans (by simpa [ready, setLocal] using getFirst)) stepAnswer
      refine ⟨finalEnv, stepExtra ++ extra, ?_, answer, ?_⟩
      · apply BlockContinues.sequence_step (items := (before ++ other :: rest).map text) (value := text other)
        · simp [Mutable.sequence, sequenceValue, foundPartners]
        · simp
        · intro following anchors stack
          have finish := (stepRan following anchors stack).trans (by
            simpa only [next, List.length_append, List.length_singleton] using continued following anchors stack)
          simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, body, Bool.false_or,
            Bool.true_or, if_true, List.append_assoc] using finish
      · intro name absent
        rw [kept name absent, stepKept name (fun member => absent (List.mem_cons_of_mem _ member))]
        simp [ready, setLocal, show name ≠ "w2" from fun same => absent (by simp [localNames, same])]

theorem loop_result (width : Int) (word digits : List Char) (others : List String) (firstSquare best : Int)
    (heap : Mutable.Heap) (partners mapping outer recordAddress : Nat) (env : HeapCalls.Env)
    (matched : labels [] word = labels [] digits)
    (valid : ∀ other ∈ others, other.toList ≠ [] ∧ ∀ letter ∈ other.toList, letter ∈ word)
    (foundPartners : heap[partners]? = some (.list (others.map text)))
    (foundMapping : heap[mapping]? = some (.dict (table (alphabet [] word) (alphabet [] digits))))
    (stored : square_cache.Stored outer heap width recordAddress)
    (getPartners : env "others" = some (.ref partners)) (getMapping : env "char_to_digit" = some (.ref mapping))
    (getRecord : env "sqinfo" = some (.ref recordAddress)) (getFirst : env "sq1" = some (number firstSquare))
    (getBest : env "best" = some (number best)) :
    ∃ finalEnv extra, BlockContinues program env (bijection.candidateBody.drop 6) heap finalEnv (heap ++ extra) ∧
      finalEnv "best" = some (number (scan width.toNat word digits others firstSquare best)) ∧
      (∀ name, name ∉ localNames → finalEnv name = env name) := by
  obtain ⟨finalEnv, extra, executed, answer, kept⟩ := loop width word digits [] others firstSquare best heap partners mapping outer recordAddress env
    matched valid foundPartners foundMapping stored getMapping getRecord getFirst getBest
  refine ⟨finalEnv, extra, ?_, answer, kept⟩
  apply BlockContinues.for_sequence (items := others.map text) (by intros; simp) (Evaluates.lookup getPartners)
  · simp [Mutable.sequence, foundPartners]
  · exact executed

end solvers.«98».mapped_word
