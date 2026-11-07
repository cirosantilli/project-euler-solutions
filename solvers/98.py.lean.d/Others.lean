import solvers.«98.py.lean.d».Candidates

namespace solvers.«98».others
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open pattern_signature (program)
open word_patterns (text)
set_option Elab.async false

def values (words : List String) (omitted : String) : List String := words.filter (· ≠ omitted)
abbrev condition : Expr := .compare (.name "w") [(.notEq, .name "w1")]
abbrev expression : Expr := .listComp (.name "w") [("w", .name "g", [condition])]
abbrev body (address : Nat) : List Stmt := comprehensionFilters [condition] [.comprehensionAppend address (.name "w")]
abbrev work (base : Mutable.Heap) (output : List String) := HeapCollection.working base (output.map text) []

theorem source : candidates.wordBody[3]? = some (.assign [.name "others"] expression) := rfl

theorem values_cons (word : String) (rest : List String) (omitted : String) :
    values (word :: rest) omitted = (if word ≠ omitted then [word] else []) ++ values rest omitted := by
  by_cases different : word ≠ omitted <;> simp [values, different]

theorem collect (base : Mutable.Heap) (output : List String) (word omitted : String) (env : HeapCalls.Env)
    (getWord : env "w" = some (text word)) (getOmitted : env "w1" = some (text omitted)) :
    BlockContinues program env (body base.length) (work base output) env
      (work base (output ++ if word ≠ omitted then [word] else [])) := by
  have tested : Evaluates program env condition (work base output) (.scalar (.bool (decide (word ≠ omitted)))) (work base output) := by
    apply Evaluates.compare (Evaluates.lookup getWord) (Evaluates.lookup getOmitted)
    simp [HeapCalls.compare, scalarCompare]
  by_cases different : word ≠ omitted
  · simp only [if_pos different]
    apply BlockContinues.branch (flag := true) tested (by simp [Mutable.truth, scalarTruth, different])
    have appended := BlockContinues.comprehension_append (module := program) (globals := []) (Evaluates.lookup getWord)
      (HeapCollection.found base (output.map text) [])
    simpa only [work, HeapCollection.store, List.map_append, List.map_cons, List.map_nil] using appended
  · simp only [if_neg different, List.append_nil]
    exact BlockContinues.branch (flag := false) tested (by simp [Mutable.truth, scalarTruth, different]) BlockContinues.nil

theorem loop (base : Mutable.Heap) (address : Nat) (before remaining output : List String) (omitted : String) (env : HeapCalls.Env)
    (found : base[address]? = some (.list ((before ++ remaining).map text))) (getOmitted : env "w1" = some (text omitted)) :
    ∃ finalEnv, BlockContinues program env [.sequenceStepLoop "w" (.ref address) before.length (body base.length) []]
      (work base output) finalEnv (work base (output ++ values remaining omitted)) := by
  induction remaining generalizing before output env with
  | nil =>
      refine ⟨env, ?_⟩
      simp only [values, List.filter_nil, List.append_nil]
      apply BlockContinues.sequence_done (items := before.map text)
      · have kept := HeapCollection.prefix_found base [.list (output.map text)] address _ found
        simp [Mutable.sequence, sequenceValue, work, HeapCollection.working, kept]
      · simp
      · exact BlockContinues.nil
  | cons word rest ih =>
      let ready := setLocal env "w" (text word)
      have keepOmitted : ready "w1" = some (text omitted) := by simpa [ready, setLocal] using getOmitted
      have emitted := collect base output word omitted ready (by simp [ready, setLocal]) keepOmitted
      obtain ⟨finalEnv, continued⟩ := ih (before ++ [word]) (output ++ if word ≠ omitted then [word] else []) ready
        (by simpa only [List.append_assoc, List.singleton_append] using found) keepOmitted
      refine ⟨finalEnv, ?_⟩
      apply BlockContinues.sequence_step (items := (before ++ word :: rest).map text) (value := text word)
      · have kept := HeapCollection.prefix_found base [.list (output.map text)] address _ found
        simp [Mutable.sequence, sequenceValue, work, HeapCollection.working, kept]
      · simp
      · simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, body, comprehensionFilters,
          Bool.false_or, Bool.false_eq_true, if_false, values_cons, List.append_assoc,
          List.length_append, List.length_singleton] using emitted.append continued

/-- Exclude every occurrence of the current word, while preserving the order and duplicates of distinct partners. -/
theorem result (words : List String) (omitted : String) (heap : Mutable.Heap) (address : Nat) (env : HeapCalls.Env)
    (found : heap[address]? = some (.list (words.map text))) (getGroup : env "g" = some (.ref address))
    (getOmitted : env "w1" = some (text omitted)) :
    Evaluates program env expression heap (.ref heap.length) (work heap (values words omitted)) := by
  let inner := comprehensionEnv env [("w", .name "g", [condition])]
  obtain ⟨finalEnv, executed⟩ := loop heap address [] words [] omitted inner (by simpa using found)
    (by simpa [inner, comprehensionEnv] using getOmitted)
  apply Evaluates.list_comp_sequence (finalEnv := finalEnv) (items := words.map text)
    (by intros; simp) (Evaluates.lookup getGroup)
  · simp [Mutable.sequence, found]
  · exact executed

end solvers.«98».others
