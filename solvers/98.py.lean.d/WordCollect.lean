import solvers.«98.py.lean.d».WordStorage
import solvers.«98.py.lean.d».Lengths

namespace solvers.«98».word_patterns
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open StringValueBuckets (Bucket)
open pattern_signature (program)
set_option Elab.async false

def allWords (words : List String) (buckets : List Bucket) : List String := buckets.flatMap (lengths.wordsIn words)
abbrev groupBody : List Stmt := [.forIn (.name "w") (.name "g") body []]
abbrev groupLoop : Stmt := .forIn (.name "g") (.name "anagram_groups") groupBody []

theorem source : (((findFunctionIn program.body "solve").getD ([], [])).2)[7]? = some groupLoop := rfl

theorem word_loop (base : Mutable.Heap) (address : Nat) (before remaining : List String) (rows : Rows) (garbage : Mutable.Heap)
    (env : HeapCalls.Env) (found : base[address]? = some (.list ((before ++ remaining).map text)))
    (getDictionary : env "word_pat" = some (.ref base.length)) (valid : Valid (work base rows garbage) rows) :
    ∃ finalEnv finalRows finalGarbage,
      BlockContinues program env [.sequenceStepLoop "w" (.ref address) before.length body []]
        (work base rows garbage) finalEnv (work base finalRows finalGarbage) ∧ Valid (work base finalRows finalGarbage) finalRows ∧
      (∀ word, (KeyedReferences.lookup finalRows word).isSome ↔ word ∈ remaining ∨ (KeyedReferences.lookup rows word).isSome) ∧
      (∀ name, name ≠ "w" → finalEnv name = env name) := by
  induction remaining generalizing before rows garbage env with
  | nil =>
      refine ⟨env, rows, garbage, ?_, valid, by simp, by intros; rfl⟩
      apply BlockContinues.sequence_done (items := before.map text)
      · have kept := HeapCollection.prefix_found base (.dict (KeyedReferences.entries KeyedReferences.strings rows) :: garbage) address _ found
        simp [Mutable.sequence, sequenceValue, work, KeyedReferences.work, kept]
      · simp
      · exact BlockContinues.nil
  | cons word rest ih =>
      let ready := setLocal env "w" (text word)
      obtain ⟨extra, patternAddress, executed, stepValid⟩ := insert base rows garbage word ready (by simp [ready, setLocal])
        (by simpa [ready, setLocal] using getDictionary) valid
      obtain ⟨finalEnv, finalRows, finalGarbage, continued, finalValid, coverage, kept⟩ :=
        ih (before ++ [word]) (KeyedReferences.insert rows word patternAddress) (garbage ++ extra) ready
          (by simpa only [List.append_assoc, List.singleton_append] using found)
          (by simpa [ready, setLocal] using getDictionary) stepValid
      refine ⟨finalEnv, finalRows, finalGarbage, ?_, finalValid, ?_, ?_⟩
      · apply BlockContinues.sequence_step (items := (before ++ word :: rest).map text) (value := text word)
        · have kept := HeapCollection.prefix_found base (.dict (KeyedReferences.entries KeyedReferences.strings rows) :: garbage) address _ found
          simp [Mutable.sequence, sequenceValue, work, KeyedReferences.work, kept]
        · simp
        · simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, body,
            Bool.false_or, Bool.false_eq_true, if_false, List.length_append, List.length_singleton] using executed.append continued
      · intro query
        rw [coverage query, KeyedReferences.lookup_insert]
        by_cases same : query = word <;> simp [same, or_comm]
      · intro name different
        rw [kept name different]
        simp [ready, setLocal, different]

theorem group_loop (words : List String) (base : Mutable.Heap) (address : Nat) (before remaining : List Bucket)
    (rows : Rows) (garbage : Mutable.Heap) (env : HeapCalls.Env)
    (found : base[address]? = some (.list ((before ++ remaining).map groups.reference)))
    (groupsFound : ∀ bucket ∈ remaining, base[bucket.address]? = some (.list ((lengths.wordsIn words bucket).map text)))
    (getDictionary : env "word_pat" = some (.ref base.length)) (valid : Valid (work base rows garbage) rows) :
    ∃ finalEnv finalRows finalGarbage,
      BlockContinues program env [.sequenceStepLoop "g" (.ref address) before.length groupBody []]
        (work base rows garbage) finalEnv (work base finalRows finalGarbage) ∧ Valid (work base finalRows finalGarbage) finalRows ∧
      (∀ word, (KeyedReferences.lookup finalRows word).isSome ↔ word ∈ allWords words remaining ∨ (KeyedReferences.lookup rows word).isSome) ∧
      (∀ name, name ∉ ["g", "w"] → finalEnv name = env name) := by
  induction remaining generalizing before rows garbage env with
  | nil =>
      refine ⟨env, rows, garbage, ?_, valid, by simp [allWords], by intros; rfl⟩
      apply BlockContinues.sequence_done (items := before.map groups.reference)
      · have kept := HeapCollection.prefix_found base (.dict (KeyedReferences.entries KeyedReferences.strings rows) :: garbage) address _ found
        simp [Mutable.sequence, sequenceValue, work, KeyedReferences.work, kept]
      · simp
      · exact BlockContinues.nil
  | cons bucket rest ih =>
      let ready := setLocal env "g" (groups.reference bucket)
      have row := groupsFound bucket (by simp)
      obtain ⟨wordEnv, wordRows, wordGarbage, scanned, wordValid, wordCoverage, wordKept⟩ :=
        word_loop base bucket.address [] (lengths.wordsIn words bucket) rows garbage ready (by simpa using row)
          (by simpa [ready, setLocal] using getDictionary) valid
      have looped : BlockContinues program ready groupBody (work base rows garbage) wordEnv (work base wordRows wordGarbage) := by
        apply BlockContinues.for_sequence (value := groups.reference bucket) (items := (lengths.wordsIn words bucket).map text)
          (by intros; simp) (Evaluates.lookup (by simp [ready, setLocal]))
        · have kept := HeapCollection.prefix_found base (.dict (KeyedReferences.entries KeyedReferences.strings rows) :: garbage) bucket.address _ row
          simp [Mutable.sequence, groups.reference, work, KeyedReferences.work, kept]
        · exact scanned
      obtain ⟨finalEnv, finalRows, finalGarbage, continued, finalValid, coverage, kept⟩ :=
        ih (before ++ [bucket]) wordRows wordGarbage wordEnv
          (by simpa only [List.append_assoc, List.singleton_append] using found)
          (fun next member => groupsFound next (by simp [member]))
          ((wordKept "word_pat" (by decide)).trans (by simpa [ready, setLocal] using getDictionary)) wordValid
      refine ⟨finalEnv, finalRows, finalGarbage, ?_, finalValid, ?_, ?_⟩
      · apply BlockContinues.sequence_step (items := (before ++ bucket :: rest).map groups.reference) (value := groups.reference bucket)
        · have kept := HeapCollection.prefix_found base (.dict (KeyedReferences.entries KeyedReferences.strings rows) :: garbage) address _ found
          simp [Mutable.sequence, sequenceValue, work, KeyedReferences.work, kept]
        · simp
        · simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, groupBody, body,
            Bool.false_or, Bool.false_eq_true, if_false, List.length_append, List.length_singleton] using looped.append continued
      · intro query
        rw [coverage query, wordCoverage query]
        simp [allWords, or_assoc, or_left_comm]
      · intro name different
        have notG : name ≠ "g" := fun same => different (by simp [same])
        have notW : name ≠ "w" := fun same => different (by simp [same])
        rw [kept name different, wordKept name notW]
        simp [ready, setLocal, notG]

/-- The unchanged word-pattern cache covers every word in every retained anagram group. -/
theorem result (words : List String) (buckets : List Bucket) (heap : Mutable.Heap) (dictionary address : Nat)
    (env : HeapCalls.Env) (view : StringValueBuckets.View heap dictionary buckets)
    (partition : StringValueBuckets.Partition solve.keyOfValue (solve.groupValues words) buckets)
    (found : heap[address]? = some (.list ((groups.selected buckets).map groups.reference)))
    (getGroups : env "anagram_groups" = some (.ref address)) :
    ∃ finalEnv rows garbage,
      BlockContinues program env ((((findFunctionIn program.body "solve").getD ([], [])).2.drop 6).take 2)
        heap finalEnv (work heap rows garbage) ∧ Valid (work heap rows garbage) rows ∧
      (∀ word, (KeyedReferences.lookup rows word).isSome ↔ word ∈ allWords words (groups.selected buckets)) ∧
      finalEnv "word_pat" = some (.ref heap.length) ∧
      (∀ name, name ∉ ["word_pat", "g", "w"] → finalEnv name = env name) := by
  let ready := setLocal env "word_pat" (.ref heap.length)
  have rowsFound (bucket : Bucket) (member : bucket ∈ groups.selected buckets) :
      heap[bucket.address]? = some (.list ((lengths.wordsIn words bucket).map text)) := by
    have inside := (List.mem_filter.mp member).1
    rw [← lengths.bucket_values words buckets partition bucket inside]
    exact view.lists bucket inside
  obtain ⟨finalEnv, rows, garbage, executed, valid, coverage, kept⟩ := group_loop words heap address [] (groups.selected buckets)
    [] [] ready (by simpa using found) rowsFound (by simp [ready, setLocal]) (by intro row member; cases member)
  refine ⟨finalEnv, rows, garbage, ?_, valid, ?_, ?_, ?_⟩
  · apply (BlockContinues.annAssign (name := "word_pat") (Evaluates.dictionary .nil)).append
    apply BlockContinues.for_sequence (items := (groups.selected buckets).map groups.reference) (by intros; simp)
      (Evaluates.lookup (show ready "anagram_groups" = some (.ref address) by simpa [ready, setLocal] using getGroups))
    · have shared := HeapCollection.prefix_found heap [.dict []] address _ found
      simp [Mutable.sequence, shared]
    · exact executed
  · intro word
    simpa [KeyedReferences.lookup] using coverage word
  · exact (kept "word_pat" (by decide)).trans (by simp [ready, setLocal])
  · intro name different
    have notCache : name ≠ "word_pat" := fun same => different (by simp [same])
    have notLoop : name ∉ ["g", "w"] := fun member => different (List.mem_cons_of_mem _ member)
    rw [kept name notLoop]
    simp [ready, setLocal, notCache]

end solvers.«98».word_patterns
