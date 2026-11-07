import solvers.«98.py.lean.d».Word

namespace solvers.«98».word_search
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open StringValueBuckets (Bucket)
open pattern_signature (program)
open word_patterns (text)
open mapped_word (number)
set_option Elab.async false

def scan (width : Nat) (group remaining : List String) (best : Int) : Int :=
  remaining.foldl (fun current word => score width group word current) best
def loopLocals : List String := "i" :: "w1" :: localNames

theorem loop_continuation (next : Stmt) : Calls.loopContinuation candidates.wordBody next = .loopResume next := rfl

/-- Enumerate every word in a retained group, including duplicate occurrences and empty-candidate skips. -/
theorem loop (layout : search_state.Layout) (words : List String) (bucket : Bucket) (before remaining : List String)
    (width best counter : Int) (heap : Mutable.Heap) (recordAddress : Nat) (env : HeapCalls.Env)
    (valid : search_state.Valid layout words heap) (selected : bucket ∈ groups.selected layout.buckets)
    (view : lengths.wordsIn words bucket = before ++ remaining)
    (stored : square_cache.Stored layout.squareAddress heap width recordAddress)
    (getPatterns : env "word_pat" = some (.ref layout.patternAddress)) (getGroup : env "g" = some (.ref bucket.address))
    (getRecord : env "sqinfo" = some (.ref recordAddress)) (getBest : env "best" = some (number best)) :
    ∃ finalEnv extra, BlockContinues program env
      [.enumerateStepLoop "i" "w1" (.ref bucket.address) before.length counter candidates.wordBody []] heap finalEnv (heap ++ extra) ∧
      finalEnv "best" = some (number (scan width.toNat (lengths.wordsIn words bucket) remaining best)) ∧
      (∀ name, name ∉ loopLocals → finalEnv name = env name) := by
  induction remaining generalizing before best counter heap env with
  | nil =>
      refine ⟨env, [], ?_, getBest, by intros; rfl⟩
      simp only [List.append_nil]
      apply BlockContinues.enumerate_done (items := before.map text)
      · have found := valid.group_found bucket selected
        simp [Mutable.sequence, sequenceValue, found, view]
      · simp
      · exact BlockContinues.nil
  | cons word rest ih =>
      let ready := setLocal (setLocal env "i" (number counter)) "w1" (text word)
      let next := Stmt.enumerateStepLoop "i" "w1" (.ref bucket.address) (before.length + 1) (counter + 1) candidates.wordBody []
      obtain ⟨stepEnv, stepExtra, stepRan, stepAnswer, stepKept⟩ := result layout words bucket word width best heap recordAddress ready next
        valid selected (by simp [view]) stored (by simpa [ready, setLocal] using getPatterns) (by simp [ready, setLocal])
        (by simpa [ready, setLocal] using getGroup) (by simpa [ready, setLocal] using getRecord) (by simpa [ready, setLocal] using getBest)
      obtain ⟨finalEnv, extra, continued, answer, kept⟩ := ih (before ++ [word])
        (score width.toNat (lengths.wordsIn words bucket) word best) (counter + 1) (heap ++ stepExtra) stepEnv
        (valid.append stepExtra) (by simpa only [List.append_assoc, List.singleton_append] using view) (stored.append stepExtra)
        ((stepKept "word_pat" (by decide)).trans (by simpa [ready, setLocal] using getPatterns))
        ((stepKept "g" (by decide)).trans (by simpa [ready, setLocal] using getGroup))
        ((stepKept "sqinfo" (by decide)).trans (by simpa [ready, setLocal] using getRecord)) stepAnswer
      refine ⟨finalEnv, stepExtra ++ extra, ?_, answer, ?_⟩
      · apply BlockContinues.enumerate_step (items := (before ++ word :: rest).map text) (value := text word)
        · have found := valid.group_found bucket selected
          simp [Mutable.sequence, sequenceValue, found, view]
        · simp
        · rw [loop_continuation]
          intro following anchors stack
          have finish := (stepRan following anchors stack).trans (by
            simpa only [next, List.length_append, List.length_singleton] using continued following anchors stack)
          simpa only [List.append_assoc] using finish
      · intro name absent
        rw [kept name absent, stepKept name (fun member => absent (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ member)))]
        have notIndex : name ≠ "i" := fun same => absent (by simp [loopLocals, same])
        have notWord : name ≠ "w1" := fun same => absent (by simp [loopLocals, same])
        simp [ready, setLocal, notIndex, notWord]

theorem result_loop (layout : search_state.Layout) (words : List String) (bucket : Bucket) (width best : Int)
    (heap : Mutable.Heap) (recordAddress : Nat) (env : HeapCalls.Env)
    (valid : search_state.Valid layout words heap) (selected : bucket ∈ groups.selected layout.buckets)
    (stored : square_cache.Stored layout.squareAddress heap width recordAddress)
    (getPatterns : env "word_pat" = some (.ref layout.patternAddress)) (getGroup : env "g" = some (.ref bucket.address))
    (getRecord : env "sqinfo" = some (.ref recordAddress)) (getBest : env "best" = some (number best)) :
    ∃ finalEnv extra, BlockContinues program env (candidates.groupBody.drop 2) heap finalEnv (heap ++ extra) ∧
      finalEnv "best" = some (number (scan width.toNat (lengths.wordsIn words bucket) (lengths.wordsIn words bucket) best)) ∧
      (∀ name, name ∉ loopLocals → finalEnv name = env name) := by
  obtain ⟨finalEnv, extra, executed, answer, kept⟩ := loop layout words bucket [] (lengths.wordsIn words bucket) width best 0 heap recordAddress env
    valid selected rfl stored getPatterns getGroup getRecord getBest
  refine ⟨finalEnv, extra, ?_, answer, kept⟩
  apply BlockContinues.for_enumerate (items := (lengths.wordsIn words bucket).map text) (Evaluates.lookup getGroup)
  · have found := valid.group_found bucket selected
    simp [Mutable.sequence, found]
  · exact executed

end solvers.«98».word_search
