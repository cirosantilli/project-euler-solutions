import solvers.«98.py.lean.d».Group

namespace solvers.«98».group_search
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open StringValueBuckets (Bucket)
open pattern_signature (program)
open mapped_word (number)
set_option Elab.async false

def scan (words : List String) (buckets : List Bucket) (best : Int) : Int :=
  buckets.foldl (fun current bucket => score words bucket current) best
def loopLocals : List String := "g" :: localNames

theorem loop_continuation (next : Stmt) : Calls.loopContinuation candidates.groupBody next = next := rfl

/-- The outer source loop visits every retained anagram group and preserves the immutable caches. -/
theorem loop (layout : search_state.Layout) (words : List String) (before remaining : List Bucket) (best : Int)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (valid : search_state.Valid layout words heap) (view : groups.selected layout.buckets = before ++ remaining)
    (getPatterns : env "word_pat" = some (.ref layout.patternAddress)) (getSquares : env "squares_cache" = some (.ref layout.squareAddress))
    (getBest : env "best" = some (number best)) :
    ∃ finalEnv extra, BlockContinues program env
      [.sequenceStepLoop "g" (.ref layout.groupAddress) before.length candidates.groupBody []] heap finalEnv (heap ++ extra) ∧
      finalEnv "best" = some (number (scan words remaining best)) ∧
      (∀ name, name ∉ loopLocals → finalEnv name = env name) := by
  induction remaining generalizing before best heap env with
  | nil =>
      refine ⟨env, [], ?_, getBest, by intros; rfl⟩
      simp only [List.append_nil]
      apply BlockContinues.sequence_done (items := before.map groups.reference)
      · simp [Mutable.sequence, sequenceValue, valid.groupList, view]
      · simp
      · exact BlockContinues.nil
  | cons bucket rest ih =>
      let ready := setLocal env "g" (groups.reference bucket)
      obtain ⟨stepEnv, stepExtra, stepRan, stepAnswer, stepKept⟩ := result layout words bucket best heap ready valid
        (by simp [view]) (by simpa [ready, setLocal] using getPatterns) (by simpa [ready, setLocal] using getSquares)
        (by simp [ready, setLocal, groups.reference]) (by simpa [ready, setLocal] using getBest)
      obtain ⟨finalEnv, extra, continued, answer, kept⟩ := ih (before ++ [bucket]) (score words bucket best) (heap ++ stepExtra) stepEnv
        (valid.append stepExtra) (by simpa only [List.append_assoc, List.singleton_append] using view)
        ((stepKept "word_pat" (by decide)).trans (by simpa [ready, setLocal] using getPatterns))
        ((stepKept "squares_cache" (by decide)).trans (by simpa [ready, setLocal] using getSquares)) stepAnswer
      refine ⟨finalEnv, stepExtra ++ extra, ?_, answer, ?_⟩
      · apply BlockContinues.sequence_step (items := (before ++ bucket :: rest).map groups.reference) (value := groups.reference bucket)
        · simp [Mutable.sequence, sequenceValue, valid.groupList, view]
        · simp
        · rw [loop_continuation]
          simpa only [List.length_append, List.length_singleton, List.append_assoc] using stepRan.append continued
      · intro name absent
        rw [kept name absent, stepKept name (fun member => absent (List.mem_cons_of_mem _ member))]
        simp [ready, setLocal, show name ≠ "g" from fun same => absent (by simp [loopLocals, same])]

theorem result_loop (layout : search_state.Layout) (words : List String) (best : Int) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (valid : search_state.Valid layout words heap) (locals : search_state.Locals layout env) (getBest : env "best" = some (number best)) :
    ∃ finalEnv extra, BlockContinues program env (solve.source.drop 9 |>.take 1) heap finalEnv (heap ++ extra) ∧
      finalEnv "best" = some (number (scan words (groups.selected layout.buckets) best)) ∧
      (∀ name, name ∉ loopLocals → finalEnv name = env name) := by
  obtain ⟨finalEnv, extra, executed, answer, kept⟩ := loop layout words [] (groups.selected layout.buckets) best heap env valid rfl
    locals.getPatterns locals.getSquares getBest
  refine ⟨finalEnv, extra, ?_, answer, kept⟩
  apply BlockContinues.for_sequence (items := (groups.selected layout.buckets).map groups.reference) (by intros; simp) (Evaluates.lookup locals.getGroups)
  · simp [Mutable.sequence, valid.groupList]
  · exact executed

end solvers.«98».group_search
