import solvers.«98.py.lean.d».WordLoop

namespace solvers.«98».group_search
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open StringValueBuckets (Bucket)
open pattern_signature (program)
open word_patterns (text)
open mapped_word (number)
set_option Elab.async false

def score (words : List String) (bucket : Bucket) (best : Int) : Int :=
  word_search.scan bucket.key.length (lengths.wordsIn words bucket) (lengths.wordsIn words bucket) best
def localNames : List String := "L" :: "sqinfo" :: word_search.loopLocals
abbrev measure : Expr := .builtinCall .len [.subscript (.name "g") (.constant (.int 0))]

theorem word_width (words : List String) (bucket : Bucket) (word : String)
    (member : word ∈ lengths.wordsIn words bucket) : word.length = bucket.key.length := by
  have measured := AnagramKeys.key_length word
  rw [(groups.word_member words bucket word member).2.2] at measured
  simpa using measured.symm

theorem source : candidates.groupBody.take 2 =
    [.assign [.name "L"] measure, .assign [.name "sqinfo"] (.subscript (.name "squares_cache") (.name "L"))] := rfl

/-- Each retained group reads its first word's width, finds the cached square record, and executes its complete word loop. -/
theorem result (layout : search_state.Layout) (words : List String) (bucket : Bucket) (best : Int)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (valid : search_state.Valid layout words heap) (selected : bucket ∈ groups.selected layout.buckets)
    (getPatterns : env "word_pat" = some (.ref layout.patternAddress)) (getSquares : env "squares_cache" = some (.ref layout.squareAddress))
    (getGroup : env "g" = some (.ref bucket.address)) (getBest : env "best" = some (number best)) :
    ∃ finalEnv extra, BlockContinues program env candidates.groupBody heap finalEnv (heap ++ extra) ∧
      finalEnv "best" = some (number (score words bucket best)) ∧
      (∀ name, name ∉ localNames → finalEnv name = env name) := by
  have size := groups.selected_size words layout.buckets valid.partition bucket selected
  obtain ⟨first, rest, view⟩ := List.exists_cons_of_ne_nil (show lengths.wordsIn words bucket ≠ [] by intro empty; simp [empty] at size)
  have firstMember : first ∈ lengths.wordsIn words bucket := by simp [view]
  have width := word_width words bucket first firstMember
  have found := valid.group_found bucket selected
  have firstRead : Evaluates program env (.subscript (.name "g") (.constant (.int 0))) heap (text first) heap := by
    apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getGroup) Evaluates.constant
    simp [Mutable.getItem, found, view, Mutable.index, Except.map]
  have measured : Evaluates program env measure heap (number bucket.key.length) heap := by
    apply Evaluates.builtin (.cons firstRead .nil)
    simp [primitive, Mutable.length, text, number, width]
  let lengthEnv := setLocal env "L" (number bucket.key.length)
  obtain ⟨recordAddress, readRecord, stored⟩ := square_cache.read heap layout.squareAddress layout.squareRows bucket.key.length lengthEnv
    valid.squareDictionary valid.squareValid
    ((valid.squareCoverage _).mpr ((lengths.mem_needed words layout.buckets _).mpr ⟨bucket, selected, first, firstMember, by simp [width]⟩))
    (by simpa [lengthEnv, setLocal] using getSquares) (by simp [lengthEnv, setLocal])
  let ready := setLocal lengthEnv "sqinfo" (.ref recordAddress)
  have keepReady (name : String) (absent : name ∉ ["L", "sqinfo"]) : ready name = env name := by
    simp only [List.mem_cons, not_or] at absent
    simp [ready, lengthEnv, setLocal, absent]
  have prepared : BlockContinues program env (candidates.groupBody.take 2) heap ready heap :=
    (BlockContinues.assign measured).append (BlockContinues.assign readRecord)
  obtain ⟨finalEnv, extra, looped, answer, kept⟩ := word_search.result_loop layout words bucket bucket.key.length best heap recordAddress ready
    valid selected stored ((keepReady "word_pat" (by decide)).trans getPatterns) ((keepReady "g" (by decide)).trans getGroup)
    (by simp [ready, setLocal]) ((keepReady "best" (by decide)).trans getBest)
  refine ⟨finalEnv, extra, ?_, ?_, ?_⟩
  · simpa only [List.take_append_drop] using prepared.append looped
  · simpa only [score, Int.toNat_natCast] using answer
  · intro name absent
    rw [kept name (fun member => absent (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ member)))]
    apply keepReady
    intro member
    exact absent (by simp only [localNames, List.mem_cons] at *; tauto)

end solvers.«98».group_search
