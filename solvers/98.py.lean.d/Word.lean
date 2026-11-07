import solvers.«98.py.lean.d».SearchContext

namespace solvers.«98».word_search
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open StringValueBuckets (Bucket)
open pattern_signature (program)
open word_patterns (text)
open mapped_word (number Pending)
set_option Elab.async false

def score (width : Nat) (group : List String) (word : String) (best : Int) : Int :=
  square_search.scan width word (others.values group word) (candidates.values width word) best

def localNames : List String := ["p1", "candidates", "others"] ++ square_search.localNames
abbrev emptyTest : Expr := .unaryOp .not (.name "candidates")
abbrev emptyGuard : Stmt := .ifThen emptyTest [.continueLoop] []

theorem source : candidates.wordBody =
    [.assign [.name "p1"] (.subscript (.name "word_pat") (.name "w1")),
     .assign [.name "candidates"] candidates.expression, emptyGuard,
     .assign [.name "others"] others.expression] ++ candidates.wordBody.drop 4 := rfl

/-- One actual word iteration reads its pattern, handles empty candidates, and scans all remaining square/partner combinations. -/
theorem result (layout : search_state.Layout) (words : List String) (bucket : Bucket) (word : String) (width best : Int)
    (heap : Mutable.Heap) (recordAddress : Nat) (env : HeapCalls.Env) (next : Stmt)
    (valid : search_state.Valid layout words heap) (selected : bucket ∈ groups.selected layout.buckets)
    (member : word ∈ lengths.wordsIn words bucket)
    (stored : square_cache.Stored layout.squareAddress heap width recordAddress)
    (getPatterns : env "word_pat" = some (.ref layout.patternAddress))
    (getWord : env "w1" = some (text word)) (getGroup : env "g" = some (.ref bucket.address))
    (getRecord : env "sqinfo" = some (.ref recordAddress)) (getBest : env "best" = some (number best)) :
    ∃ finalEnv extra, Pending env (candidates.wordBody ++ [.loopResume next]) heap finalEnv [next] (heap ++ extra) ∧
      finalEnv "best" = some (number (score width.toNat (lengths.wordsIn words bucket) word best)) ∧
      (∀ name, name ∉ localNames → finalEnv name = env name) := by
  obtain ⟨patternAddress, readPattern, patternFound⟩ := word_patterns.read heap layout.patternAddress layout.patternRows word env
    valid.patternDictionary valid.patternValid (valid.pattern_present bucket word selected member) getPatterns getWord
  let patternEnv := setLocal env "p1" (.ref patternAddress)
  obtain ⟨candidateAddress, readCandidates, candidatesFound⟩ := candidates.result heap layout.squareAddress width recordAddress patternAddress word patternEnv
    stored patternFound (by simpa [patternEnv, setLocal] using getRecord) (by simp [patternEnv, setLocal])
  let current := heap ++ [Mutable.Object.list []]
  change current[candidateAddress]? = some (.list ((candidates.values width.toNat word).map text)) at candidatesFound
  let ready := setLocal patternEnv "candidates" (.ref candidateAddress)
  have prepared : BlockContinues program env (candidates.wordBody.take 2) heap ready current :=
    (BlockContinues.assign readPattern).append (BlockContinues.assign readCandidates)
  have keepReady (name : String) (absent : name ∉ ["p1", "candidates"]) : ready name = env name := by
    simp only [List.mem_cons, not_or] at absent
    simp [ready, patternEnv, setLocal, absent]
  have tested : Evaluates program ready emptyTest current
      (.scalar (.bool (decide (candidates.values width.toNat word = [])))) current := by
    apply Evaluates.unary (Evaluates.lookup (show ready "candidates" = some (.ref candidateAddress) by simp [ready, setLocal]))
    simp [HeapCalls.unary, Mutable.truth, candidatesFound]
    cases candidates.values width.toNat word <;> rfl
  by_cases empty : candidates.values width.toNat word = []
  · refine ⟨ready, [.list []], ?_, ?_, ?_⟩
    · intro following anchors stack
      apply (prepared ((candidates.wordBody.drop 2) ++ .loopResume next :: following) anchors stack).trans
      have resumed : Reaches program anchors
          ⟨.block (.continueLoop :: candidates.wordBody.drop 3 ++ (.loopResume next :: following)) ready, stack, current⟩
          ⟨.block (next :: following) ready, stack, current⟩ := .next rfl .refl
      apply Reaches.next rfl
      apply (tested anchors _).trans
      refine Reaches.next ?_ resumed
      simp [step, run, Mutable.truth, scalarTruth, empty]
      rfl
    · simpa [score, square_search.scan, empty] using (keepReady "best" (by decide)).trans getBest
    · intro name absent
      exact keepReady name (fun member => absent (by simp only [localNames, List.mem_append, List.mem_cons] at *; tauto))
  · have guard : BlockContinues program ready [emptyGuard] current ready current :=
      BlockContinues.branch (flag := false) tested (by simp [Mutable.truth, scalarTruth, empty]) BlockContinues.nil
    let output := Mutable.Object.list ((others.values (lengths.wordsIn words bucket) word).map text)
    let partnersEnv := setLocal ready "others" (.ref current.length)
    have filtered := others.result (lengths.wordsIn words bucket) word current bucket.address ready
      (HeapCollection.prefix_found heap [.list []] bucket.address _ (valid.group_found bucket selected))
      ((keepReady "g" (by decide)).trans getGroup) ((keepReady "w1" (by decide)).trans getWord)
    have assigned : BlockContinues program ready [.assign [.name "others"] others.expression]
        current partnersEnv (current ++ [output]) := BlockContinues.assign filtered
    have keepPartners (name : String) (absent : name ∉ ["p1", "candidates", "others"]) : partnersEnv name = env name := by
      have notOthers : name ≠ "others" := fun same => absent (by simp [same])
      simp only [partnersEnv, setLocal, if_neg notOthers]
      apply keepReady
      intro member
      exact absent (List.mem_append_left ["others"] member)
    obtain ⟨finalEnv, extra, scanned, answer, kept⟩ := square_search.result width word
      (others.values (lengths.wordsIn words bucket) word) best (current ++ [output]) candidateAddress current.length layout.squareAddress recordAddress partnersEnv
      (groups.partners_valid words bucket word member)
      (HeapCollection.prefix_found current [output] candidateAddress _ candidatesFound)
      (by simp [output]) ((stored.append [.list []]).append [output])
      (by simp [partnersEnv, ready, setLocal]) ((keepPartners "w1" (by decide)).trans getWord)
      (by simp [partnersEnv, setLocal]) ((keepPartners "sqinfo" (by decide)).trans getRecord)
      ((keepPartners "best" (by decide)).trans getBest)
    have executed : BlockContinues program env candidates.wordBody heap finalEnv ((current ++ [output]) ++ extra) :=
      prepared.append (guard.append (assigned.append scanned))
    refine ⟨finalEnv, [.list [], output] ++ extra, ?_, answer, ?_⟩
    · intro following anchors stack
      simpa only [current, List.append_assoc, List.singleton_append] using
        (executed (.loopResume next :: following) anchors stack).trans (.next rfl .refl)
    · intro name absent
      rw [kept name (fun member => absent (List.mem_append_right _ member))]
      exact keepPartners name (fun member => absent (List.mem_append_left _ member))

end solvers.«98».word_search
