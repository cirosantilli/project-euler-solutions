import solvers.«98.py.lean.d».CandidateExecution

namespace solvers.«98».square_search
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open pattern_signature (program)
open mapped_word (number integerValue)
open word_patterns (text)
set_option Elab.async false

def scan (width : Nat) (word : String) (others candidates : List String) (best : Int) : Int :=
  candidates.foldl (fun current square => mapped_word.scan width word.toList square.toList others (integerValue square) current) best

def localNames : List String := "sq_str" :: candidateLocals

theorem loop_continuation (next : Stmt) : Calls.loopContinuation bijection.candidateBody next = .loopResume next := rfl

/-- Every selected square is processed exactly once, with all partner attempts and caller objects preserved. -/
theorem loop (width : Int) (word : String) (others before remaining : List String) (best : Int)
    (heap : Mutable.Heap) (selected partners outer recordAddress : Nat) (env : HeapCalls.Env)
    (members : ∀ square ∈ remaining, square ∈ candidates.values width.toNat word)
    (valid : ∀ other ∈ others, other.toList ≠ [] ∧ ∀ letter ∈ other.toList, letter ∈ word.toList)
    (foundSelected : heap[selected]? = some (.list ((before ++ remaining).map text)))
    (foundPartners : heap[partners]? = some (.list (others.map text)))
    (stored : square_cache.Stored outer heap width recordAddress)
    (getWord : env "w1" = some (.scalar (.string word)))
    (getPartners : env "others" = some (.ref partners)) (getRecord : env "sqinfo" = some (.ref recordAddress))
    (getBest : env "best" = some (number best)) :
    ∃ finalEnv extra, BlockContinues program env
      [.sequenceStepLoop "sq_str" (.ref selected) before.length bijection.candidateBody []] heap finalEnv (heap ++ extra) ∧
      finalEnv "best" = some (number (scan width.toNat word others remaining best)) ∧
      (∀ name, name ∉ localNames → finalEnv name = env name) := by
  induction remaining generalizing before best heap env with
  | nil =>
      refine ⟨env, [], ?_, getBest, by intros; rfl⟩
      simp only [List.append_nil]
      apply BlockContinues.sequence_done (items := before.map text)
      · simp [Mutable.sequence, sequenceValue, foundSelected]
      · simp
      · exact BlockContinues.nil
  | cons square rest ih =>
      let ready := setLocal env "sq_str" (text square)
      obtain ⟨stepEnv, stepExtra, stepRan, stepAnswer, stepKept⟩ := candidate_result width word square others best
        heap partners outer recordAddress ready (members square (by simp)) valid foundPartners stored
        (by simpa [ready, setLocal] using getWord) (by simp [ready, setLocal])
        (by simpa [ready, setLocal] using getPartners) (by simpa [ready, setLocal] using getRecord)
        (by simpa [ready, setLocal] using getBest)
      obtain ⟨finalEnv, extra, continued, answer, kept⟩ := ih (before ++ [square])
        (mapped_word.scan width.toNat word.toList square.toList others (integerValue square) best) (heap ++ stepExtra) stepEnv
        (fun item member => members item (by simp [member]))
        (by simpa only [List.append_assoc, List.singleton_append] using HeapCollection.prefix_found heap stepExtra selected _ foundSelected)
        (HeapCollection.prefix_found heap stepExtra partners _ foundPartners) (stored.append stepExtra)
        ((stepKept "w1" (by decide)).trans (by simpa [ready, setLocal] using getWord))
        ((stepKept "others" (by decide)).trans (by simpa [ready, setLocal] using getPartners))
        ((stepKept "sqinfo" (by decide)).trans (by simpa [ready, setLocal] using getRecord)) stepAnswer
      refine ⟨finalEnv, stepExtra ++ extra, ?_, answer, ?_⟩
      · apply BlockContinues.sequence_step (items := (before ++ square :: rest).map text) (value := text square)
        · simp [Mutable.sequence, sequenceValue, foundSelected]
        · simp
        · rw [loop_continuation]
          intro following anchors stack
          apply (stepRan _ anchors stack).trans
          refine Reaches.next (following := ⟨.block (.sequenceStepLoop "sq_str" (.ref selected) (before.length + 1)
            bijection.candidateBody [] :: following) stepEnv, stack, heap ++ stepExtra⟩) rfl ?_
          simpa only [List.length_append, List.length_singleton, List.append_assoc] using continued following anchors stack
      · intro name absent
        rw [kept name absent, stepKept name (fun member => absent (List.mem_cons_of_mem _ member))]
        simp [ready, setLocal, show name ≠ "sq_str" from fun same => absent (by simp [localNames, same])]

theorem result (width : Int) (word : String) (others : List String) (best : Int)
    (heap : Mutable.Heap) (selected partners outer recordAddress : Nat) (env : HeapCalls.Env)
    (valid : ∀ other ∈ others, other.toList ≠ [] ∧ ∀ letter ∈ other.toList, letter ∈ word.toList)
    (foundSelected : heap[selected]? = some (.list ((candidates.values width.toNat word).map text)))
    (foundPartners : heap[partners]? = some (.list (others.map text)))
    (stored : square_cache.Stored outer heap width recordAddress)
    (getSelected : env "candidates" = some (.ref selected)) (getWord : env "w1" = some (.scalar (.string word)))
    (getPartners : env "others" = some (.ref partners)) (getRecord : env "sqinfo" = some (.ref recordAddress))
    (getBest : env "best" = some (number best)) :
    ∃ finalEnv extra, BlockContinues program env (candidates.wordBody.drop 4) heap finalEnv (heap ++ extra) ∧
      finalEnv "best" = some (number (scan width.toNat word others (candidates.values width.toNat word) best)) ∧
      (∀ name, name ∉ localNames → finalEnv name = env name) := by
  obtain ⟨finalEnv, extra, executed, answer, kept⟩ := loop width word others [] (candidates.values width.toNat word) best
    heap selected partners outer recordAddress env (by intros; assumption) valid foundSelected foundPartners stored
    getWord getPartners getRecord getBest
  refine ⟨finalEnv, extra, ?_, answer, kept⟩
  apply BlockContinues.for_sequence (items := (candidates.values width.toNat word).map text) (by intros; simp) (Evaluates.lookup getSelected)
  · simp [Mutable.sequence, foundSelected]
  · exact executed

end solvers.«98».square_search
