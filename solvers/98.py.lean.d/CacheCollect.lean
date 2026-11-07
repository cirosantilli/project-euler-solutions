import solvers.«98.py.lean.d».CacheStorage
import solvers.«98.py.lean.d».Lengths

namespace solvers.«98».square_cache
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open pattern_signature (program)
set_option Elab.async false

abbrev expression : Expr := .dictComp (.name "L") (.call (.name "squares_with_length") [.name "L"])
  [(.name "L", .name "lengths_needed", [])]

theorem source : (((findFunctionIn program.body "solve").getD ([], [])).2)[5]? = some
    (.annAssign (.name "squares_cache") (.subscript (.name "Dict") (.tuple [.name "int", .name "SquaresByLength"])) (some expression)) := rfl

theorem loop (base : Mutable.Heap) (address : Nat) (before remaining : List Int) (rows : Rows) (garbage : Mutable.Heap)
    (env : HeapCalls.Env) (found : base[address]? = some (.list ((before ++ remaining).map number)))
    (positive : ∀ width ∈ remaining, 0 < width) (valid : Valid base.length (work base rows garbage) rows) :
    ∃ finalEnv finalRows finalGarbage,
      BlockContinues program env [.sequenceStepLoop "L" (.ref address) before.length (body base.length) []]
        (work base rows garbage) finalEnv (work base finalRows finalGarbage) ∧
      Valid base.length (work base finalRows finalGarbage) finalRows ∧
      (∀ width, (KeyedReferences.lookup finalRows width).isSome ↔ width ∈ remaining ∨ (KeyedReferences.lookup rows width).isSome) := by
  induction remaining generalizing before rows garbage env with
  | nil =>
      refine ⟨env, rows, garbage, ?_, valid, by simp⟩
      apply BlockContinues.sequence_done (items := before.map number)
      · have kept := HeapCollection.prefix_found base (.dict (KeyedReferences.entries KeyedReferences.integers rows) :: garbage) address _ found
        simp [Mutable.sequence, sequenceValue, work, KeyedReferences.work, kept]
      · simp
      · exact BlockContinues.nil
  | cons width rest ih =>
      let ready := setLocal env "L" (number width)
      obtain ⟨extra, recordAddress, executed, stepValid⟩ := insert base rows garbage width (positive width (by simp)) ready
        (by simp [ready, setLocal]) valid
      obtain ⟨finalEnv, finalRows, finalGarbage, continued, finalValid, coverage⟩ :=
        ih (before ++ [width]) (KeyedReferences.insert rows width recordAddress) (garbage ++ extra) ready
          (by simpa only [List.append_assoc, List.singleton_append] using found)
          (fun next member => positive next (by simp [member])) stepValid
      refine ⟨finalEnv, finalRows, finalGarbage, ?_, finalValid, ?_⟩
      · apply BlockContinues.sequence_step (items := (before ++ width :: rest).map number) (value := number width)
        · have kept := HeapCollection.prefix_found base (.dict (KeyedReferences.entries KeyedReferences.integers rows) :: garbage) address _ found
          simp [Mutable.sequence, sequenceValue, work, KeyedReferences.work, kept]
        · simp
        · simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, body,
            Bool.false_or, Bool.false_eq_true, if_false, List.length_append, List.length_singleton] using executed.append continued
      · intro query
        rw [coverage query, KeyedReferences.lookup_insert]
        by_cases same : query = width <;> simp [same, or_comm]

/-- Build the actual dictionary of square-cache records for every requested positive width. -/
theorem result (widths : List Int) (positive : ∀ width ∈ widths, 0 < width) (heap : Mutable.Heap) (address : Nat)
    (env : HeapCalls.Env) (found : heap[address]? = some (.list (widths.map number)))
    (getWidths : env "lengths_needed" = some (.ref address)) :
    ∃ rows garbage, Evaluates program env expression heap (.ref heap.length) (work heap rows garbage) ∧
      Valid heap.length (work heap rows garbage) rows ∧
      (∀ width, (KeyedReferences.lookup rows width).isSome ↔ width ∈ widths) := by
  let inner := comprehensionTargetsEnv env [(.name "L", .name "lengths_needed", [])]
  obtain ⟨finalEnv, rows, garbage, executed, valid, coverage⟩ := loop heap address [] widths [] [] inner
    (by simpa using found) positive (by intro row member; cases member)
  refine ⟨rows, garbage, ?_, valid, ?_⟩
  · apply Evaluates.dict_comp_sequence (items := widths.map number) (finalEnv := finalEnv)
      (by intros; simp) (by intros; simp) (by intros; simp) (Evaluates.lookup getWidths)
    · simp [Mutable.sequence, found]
    · exact executed
  · intro width
    simpa [KeyedReferences.lookup] using coverage width

end solvers.«98».square_cache
