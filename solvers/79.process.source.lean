import solvers.«79.sequence.source»

/-! Graph construction consumes the sequence once and stops at the first invalid attempt. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls
set_option Elab.async false

instance (s : String) : Decidable (ValidAttempt s) := by
  unfold ValidAttempt
  infer_instance

def GraphData.process (data : GraphData) : List String → Except ExceptionObject GraphData
  | [] => .ok data
  | s :: rest =>
      if ValidAttempt s then (data.extend s).process rest else .error (invalidException s)

def SequenceResult (caller : Mutable.Heap) (roots : Roots) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (source : SequenceSource) (position : Nat) :
    Except ExceptionObject GraphData → Prop
  | .ok data => ∃ after next nextEnv,
      BlockContinues program env [sequenceLoop source position] heap nextEnv after ∧
      Represents roots next after ∧ CallerPreserved caller roots next after ∧
      Bindings roots nextEnv ∧ next.graphData = data
  | .error error => ∃ after, caller <+: after ∧ ∀ following anchors stack,
      Finishes (step program anchors)
        ⟨.block ([sequenceLoop source position] ++ following) env, stack, heap⟩
        (.ok ⟨.error error, [], after, anchors⟩)

theorem sequence_enter (caller : Mutable.Heap) (source : SequenceSource)
    (attempts before rest : List String) (s : String)
    (input : SequenceInput caller source attempts) (partition : attempts = before ++ s :: rest)
    (heap : Mutable.Heap) (saved : caller <+: heap) (env : HeapCalls.Env)
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) :
    Reaches program anchors
      ⟨.block ([sequenceLoop source before.length] ++ following) env, stack, heap⟩
      ⟨.block (attemptBody ++ ([sequenceLoop source (before.length + 1)] ++ following))
        (setLocal env "s" (text s)), stack, heap⟩ := by
  have item : (attempts.map text)[before.length]? = some (text s) := by simp [partition]
  exact .next (by
    simp [sequenceLoop, step, run, input.read heap saved, item, List.append_assoc]
    rfl) .refl

/-- The loop has the same success or first-error result as structural graph construction. -/
theorem process_sequence (caller : Mutable.Heap) (source : SequenceSource)
    (attempts : List String) (input : SequenceInput caller source attempts)
    (remaining before : List String) (partition : attempts = before ++ remaining)
    (roots : Roots) (state : Storage) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (represented : Represents roots state heap) (saved : CallerPreserved caller roots state heap)
    (bound : Bindings roots env) :
    SequenceResult caller roots heap env source before.length
      (state.graphData.process remaining) := by
  induction remaining generalizing before state heap env with
  | nil =>
      refine ⟨heap, state, env, ?_, represented, saved, bound, rfl⟩
      apply BlockContinues.sequence_done (input.read heap saved.unchanged) _ .nil
      simp [partition]
  | cons s rest ih =>
      rw [GraphData.process]
      by_cases valid : ValidAttempt s
      · rw [if_pos valid]
        obtain ⟨middle, current, middleEnv, iteration,
          currentRep, currentSaved, currentBound, data⟩ :=
          valid_iteration caller roots state heap env represented saved bound s valid
        have recursive := ih (before ++ [s]) (by simpa [List.append_assoc] using partition)
          current middle middleEnv currentRep currentSaved currentBound
        rw [← data]
        cases outcome : current.graphData.process rest with
        | ok result =>
            rw [outcome] at recursive
            obtain ⟨after, next, nextEnv, continued,
              resultRep, framed, bindings, contents⟩ := recursive
            refine ⟨after, next, nextEnv, ?_, resultRep, framed, bindings, contents⟩
            apply BlockContinues.sequence_step (input.read heap saved.unchanged)
              (value := text s)
            · simp [partition]
            · have resume : BlockContinues program middleEnv
                  [sequenceLoop source (before.length + 1)] middle nextEnv after := by
                simpa using continued
              exact iteration.append resume
        | error error =>
            rw [outcome] at recursive
            obtain ⟨after, framed, failed⟩ := recursive
            refine ⟨after, framed, ?_⟩
            intro following anchors stack
            apply (sequence_enter caller source attempts before rest s input partition heap
              saved.unchanged env following anchors stack).finishes
            apply (iteration ([sequenceLoop source (before.length + 1)] ++ following)
              anchors stack).finishes
            simpa using failed following anchors stack
      · rw [if_neg valid]
        refine ⟨heap, saved.unchanged, ?_⟩
        intro following anchors stack
        apply (sequence_enter caller source attempts before rest s input partition heap
          saved.unchanged env following anchors stack).finishes
        apply invalid_attempt (setLocal env "s" (text s)) heap s (by simp [setLocal])
        by_cases length : s.length = 3
        · exact Or.inr (by simpa [ValidAttempt, length] using valid)
        · exact Or.inl length

end solvers.«79».graph
