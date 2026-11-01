import solvers.«79.invalid.source»

/-! The graph-building loop over a sequence stored in the preserved caller objects. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls
set_option Elab.async false

/-- Reading the retained sequence reference is stable across private graph allocations. -/
structure SequenceInput (caller : Mutable.Heap) (source : SequenceSource)
    (attempts : List String) : Prop where
  read : ∀ heap, caller <+: heap →
    Mutable.sequence (sequenceValue source) heap = some (.ok (attempts.map text), heap)

theorem SequenceInput.list (caller : Mutable.Heap) (address : Nat) (attempts : List String)
    (found : caller[address]? = some (.list (attempts.map text))) :
    SequenceInput caller (.ref address) attempts := by
  constructor
  intro heap saved
  obtain ⟨suffix, rfl⟩ := saved
  have retained : (caller ++ suffix)[address]? = some (.list (attempts.map text)) := by
    rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
    exact found
  simp [sequenceValue, Mutable.sequence, retained]

theorem SequenceInput.tuple (caller : Mutable.Heap) (address : Nat) (attempts : List String)
    (found : caller[address]? = some (.tuple (attempts.map text))) :
    SequenceInput caller (.ref address) attempts := by
  constructor
  intro heap saved
  obtain ⟨suffix, rfl⟩ := saved
  have retained : (caller ++ suffix)[address]? = some (.tuple (attempts.map text)) := by
    rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
    exact found
  simp [sequenceValue, Mutable.sequence, retained]

structure GraphData where
  vertices : List String
  edges : List (String × String)
  deriving DecidableEq

def Storage.graphData (state : Storage) : GraphData := ⟨state.vertices, state.edges⟩

def GraphData.extend (data : GraphData) (s : String) : GraphData :=
  match s.toList with
  | [a, b, c] =>
      ⟨insertVertex (character c) (insertVertex (character b)
        (insertVertex (character a) data.vertices)),
       insertEdge (character b) (character c)
        (insertEdge (character a) (character b) data.edges)⟩
  | _ => data

def ValidAttempt (s : String) : Prop := s.length = 3 ∧ Strings.isDigit s = true

theorem Bindings.setAttempt (roots : Roots) (env : HeapCalls.Env)
    (bound : Bindings roots env) (s : String) : Bindings roots (setLocal env "s" (text s)) := by
  constructor
  · simpa [setLocal] using bound.nodes
  · simpa [setLocal] using bound.adjacency
  · simpa [setLocal] using bound.degrees

theorem valid_iteration (caller : Mutable.Heap) (roots : Roots) (state : Storage)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (represented : Represents roots state heap)
    (saved : CallerPreserved caller roots state heap) (bound : Bindings roots env)
    (s : String) (valid : ValidAttempt s) :
    ∃ after next nextEnv,
      BlockContinues program (setLocal env "s" (text s)) attemptBody heap nextEnv after ∧
      Represents roots next after ∧ CallerPreserved caller roots next after ∧
      Bindings roots nextEnv ∧ next.graphData = state.graphData.extend s := by
  obtain ⟨a, b, c, after, next, characters, executed, result, bindings, vertices, edges, framed⟩ :=
    valid_attempt roots state heap (setLocal env "s" (text s)) represented
      (bound.setAttempt roots env s) s (by simp [setLocal]) valid.1 valid.2
  refine ⟨after, next, _, executed, result, framed caller saved, bindings, ?_⟩
  simp only [GraphData.extend, characters, Storage.graphData]
  exact congrArg₂ GraphData.mk vertices edges

def sequenceLoop (source : SequenceSource) (position : Nat) : Stmt :=
  .sequenceStepLoop "s" source position attemptBody []

/-- Every successful iteration consumes one item and leaves the original sequence untouched. -/
theorem valid_sequence (caller : Mutable.Heap) (source : SequenceSource)
    (attempts : List String) (input : SequenceInput caller source attempts)
    (remaining before : List String) (partition : attempts = before ++ remaining)
    (valid : ∀ s ∈ remaining, ValidAttempt s)
    (roots : Roots) (state : Storage) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (represented : Represents roots state heap) (saved : CallerPreserved caller roots state heap)
    (bound : Bindings roots env) :
    ∃ after next nextEnv,
      BlockContinues program env [sequenceLoop source before.length] heap nextEnv after ∧
      Represents roots next after ∧ CallerPreserved caller roots next after ∧
      Bindings roots nextEnv ∧
      next.graphData = remaining.foldl GraphData.extend state.graphData := by
  induction remaining generalizing before state heap env with
  | nil =>
      refine ⟨heap, state, env, ?_, represented, saved, bound, rfl⟩
      apply BlockContinues.sequence_done (input.read heap saved.unchanged) _ .nil
      simp [partition]
  | cons s rest ih =>
      obtain ⟨middle, current, middleEnv, iteration,
        currentRep, currentSaved, currentBound, data⟩ :=
        valid_iteration caller roots state heap env represented saved bound s (valid s (by simp))
      obtain ⟨after, next, nextEnv, continued, result, framed, bindings, contents⟩ :=
        ih (before ++ [s]) (by simpa [List.append_assoc] using partition)
          (fun t member => valid t (by simp [member])) current middle middleEnv
          currentRep currentSaved currentBound
      refine ⟨after, next, nextEnv, ?_, result, framed, bindings, ?_⟩
      · apply BlockContinues.sequence_step (input.read heap saved.unchanged)
          (value := text s)
        · simp [partition]
        · have resume : BlockContinues program middleEnv
              [sequenceLoop source (before.length + 1)] middle nextEnv after := by
            simpa using continued
          exact iteration.append resume
      · simpa only [List.foldl_cons, data] using contents

end solvers.«79».graph
