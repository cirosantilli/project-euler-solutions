import solvers.«79.iterator.isolation.source»
import lang.py.FormalSW.IteratorPrefix

/-! Graph construction stops at the first invalid yield without requiring a terminating iterator. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls Mutable.Relocation IteratorCalls
set_option Elab.async false

theorem process_iterator_prefix (context : Module) (extension : Extends context program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (iterator : Mutable.Value) (logical finalLogical : Mutable.Heap) (used finalUsed : Nat)
    (attempts : List String)
    (trace : YieldPrefix context choices iterator text logical used attempts finalLogical finalUsed)
    (noGrowth : finalLogical.length = logical.length) (valid : ∀ s ∈ attempts, ValidAttempt s)
    (physical : Mutable.Heap) (rename : Nat → Nat) (roots : Roots) (state : Storage)
    (env : HeapCalls.Env) (layout : ExecutionLayout logical physical rename)
    (represented : Represents roots state physical)
    (saved : ProtectedCells (callerImage logical rename) roots state physical) (bindings : Bindings roots env)
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (resumes : List Resume) :
    ∃ after next finalEnv,
      Represents roots next after ∧ ProtectedCells (callerImage logical rename) roots next after ∧
      Bindings roots finalEnv ∧ ExecutionLayout finalLogical after rename ∧
      next.graphData = attempts.foldl GraphData.extend state.graphData ∧
      IteratorCalls.Reaches context anchors choices []
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following)
          env, stack, physical⟩, used⟩ resumes)
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following)
          finalEnv, stack, after⟩, finalUsed⟩ resumes) := by
  induction trace generalizing physical state env with
  | nil => exact ⟨physical, state, env, represented, saved, bindings, layout, rfl, .refl⟩
  | @cons before middle after used nextUsed finalUsed s rest ran later ih =>
      obtain ⟨middleClosed, firstGrows⟩ := ran.effects source layout.closed
      have laterGrows := (later.effects source middleClosed).2
      have firstSize : middle.length = before.length := by omega
      have laterSize : after.length = middle.length := by omega
      obtain ⟨operation, inside, isObject, bound, executed⟩ := ran
      obtain ⟨middlePhysical, current, currentEnv, nextMap, agrees, iteration,
        currentRep, currentSaved, currentBindings, currentLayout, data⟩ :=
        iterator_valid_from_logical context extension source choices anchors used nextUsed before physical middle
          rename iterator operation roots state env s (valid s (by simp)) layout inside isObject bound executed
          firstSize represented saved bindings following stack resumes
      have fixedMap : ExecutionLayout middle middlePhysical rename :=
        currentLayout.congr (fun index bounded => (agrees index (by omega)).symm)
      have images : callerImage middle rename = callerImage before rename := by
        funext address
        simp only [callerImage, firstSize]
      have fixedSaved : ProtectedCells (callerImage middle rename) roots current middlePhysical := by
        rw [images]
        exact currentSaved
      obtain ⟨result, next, finalEnv, finalRep, finalSaved, finalBindings, finalLayout, finalData, exited⟩ :=
        ih laterSize (fun value member => valid value (List.mem_cons_of_mem _ member))
          middlePhysical current currentEnv fixedMap currentRep fixedSaved currentBindings
      refine ⟨result, next, finalEnv, finalRep, ?_, finalBindings, finalLayout, ?_, iteration.trans exited⟩
      · rwa [images] at finalSaved
      · simpa only [List.foldl_cons, data] using finalData


theorem iterator_invalid_from_logical (context : Module) (extension : Extends context program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (iterator : Mutable.Value) (logical yielded physical : Mutable.Heap) (used finalUsed anchors : Nat)
    (s : String) (invalid : ¬ ValidAttempt s)
    (ran : NextRun context choices iterator logical used (.ok (text s)) yielded finalUsed)
    (noGrowth : yielded.length = logical.length) (rename : Nat → Nat)
    (layout : ExecutionLayout logical physical rename)
    (env : HeapCalls.Env) (following : List Stmt) (stack : List HeapCalls.Frame) (resumes : List Resume) :
    ∃ after, ExecutionLayout yielded after rename ∧
      IteratorCalls.Reaches context anchors choices []
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following)
          env, stack, physical⟩, used⟩ resumes)
        (.completed (.ok ⟨.error (invalidException s), [], after, anchors⟩) finalUsed resumes) := by
  obtain ⟨after, afterLayout, _, advanced⟩ := ran.scalar_isolated source physical rename layout noGrowth anchors
  have rejected := invalid_attempt (setLocal env "s" (text s)) after s (by simp [setLocal])
    (show s.length ≠ 3 ∨ Strings.isDigit s = false from by
      by_cases length : s.length = 3
      · exact Or.inr (by simpa [ValidAttempt, length] using invalid)
      · exact Or.inl length)
    (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following) anchors stack
  have failed := completion_of_heap (choices := choices) (used := finalUsed) (finishes_extend rejected extension)
    (by intro completion same; cases same; simp [invalidException]) resumes
  refine ⟨after, afterLayout, (advanced (inputIteratorLoop env following stack) resumes).trans ?_⟩
  simpa only [loopCode, inputIteratorLoop, Calls.loopContinuation,
    show Calls.loopControl attemptBody = false from rfl, Bool.false_eq_true, ↓reduceIte,
    List.append_assoc, List.singleton_append] using failed

theorem reject_after_iterator_prefix (context : Module) (extension : Extends context program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (iterator : Mutable.Value) (logical middle yielded : Mutable.Heap) (used nextUsed finalUsed : Nat)
    (attempts : List String) (s : String)
    (trace : YieldPrefix context choices iterator text logical used attempts middle nextUsed)
    (ran : NextRun context choices iterator middle nextUsed (.ok (text s)) yielded finalUsed)
    (noGrowth : yielded.length = logical.length) (valid : ∀ value ∈ attempts, ValidAttempt value)
    (invalid : ¬ ValidAttempt s)
    (physical : Mutable.Heap) (rename : Nat → Nat) (roots : Roots) (state : Storage)
    (env : HeapCalls.Env) (layout : ExecutionLayout logical physical rename)
    (represented : Represents roots state physical)
    (saved : ProtectedCells (callerImage logical rename) roots state physical) (bindings : Bindings roots env)
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (resumes : List Resume) :
    ∃ after, ExecutionLayout yielded after rename ∧
      IteratorCalls.Reaches context anchors choices []
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following)
          env, stack, physical⟩, used⟩ resumes)
        (.completed (.ok ⟨.error (invalidException s), [], after, anchors⟩) finalUsed resumes) := by
  obtain ⟨middleClosed, prefixGrows⟩ := trace.effects source layout.closed
  have lastGrows := (ran.effects source middleClosed).2
  obtain ⟨middlePhysical, _, nextEnv, _, _, _, nextLayout, _, accepted⟩ :=
    process_iterator_prefix context extension source choices iterator logical middle used nextUsed attempts
      trace (by omega) valid physical rename roots state env layout represented saved bindings following anchors stack resumes
  obtain ⟨after, finalLayout, failed⟩ := iterator_invalid_from_logical context extension source choices
    iterator middle yielded middlePhysical nextUsed finalUsed anchors s invalid ran (by omega) rename nextLayout
      nextEnv following stack resumes
  exact ⟨after, finalLayout, accepted.trans failed⟩

end solvers.«79».graph
