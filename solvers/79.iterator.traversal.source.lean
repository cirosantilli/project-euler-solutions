import solvers.«79.iterator.prefix.source»

/-! Graph construction composes a complete finite iterator traversal and preserves its final caller state. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls Mutable.Relocation IteratorCalls
set_option Elab.async false

theorem iterator_exit_from_logical (context : Module) (source : IteratorCalls.Relocation.ModuleSource context)
    (choices : ChoiceSchedule) (iterator : Mutable.Value) (logical finalLogical physical : Mutable.Heap)
    (used finalUsed anchors : Nat) (error : ExceptionObject)
    (ran : NextRun context choices iterator logical used (.error error) finalLogical finalUsed)
    (noGrowth : finalLogical.length = logical.length) (rename : Nat → Nat)
    (layout : ExecutionLayout logical physical rename) (roots : Roots) (state : Storage)
    (represented : Represents roots state physical)
    (saved : ProtectedCells (callerImage logical rename) roots state physical)
    (env : HeapCalls.Env) (following : List Stmt) (stack : List HeapCalls.Frame) (resumes : List Resume) :
    ∃ after, Represents roots state after ∧ ProtectedCells (callerImage logical rename) roots state after ∧
      ExecutionLayout finalLogical after rename ∧
      ExitResult context choices anchors
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following)
          env, stack, physical⟩, used⟩ resumes)
        (inputIteratorLoop env following stack) resumes after finalUsed (exitOutcome error) := by
  obtain ⟨after, afterLayout, kept, exited⟩ := ran.exit_isolated source physical rename layout noGrowth anchors
  have outside (address : Nat) (separate : ¬ callerImage logical rename address) :
      ∀ index < logical.length, rename index ≠ address := by
    intro index bounded equal
    exact separate ⟨index, bounded, equal⟩
  have afterRep : Represents roots state after :=
    { represented with
      nodesFound := (kept roots.nodes _ ⟨layout, represented.nodesFound, outside _ saved.nodes⟩).saved
      adjacencyFound := (kept roots.adjacency _ ⟨layout, represented.adjacencyFound, outside _ saved.adjacency⟩).saved
      degreesFound := (kept roots.degrees _ ⟨layout, represented.degreesFound, outside _ saved.degrees⟩).saved
      neighborsFound := fun vertex member => (kept (state.address vertex) _
        ⟨layout, represented.neighborsFound vertex member, outside _ (saved.neighbors vertex member)⟩).saved }
  have afterSaved : ProtectedCells (callerImage logical rename) roots state after := by
    refine { saved with inside := ?_ }
    intro address selected
    obtain ⟨index, bounded, rfl⟩ := selected
    exact afterLayout.runtime.bound index (by omega)
  exact ⟨after, afterRep, afterSaved, afterLayout, exited (inputIteratorLoop env following stack) resumes⟩

theorem process_iterator_traversal (context : Module) (extension : Extends context program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (iterator : Mutable.Value) (logical finalLogical : Mutable.Heap) (used finalUsed : Nat)
    (attempts : List String) (outcome : Except ExceptionObject Unit)
    (trace : Traversal context choices iterator text logical used attempts outcome finalLogical finalUsed)
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
      ExitResult context choices anchors
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following)
          env, stack, physical⟩, used⟩ resumes)
        (inputIteratorLoop finalEnv following stack) resumes after finalUsed outcome := by
  obtain ⟨middle, nextUsed, error, first, last, result⟩ := trace.decompose
  obtain ⟨middleClosed, prefixGrows⟩ := first.effects source layout.closed
  have lastGrows := (last.effects source middleClosed).2
  have sameSize : middle.length = logical.length := by omega
  have images : callerImage middle rename = callerImage logical rename := by
    funext address
    simp only [callerImage, sameSize]
  obtain ⟨middlePhysical, next, nextEnv, nextRep, nextSaved, nextBindings, nextLayout, data, accepted⟩ :=
    process_iterator_prefix context extension source choices iterator logical middle used nextUsed attempts
      first sameSize valid physical rename roots state env layout represented saved bindings following anchors stack resumes
  obtain ⟨after, finalRep, finalSaved, finalLayout, exited⟩ := iterator_exit_from_logical context source choices
    iterator middle finalLogical middlePhysical nextUsed finalUsed anchors error last (by omega) rename nextLayout
      roots next nextRep (by rwa [images]) nextEnv following stack resumes
  refine ⟨after, next, nextEnv, finalRep, ?_, nextBindings, finalLayout, data, ?_⟩
  · rwa [images] at finalSaved
  · rw [← result]
    exact exited.before accepted

end solvers.«79».graph
