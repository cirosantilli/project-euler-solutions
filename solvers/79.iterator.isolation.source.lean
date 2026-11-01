import solvers.«79.iterator.source»
import lang.py.FormalSW.IteratorExecutionIsolation

/-! Logical iterator execution preserves the private graph before the source body updates it. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls Mutable.Relocation
set_option Elab.async false

def callerImage (logical : Mutable.Heap) (rename : Nat → Nat) (address : Nat) : Prop :=
  ∃ index < logical.length, rename index = address

/-- The same transported method trace protects all graph containers, including neighbor sets. -/
theorem iterator_valid_from_logical (context : Module) (extension : Extends context program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule) (anchors used finalUsed : Nat)
    (logical physical yielded : Mutable.Heap) (rename : Nat → Nat)
    (iterator : Mutable.Value) (operation : Operation) (roots : Roots) (state : Storage)
    (env : HeapCalls.Env) (s : String) (valid : ValidAttempt s)
    (layout : ExecutionLayout logical physical rename)
    (inside : iterator.edge.Within logical.length)
    (isObject : IteratorCalls.isInstance logical iterator = true)
    (bound : ProtocolCalls.nextMethod context iterator logical = some (.ok (some operation), logical))
    (executed : IteratorCalls.Reaches context logical.length choices []
      (.running ⟨⟨.operation operation [], [], logical⟩, used⟩ [])
      (.completed (.ok ⟨.ok (text s), [], yielded, logical.length⟩) finalUsed []))
    (noGrowth : yielded.length = logical.length)
    (represented : Represents roots state physical)
    (saved : ProtectedCells (callerImage logical rename) roots state physical)
    (bindings : Bindings roots env)
    (following : List Stmt) (stack : List HeapCalls.Frame) (resumes : List IteratorCalls.Resume) :
    ∃ after next nextEnv nextMap,
      (∀ index < logical.length, nextMap index = rename index) ∧
      IteratorCalls.Reaches context anchors choices []
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following)
          env, stack, physical⟩, used⟩ resumes)
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following)
          nextEnv, stack, after⟩, finalUsed⟩ resumes) ∧
      Represents roots next after ∧ ProtectedCells (callerImage logical rename) roots next after ∧
      Bindings roots nextEnv ∧ ExecutionLayout yielded after nextMap ∧
      next.graphData = state.graphData.extend s := by
  obtain ⟨other, nextMap, extended, kept, called⟩ := IteratorCalls.Relocation.advance_completion_isolated
    context choices logical physical rename logical.length anchors used finalUsed iterator operation (.ok (text s))
    yielded [] source layout inside (by simp [HeapCalls.Relocation.GlobalsWithin]) isObject bound executed noGrowth
  have outside (address : Nat) (separate : ¬ callerImage logical rename address) :
      ∀ index < logical.length, rename index ≠ address := by
    intro index bounded same
    exact separate ⟨index, bounded, same⟩
  have afterRep : Represents roots state other :=
    { represented with
      nodesFound := (kept roots.nodes _ ⟨layout, represented.nodesFound, outside _ saved.nodes⟩).saved
      adjacencyFound := (kept roots.adjacency _ ⟨layout, represented.adjacencyFound, outside _ saved.adjacency⟩).saved
      degreesFound := (kept roots.degrees _ ⟨layout, represented.degreesFound, outside _ saved.degrees⟩).saved
      neighborsFound := fun vertex member => (kept (state.address vertex) _
        ⟨layout, represented.neighborsFound vertex member, outside _ (saved.neighbors vertex member)⟩).saved }
  have afterSaved : ProtectedCells (callerImage logical rename) roots state other :=
    { saved with inside := fun address selected =>
      Nat.lt_of_lt_of_le (saved.inside address selected) extended.physicalGrows }
  have selected : ∀ index < yielded.length, callerImage logical rename (nextMap index) := by
    intro index bounded
    have old : index < logical.length := by omega
    exact ⟨index, old, (extended.agrees index old).symm⟩
  obtain ⟨after, next, nextEnv, body, result, framed, retained, nextBindings, interned, data⟩ :=
    valid_iteration_cells (callerImage logical rename) roots state other env afterRep afterSaved bindings s valid
  have completion : IteratorCalls.Reaches context anchors choices []
      (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following) env,
        stack, physical⟩, used⟩ resumes)
      (.completed (.ok ⟨.ok (text s), [], other, anchors⟩) finalUsed
        (.advance (inputIteratorLoop env following stack) (relocateValue rename iterator) :: resumes)) := by
    simpa only [HeapCalls.Relocation.relocateGlobals, List.map_nil, Except.map, text, relocateValue] using
      called (inputIteratorLoop env following stack) resumes
  refine ⟨after, next, nextEnv, nextMap, extended.agrees, ?_, result, framed, nextBindings,
    extended.layout.preserves (retained.runtimeLayout extended.layout.runtime selected) interned, data⟩
  apply (IteratorCalls.advanced completion).trans
  have resumed := IteratorCalls.Reaches.of_heap (choices := choices) (used := finalUsed)
    ((body.extend extension) (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following)
      anchors stack) resumes
  simpa only [IteratorCalls.loopCode, inputIteratorLoop, Calls.loopContinuation,
    show Calls.loopControl attemptBody = false from rfl, Bool.false_eq_true, ↓reduceIte,
    List.append_assoc, List.singleton_append] using resumed

end solvers.«79».graph
