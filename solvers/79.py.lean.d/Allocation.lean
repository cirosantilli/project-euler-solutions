import solvers.«79.py.lean.d».Execution
import data.«project-euler-statements».data.py.project_euler_statements.«p79.py.lean.d».Allocation
import lang.py.FormalSW.IteratorAllocatingRun

namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls Mutable.Relocation IteratorCalls
set_option Elab.async false

theorem iterator_valid_from_allocating (context : Module) (extension : Extends context program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule) (anchors used finalUsed : Nat)
    (logical physical yielded : Mutable.Heap) (rename : Nat → Nat)
    (iterator : Mutable.Value) (roots : Roots) (state : Storage)
    (env : HeapCalls.Env) (s : String) (valid : ValidAttempt s)
    (layout : ExecutionLayout logical physical rename)
    (ran : IteratorCalls.AllocatingNextRun context choices iterator logical used (.ok (text s)) yielded finalUsed)
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
      Represents roots next after ∧ ProtectedCells (callerImage yielded nextMap) roots next after ∧
      Bindings roots nextEnv ∧ ExecutionLayout yielded after nextMap ∧
      next.graphData = state.graphData.extend s := by
  obtain ⟨other, nextMap, extended, kept, called⟩ := ran.advance_isolated source physical rename layout anchors
  have outside (address : Nat) (separate : ¬ callerImage logical rename address) :
      ∀ index < logical.length, rename index ≠ address := by
    intro index bounded same
    exact separate ⟨index, bounded, same⟩
  have nodes := kept roots.nodes _ (by simp) ⟨layout, represented.nodesFound, outside _ saved.nodes⟩
  have adjacency := kept roots.adjacency _ (by simp) ⟨layout, represented.adjacencyFound, outside _ saved.adjacency⟩
  have degrees := kept roots.degrees _ (by simp) ⟨layout, represented.degreesFound, outside _ saved.degrees⟩
  have neighbors (vertex : String) (member : vertex ∈ state.vertices) := kept (state.address vertex) _ (by simp)
    ⟨layout, represented.neighborsFound vertex member, outside _ (saved.neighbors vertex member)⟩
  have afterRep : Represents roots state other :=
    { represented with
      nodesFound := nodes.saved
      adjacencyFound := adjacency.saved
      degreesFound := degrees.saved
      neighborsFound := fun vertex member => (neighbors vertex member).saved }
  have excluded (address : Nat) (object : Mutable.Object) (privateCell : PrivateLayout yielded other nextMap address object) :
      ¬ callerImage yielded nextMap address := by
    rintro ⟨index, bounded, equal⟩
    exact privateCell.excludes index bounded equal
  have afterSaved : ProtectedCells (callerImage yielded nextMap) roots state other :=
    { inside := by
        rintro address ⟨index, bounded, rfl⟩
        exact extended.layout.runtime.bound index bounded
      nodes := excluded _ _ nodes
      adjacency := excluded _ _ adjacency
      degrees := excluded _ _ degrees
      neighbors := fun vertex member => excluded _ _ (neighbors vertex member) }
  have selected : ∀ index < yielded.length, callerImage yielded nextMap (nextMap index) :=
    fun index bounded => ⟨index, bounded, rfl⟩
  obtain ⟨after, next, nextEnv, body, result, framed, retained, nextBindings, interned, data⟩ :=
    valid_iteration_cells (callerImage yielded nextMap) roots state other env afterRep afterSaved bindings s valid
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


theorem process_allocating_prefix (context : Module) (extension : Extends context program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (iterator : Mutable.Value) (logical finalLogical : Mutable.Heap) (used finalUsed : Nat)
    (attempts : List String)
    (trace : AllocatingYieldPrefix context choices iterator text logical used attempts finalLogical finalUsed)
    (valid : ∀ s ∈ attempts, ValidAttempt s)
    (physical : Mutable.Heap) (rename : Nat → Nat) (roots : Roots) (state : Storage)
    (env : HeapCalls.Env) (layout : ExecutionLayout logical physical rename)
    (represented : Represents roots state physical)
    (saved : ProtectedCells (callerImage logical rename) roots state physical) (bindings : Bindings roots env)
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (resumes : List Resume) :
    ∃ after next finalEnv finalMap,
      (∀ index < logical.length, finalMap index = rename index) ∧
      Represents roots next after ∧ ProtectedCells (callerImage finalLogical finalMap) roots next after ∧
      Bindings roots finalEnv ∧ ExecutionLayout finalLogical after finalMap ∧
      next.graphData = attempts.foldl GraphData.extend state.graphData ∧
      IteratorCalls.Reaches context anchors choices []
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following)
          env, stack, physical⟩, used⟩ resumes)
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following)
          finalEnv, stack, after⟩, finalUsed⟩ resumes) := by
  induction trace generalizing physical rename state env with
  | nil => exact ⟨physical, state, env, rename, fun _ _ => rfl, represented, saved, bindings, layout, rfl, .refl⟩
  | @cons before middle after used nextUsed finalUsed s rest ran later ih =>
    have firstGrows := (ran.nextRun.effects source layout.closed).2
    obtain ⟨middlePhysical, current, currentEnv, nextMap, agrees, iteration,
      currentRep, currentSaved, currentBindings, currentLayout, data⟩ :=
      iterator_valid_from_allocating context extension source choices anchors used nextUsed before physical middle
        rename iterator roots state env s (valid s (by simp)) layout ran represented saved bindings following stack resumes
    have sameIterator := relocateValue_congr nextMap rename before.length iterator agrees ran.nextRun.choose_spec.1
    obtain ⟨result, next, finalEnv, finalMap, finalAgrees, finalRep, finalSaved, finalBindings, finalLayout, finalData, exited⟩ :=
      ih (fun value member => valid value (List.mem_cons_of_mem _ member))
        middlePhysical nextMap current currentEnv currentLayout currentRep currentSaved currentBindings
    refine ⟨result, next, finalEnv, finalMap, ?_, finalRep, finalSaved, finalBindings, finalLayout, ?_, iteration.trans ?_⟩
    · intro index bounded
      exact (finalAgrees index (Nat.lt_of_lt_of_le bounded firstGrows)).trans (agrees index bounded)
    · simpa only [List.foldl_cons, data] using finalData
    · simpa only [sameIterator] using exited

end solvers.«79».graph
