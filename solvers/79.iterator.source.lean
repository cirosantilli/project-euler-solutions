import solvers.«79.iterator.attempt.source»
import lang.py.FormalSW.IteratorRules

/-! The graph-builder's source loop composes actual protocol execution with graph updates.
The iterator may mutate and allocate; graph updates preserve its disjoint resulting cells. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls
set_option Elab.async false

abbrev inputIteratorLoop (env : HeapCalls.Env) (following : List Stmt)
    (stack : List HeapCalls.Frame) : IteratorCalls.Loop :=
  ⟨"s", attemptBody, [], following, env, stack⟩

theorem iterator_valid_iteration (context : Module) (extension : Extends context program)
    (choices : ChoiceSchedule) (anchors used : Nat) (keep : Nat → Prop)
    (roots : Roots) (state : Storage)
    (before yielded : Mutable.Heap) (iterator : Mutable.Value) (operation : Operation)
    (env : HeapCalls.Env) (s : String) (valid : ValidAttempt s)
    (isObject : IteratorCalls.isInstance before iterator = true)
    (bound : ProtocolCalls.nextMethod context iterator before = some (.ok (some operation), before))
    (executed : IteratorCalls.OperationCompletes context choices used operation
      before (.ok (text s)) yielded)
    (represented : Represents roots state yielded) (saved : ProtectedCells keep roots state yielded)
    (bindings : Bindings roots env) (following : List Stmt)
    (stack : List HeapCalls.Frame) (resumes : List IteratorCalls.Resume) :
    ∃ after next nextEnv finalUsed,
      IteratorCalls.Reaches context anchors choices []
        (.running ⟨⟨.block (sequenceLoop (sequenceSource iterator) 0 :: following) env,
          stack, before⟩, used⟩ resumes)
        (.running ⟨⟨.block (sequenceLoop (sequenceSource iterator) 0 :: following) nextEnv,
          stack, after⟩, finalUsed⟩ resumes) ∧
      Represents roots next after ∧ ProtectedCells keep roots next after ∧
      PreservesCells keep yielded after ∧ Bindings roots nextEnv ∧
      next.graphData = state.graphData.extend s := by
  obtain ⟨after, next, nextEnv, body, result, framed, retained, nextBindings, data⟩ :=
    valid_iteration_cells keep roots state yielded env represented saved bindings s valid
  obtain ⟨finalUsed, continued⟩ := IteratorCalls.advance_continue
    (loop := inputIteratorLoop env following stack) (anchors := anchors) (used := used)
    (resumes := resumes) isObject bound executed rfl (body.extend extension)
  exact ⟨after, next, nextEnv, finalUsed, continued, result, framed, retained, nextBindings, data⟩

/-- An invalid yielded string stops immediately after validation, retaining next's writes. -/
theorem iterator_invalid_attempt (context : Module) (extension : Extends context program)
    (choices : ChoiceSchedule) (anchors used : Nat) (before yielded : Mutable.Heap)
    (iterator : Mutable.Value)
    (operation : Operation) (env : HeapCalls.Env) (s : String) (invalid : ¬ ValidAttempt s)
    (isObject : IteratorCalls.isInstance before iterator = true)
    (bound : ProtocolCalls.nextMethod context iterator before = some (.ok (some operation), before))
    (executed : IteratorCalls.OperationCompletes context choices used operation
      before (.ok (text s)) yielded)
    (following : List Stmt) (stack : List HeapCalls.Frame)
    (resumes : List IteratorCalls.Resume) :
    ∃ finalUsed, IteratorCalls.Reaches context anchors choices []
      (.running ⟨⟨.block (sequenceLoop (sequenceSource iterator) 0 :: following) env,
        stack, before⟩, used⟩ resumes)
      (.completed (.ok ⟨.error (invalidException s), [], yielded, anchors⟩) finalUsed resumes) := by
  obtain ⟨middleUsed, advanced⟩ := IteratorCalls.advance_returns
    (loop := inputIteratorLoop env following stack) (anchors := anchors) (used := used)
    (resumes := resumes) isObject bound executed
  have rejected := invalid_attempt (setLocal env "s" (text s)) yielded s (by simp [setLocal])
    (show s.length ≠ 3 ∨ Strings.isDigit s = false from by
      by_cases length : s.length = 3
      · exact Or.inr (by simpa [ValidAttempt, length] using invalid)
      · exact Or.inl length)
    (sequenceLoop (sequenceSource iterator) 0 :: following) anchors stack
  obtain ⟨finalUsed, failed⟩ := IteratorCalls.completion_of_set
    (SetCalls.finishes_of_heap (finishes_extend rejected extension) choices middleUsed)
    (by intro completion same; cases same; simp [invalidException]) resumes
  refine ⟨finalUsed, advanced.trans ?_⟩
  simpa only [IteratorCalls.loopCode, inputIteratorLoop, Calls.loopContinuation,
    show Calls.loopControl attemptBody = false from rfl, Bool.false_eq_true, ↓reduceIte,
    List.append_assoc, List.singleton_append] using failed


/-- The source for statement evaluates and acquires its input exactly once. -/
theorem iterator_enter (context : Module) (choices : ChoiceSchedule) (anchors used : Nat)
    (before after : Mutable.Heap) (receiver iterator : Mutable.Value)
    (operation advance : Operation) (env : HeapCalls.Env)
    (getInput : env "attempts" = some receiver)
    (isObject : IteratorCalls.isInstance before receiver = true)
    (bound : ProtocolCalls.iterMethod context receiver before = some (.ok operation, before))
    (executed : IteratorCalls.OperationCompletes context choices used operation
      before (.ok iterator) after)
    (valid : ProtocolCalls.nextMethod context iterator after = some (.ok (some advance), after))
    (following : List Stmt) (stack : List HeapCalls.Frame)
    (resumes : List IteratorCalls.Resume) :
    ∃ finalUsed, IteratorCalls.Reaches context anchors choices []
      (.running ⟨⟨.block (.forIn (.name "s") (.name "attempts") attemptBody [] ::
        following) env, stack, before⟩, used⟩ resumes)
      (.running ⟨⟨.block (sequenceLoop (sequenceSource iterator) 0 :: following) env,
        stack, after⟩, finalUsed⟩ resumes) :=
  IteratorCalls.for_instance (loop := inputIteratorLoop env following stack)
    (by intro expressions impossible; cases impossible)
    (Evaluates.lookup getInput) isObject bound executed valid

/-- Non-exhaustion failures escape before validation or updates to the graph or checked list. -/
theorem iterator_failed (context : Module) (choices : ChoiceSchedule) (anchors used : Nat)
    (before after : Mutable.Heap) (iterator : Mutable.Value) (operation : Operation)
    (env : HeapCalls.Env) (error : ExceptionObject)
    (notExhausted : error.typeName ≠ "StopIteration")
    (isObject : IteratorCalls.isInstance before iterator = true)
    (bound : ProtocolCalls.nextMethod context iterator before =
      some (.ok (some operation), before))
    (executed : IteratorCalls.OperationCompletes context choices used operation
      before (.error error) after)
    (following : List Stmt) (stack : List HeapCalls.Frame)
    (resumes : List IteratorCalls.Resume) :
    ∃ finalUsed, IteratorCalls.Reaches context anchors choices []
      (.running ⟨⟨.block (sequenceLoop (sequenceSource iterator) 0 :: following) env,
        stack, before⟩, used⟩ resumes)
      (.completed (.ok ⟨.error error, [], after, anchors⟩) finalUsed resumes) :=
  IteratorCalls.advance_failed (loop := inputIteratorLoop env following stack)
    isObject bound executed notExhausted

/-- Exhaustion finishes the consuming loop with the actual final iterator state. -/
theorem iterator_exhausted (context : Module) (choices : ChoiceSchedule) (anchors used : Nat)
    (before after : Mutable.Heap) (iterator : Mutable.Value) (operation : Operation)
    (env : HeapCalls.Env) (args : List Constant)
    (isObject : IteratorCalls.isInstance before iterator = true)
    (bound : ProtocolCalls.nextMethod context iterator before = some (.ok (some operation), before))
    (executed : IteratorCalls.OperationCompletes context choices used operation before
      (.error ⟨"StopIteration", args⟩) after)
    (following : List Stmt) (stack : List HeapCalls.Frame)
    (resumes : List IteratorCalls.Resume) :
    ∃ finalUsed, IteratorCalls.Reaches context anchors choices []
      (.running ⟨⟨.block (sequenceLoop (sequenceSource iterator) 0 :: following) env,
        stack, before⟩, used⟩ resumes)
      (.running ⟨⟨.block following env, stack, after⟩, finalUsed⟩ resumes) :=
  IteratorCalls.advance_exhausted (loop := inputIteratorLoop env following stack)
    isObject bound executed

end solvers.«79».graph
