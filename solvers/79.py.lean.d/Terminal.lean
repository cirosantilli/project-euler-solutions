import solvers.«79.py.lean.d».Execution
import lang.py.FormalSW.RuntimeExceptionObservation
import lang.py.FormalSW.IteratorTerminal

namespace solvers.«79».source
open FormalSW.PyAst HeapCalls Mutable.Relocation IteratorCalls
open solvers.«79».graph
open data.«project-euler-statements».data.py.project_euler_statements
set_option Elab.async false

/-- Execute initialization, acquisition and a valid prefix before a possibly allocating final call. -/
theorem solve_iterator_prefix_entry
    (context : Module) (extension : Extends context graph.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (receiver iterator : Mutable.Value) (base acquired middle : Mutable.Heap)
    (acquiredUsed nextUsed : Nat) (closed : Closed base)
    (acquisition : AcquireRun context choices receiver base 0 iterator acquired acquiredUsed)
    (attempts : List String)
    (trace : YieldPrefix context choices iterator graph.text acquired acquiredUsed attempts middle nextUsed)
    (noGrowth : middle.length = base.length) (valid : ∀ value ∈ attempts, ValidAttempt value) :
    ∃ physical env, ExecutionLayout middle physical id ∧
      IteratorCalls.Reaches context base.length choices []
        (.running ⟨⟨.operation (.call "solve") [receiver], [], base⟩, 0⟩ [])
        (.running ⟨⟨.block (sequenceLoop (sequenceSource iterator) 0 :: returnGraph) env,
          iteratorBuildStack receiver [], physical⟩, nextUsed⟩ []) := by
  obtain ⟨acquiredClosed, acquisitionGrows, _⟩ := acquisition.effects source closed
  have prefixGrows := (trace.effects source acquiredClosed).2
  obtain ⟨initialized, layout, represented, saved, entered⟩ := solve_iterator_entry context extension source choices
    receiver iterator base acquired 0 acquiredUsed base.length closed acquisition (by omega) [] []
  obtain ⟨physical, _, env, _, _, _, finalLayout, _, accepted⟩ := process_iterator_prefix context extension source choices
    iterator acquired middle acquiredUsed nextUsed attempts trace (by omega) valid initialized id (initialRoots base)
      emptyStorage (initialEnv (buildEnv receiver) base) layout represented saved (initial_bindings (buildEnv receiver) base)
      returnGraph base.length (iteratorBuildStack receiver []) []
  exact ⟨physical, env, finalLayout, entered.trans (by simpa only [relocateValue_id] using accepted)⟩

/-- The final failing method may allocate; its caller-visible graph and exact error survive. -/
theorem solve_iterator_terminal_failure
    (context : Module) (extension : Extends context graph.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (receiver iterator : Mutable.Value) (base acquired middle : Mutable.Heap)
    (acquiredUsed nextUsed : Nat) (closed : Closed base)
    (acquisition : AcquireRun context choices receiver base 0 iterator acquired acquiredUsed)
    (attempts : List String)
    (trace : YieldPrefix context choices iterator graph.text acquired acquiredUsed attempts middle nextUsed)
    (noGrowth : middle.length = base.length) (valid : ∀ value ∈ attempts, ValidAttempt value)
    (failed : Mutable.Heap) (finalUsed : Nat) (error : ExceptionObject)
    (ran : NextRun context choices iterator middle nextUsed (.error error) failed finalUsed)
    (notExhausted : error.typeName ≠ "StopIteration") :
    ∃ after rename, ExecutionLayout failed after rename ∧
      (∀ index < base.length, rename index = index) ∧
      ((IteratorCalls.call context "solve" [receiver] base).run choices).Finishes
        ⟨.error error, [], after, base.length⟩ := by
  obtain ⟨physical, env, layout, entered⟩ := solve_iterator_prefix_entry context extension source choices receiver iterator
    base acquired middle acquiredUsed nextUsed closed acquisition attempts trace noGrowth valid
  obtain ⟨after, rename, extended, failed⟩ := ran.failure_relocates notExhausted source physical id layout base.length
  refine ⟨after, rename, extended.layout, fun index bounded => extended.agrees index (by omega), ?_⟩
  exact (entered.trans (by simpa only [loopCode, inputIteratorLoop, relocateValue_id] using (failed (inputIteratorLoop env returnGraph (iteratorBuildStack receiver [])) []))).finishes (.done rfl)

theorem terminal_rejection_relocates (context : Module) (extension : Extends context program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (iterator : Mutable.Value) (logical yielded physical : Mutable.Heap) (used finalUsed anchors : Nat)
    (s : String) (invalid : ¬ ValidAttempt s)
    (ran : NextRun context choices iterator logical used (.ok (text s)) yielded finalUsed)
    (rename : Nat → Nat)
    (layout : ExecutionLayout logical physical rename)
    (env : HeapCalls.Env) (following : List Stmt) (stack : List HeapCalls.Frame) (resumes : List Resume) :
    ∃ after nextMap, IteratorCalls.Relocation.ExtendedLayout logical.length physical.length rename yielded after nextMap ∧
      IteratorCalls.Reaches context anchors choices []
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following)
          env, stack, physical⟩, used⟩ resumes)
        (.completed (.ok ⟨.error (invalidException s), [], after, anchors⟩) finalUsed resumes) := by
  obtain ⟨after, nextMap, extended, advanced⟩ := ran.scalar_relocates source physical rename layout anchors
  have rejected := invalid_attempt (setLocal env "s" (text s)) after s (by simp [setLocal])
    (show s.length ≠ 3 ∨ Strings.isDigit s = false from by
      by_cases length : s.length = 3
      · exact Or.inr (by simpa [ValidAttempt, length] using invalid)
      · exact Or.inl length)
    (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following) anchors stack
  have failed := completion_of_heap (choices := choices) (used := finalUsed) (finishes_extend rejected extension)
    (by intro completion same; cases same; simp [invalidException]) resumes
  refine ⟨after, nextMap, extended, (advanced (inputIteratorLoop env following stack) resumes).trans ?_⟩
  simpa only [loopCode, inputIteratorLoop, Calls.loopContinuation,
    show Calls.loopControl attemptBody = false from rfl, Bool.false_eq_true, ↓reduceIte,
    List.append_assoc, List.singleton_append] using failed

/-- Reject a malformed scalar after the final iterator method's allocations and mutations. -/
theorem solve_iterator_terminal_rejection
    (context : Module) (extension : Extends context graph.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (receiver iterator : Mutable.Value) (base acquired middle : Mutable.Heap)
    (acquiredUsed nextUsed : Nat) (closed : Closed base)
    (acquisition : AcquireRun context choices receiver base 0 iterator acquired acquiredUsed)
    (attempts : List String)
    (trace : YieldPrefix context choices iterator graph.text acquired acquiredUsed attempts middle nextUsed)
    (noGrowth : middle.length = base.length) (valid : ∀ value ∈ attempts, ValidAttempt value)
    (yielded : Mutable.Heap) (finalUsed : Nat) (s : String)
    (ran : NextRun context choices iterator middle nextUsed (.ok (graph.text s)) yielded finalUsed)
    (invalid : ¬ ValidAttempt s) :
    ∃ after rename, ExecutionLayout yielded after rename ∧
      (∀ index < base.length, rename index = index) ∧
      ((IteratorCalls.call context "solve" [receiver] base).run choices).Finishes
        ⟨.error (invalidException s), [], after, base.length⟩ := by
  obtain ⟨physical, env, layout, entered⟩ := solve_iterator_prefix_entry context extension source choices receiver iterator
    base acquired middle acquiredUsed nextUsed closed acquisition attempts trace noGrowth valid
  obtain ⟨after, rename, extended, rejected⟩ := terminal_rejection_relocates context extension source choices iterator
    middle yielded physical nextUsed finalUsed base.length s invalid ran id layout env returnGraph (iteratorBuildStack receiver []) []
  refine ⟨after, rename, extended.layout, fun index bounded => extended.agrees index (by omega), ?_⟩
  exact (entered.trans (by simpa only [relocateValue_id] using rejected)).finishes (.done rfl)

end solvers.«79».source

namespace solvers.«79».source
open FormalSW.PyAst HeapCalls Mutable.Relocation IteratorCalls
open solvers.«79».graph
open data.«project-euler-statements».data.py.project_euler_statements
set_option Elab.async false

theorem iterator_terminal_failure_equiv (context : Module)
    (solverExtension : Extends context graph.program) (referenceExtension : Extends context p79.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (receiver iterator : Mutable.Value) (base acquired middle failed : Mutable.Heap)
    (acquiredUsed nextUsed finalUsed : Nat) (closed : Closed base)
    (acquisition : AcquireRun context choices receiver base 0 iterator acquired acquiredUsed)
    (attempts : List String)
    (trace : YieldPrefix context choices iterator graph.text acquired acquiredUsed attempts middle nextUsed)
    (noGrowth : middle.length = base.length) (valid : ∀ value ∈ attempts, ValidAttempt value)
    (error : ExceptionObject) (ran : NextRun context choices iterator middle nextUsed (.error error) failed finalUsed)
    (notExhausted : error.typeName ≠ "StopIteration") :
    ((IteratorCalls.call context "solve" [receiver] base).run choices).Equiv
      ((IteratorCalls.call context "naive" [receiver] base).run choices) := by
  obtain ⟨left, leftMap, leftLayout, leftFixed, optimized⟩ := solve_iterator_terminal_failure
    context solverExtension source choices receiver iterator base acquired middle acquiredUsed nextUsed closed
      acquisition attempts trace noGrowth valid failed finalUsed error ran notExhausted
  obtain ⟨right, rightMap, rightLayout, rightFixed, reference⟩ := p79.naive_iterator_terminal_failure
    context referenceExtension source choices receiver iterator base acquired middle acquiredUsed nextUsed closed
      acquisition attempts trace noGrowth valid failed finalUsed error ran notExhausted
  have acquiredClosed := (acquisition.effects source closed).1
  have middleClosed := (trace.effects source acquiredClosed).1
  have grows := (ran.effects source middleClosed).2
  exact Mutable.Call.equiv_of_finishes optimized reference
    (leftLayout.equivalent_exceptions rightLayout base.length (by omega) leftFixed rightFixed error)

theorem iterator_terminal_rejection_equiv (context : Module)
    (solverExtension : Extends context graph.program) (referenceExtension : Extends context p79.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (receiver iterator : Mutable.Value) (base acquired middle yielded : Mutable.Heap)
    (acquiredUsed nextUsed finalUsed : Nat) (closed : Closed base)
    (acquisition : AcquireRun context choices receiver base 0 iterator acquired acquiredUsed)
    (attempts : List String)
    (trace : YieldPrefix context choices iterator graph.text acquired acquiredUsed attempts middle nextUsed)
    (noGrowth : middle.length = base.length) (valid : ∀ value ∈ attempts, ValidAttempt value)
    (s : String) (ran : NextRun context choices iterator middle nextUsed (.ok (graph.text s)) yielded finalUsed)
    (invalid : ¬ ValidAttempt s) :
    ((IteratorCalls.call context "solve" [receiver] base).run choices).Equiv
      ((IteratorCalls.call context "naive" [receiver] base).run choices) := by
  obtain ⟨left, leftMap, leftLayout, leftFixed, optimized⟩ := solve_iterator_terminal_rejection
    context solverExtension source choices receiver iterator base acquired middle acquiredUsed nextUsed closed
      acquisition attempts trace noGrowth valid yielded finalUsed s ran invalid
  have rejected : s.length ≠ 3 ∨ Strings.isDigit s = false := by
    by_cases length : s.length = 3
    · exact Or.inr (by simpa [ValidAttempt, length] using invalid)
    · exact Or.inl length
  obtain ⟨right, rightMap, rightLayout, rightFixed, reference⟩ := p79.naive_iterator_terminal_rejection
    context referenceExtension source choices receiver iterator base acquired middle acquiredUsed nextUsed closed
      acquisition attempts trace noGrowth valid yielded finalUsed s ran rejected
  have acquiredClosed := (acquisition.effects source closed).1
  have middleClosed := (trace.effects source acquiredClosed).1
  have grows := (ran.effects source middleClosed).2
  exact Mutable.Call.equiv_of_finishes optimized reference
    (leftLayout.equivalent_exceptions rightLayout base.length (by omega) leftFixed rightFixed (invalidException s))

end solvers.«79».source
