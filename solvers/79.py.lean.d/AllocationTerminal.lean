import solvers.«79.py.lean.d».Terminal
import solvers.«79.py.lean.d».Allocation
import solvers.«79.py.lean.d».Acquisition
import data.«project-euler-statements».data.py.project_euler_statements.«p79.py.lean.d».AllocationTerminal
import lang.py.FormalSW.RuntimeExceptionObservation
import lang.py.FormalSW.IteratorTerminal

namespace solvers.«79».source
open FormalSW.PyAst HeapCalls Mutable.Relocation IteratorCalls
open solvers.«79».graph
open data.«project-euler-statements».data.py.project_euler_statements
set_option Elab.async false

/-- Execute initialization, acquisition and a valid prefix before a possibly allocating final call. -/
theorem solve_allocating_prefix_entry
    (context : Module) (extension : Extends context graph.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (receiver iterator : Mutable.Value) (base acquired middle : Mutable.Heap)
    (acquiredUsed nextUsed : Nat) (closed : Closed base)
    (acquisition : AllocatingAcquireRun context choices receiver base 0 iterator acquired acquiredUsed)
    (attempts : List String)
    (trace : AllocatingYieldPrefix context choices iterator graph.text acquired acquiredUsed attempts middle nextUsed)
    (valid : ∀ value ∈ attempts, ValidAttempt value) :
    ∃ physical env rename, ExecutionLayout middle physical rename ∧
      (∀ index < base.length, rename index = index) ∧
      IteratorCalls.Reaches context base.length choices []
        (.running ⟨⟨.operation (.call "solve") [receiver], [], base⟩, 0⟩ [])
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: returnGraph) env,
          iteratorBuildStack receiver [], physical⟩, nextUsed⟩ []) := by
  obtain ⟨acquiredClosed, acquisitionGrows, iteratorInside⟩ := acquisition.acquireRun.effects source closed
  obtain ⟨initialized, entryMap, layout, entryFixed, represented, saved, entered⟩ := solve_allocating_entry context extension source choices
    receiver iterator base acquired 0 acquiredUsed base.length closed acquisition [] []
  obtain ⟨physical, _, env, rename, agrees, _, _, _, finalLayout, _, accepted⟩ := process_allocating_prefix context extension source choices
    iterator acquired middle acquiredUsed nextUsed attempts trace valid initialized entryMap (initialRoots base)
      emptyStorage (initialEnv (buildEnv receiver) base) layout represented saved (initial_bindings (buildEnv receiver) base)
      returnGraph base.length (iteratorBuildStack receiver []) []
  have sameIterator := relocateValue_congr rename entryMap acquired.length iterator agrees iteratorInside
  refine ⟨physical, env, rename, finalLayout, ?_, ?_⟩
  · intro index bounded
    exact (agrees index (by omega)).trans (entryFixed index bounded)
  · simpa only [sameIterator] using entered.trans accepted

/-- The final failing method may allocate; its caller-visible graph and exact error survive. -/
theorem solve_allocating_terminal_failure
    (context : Module) (extension : Extends context graph.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (receiver iterator : Mutable.Value) (base acquired middle : Mutable.Heap)
    (acquiredUsed nextUsed : Nat) (closed : Closed base)
    (acquisition : AllocatingAcquireRun context choices receiver base 0 iterator acquired acquiredUsed)
    (attempts : List String)
    (trace : AllocatingYieldPrefix context choices iterator graph.text acquired acquiredUsed attempts middle nextUsed)
    (valid : ∀ value ∈ attempts, ValidAttempt value)
    (failed : Mutable.Heap) (finalUsed : Nat) (error : ExceptionObject)
    (ran : NextRun context choices iterator middle nextUsed (.error error) failed finalUsed)
    (notExhausted : error.typeName ≠ "StopIteration") :
    ∃ after rename, ExecutionLayout failed after rename ∧
      (∀ index < base.length, rename index = index) ∧
      ((IteratorCalls.call context "solve" [receiver] base).run choices).Finishes
        ⟨.error error, [], after, base.length⟩ := by
  obtain ⟨physical, env, entryMap, layout, entryFixed, entered⟩ := solve_allocating_prefix_entry context extension source choices receiver iterator
    base acquired middle acquiredUsed nextUsed closed acquisition attempts trace valid
  obtain ⟨acquiredClosed, acquisitionGrows, _⟩ := acquisition.acquireRun.effects source closed
  have prefixGrows := (trace.yieldPrefix.effects source acquiredClosed).2
  obtain ⟨after, rename, extended, failed⟩ := ran.failure_relocates notExhausted source physical entryMap layout base.length
  refine ⟨after, rename, extended.layout, fun index bounded => (extended.agrees index (by omega)).trans (entryFixed index bounded), ?_⟩
  exact (entered.trans (by simpa only [loopCode, inputIteratorLoop, relocateValue_id] using (failed (inputIteratorLoop env returnGraph (iteratorBuildStack receiver [])) []))).finishes (.done rfl)

/-- Reject a malformed scalar after the final iterator method's allocations and mutations. -/
theorem solve_allocating_terminal_rejection
    (context : Module) (extension : Extends context graph.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (receiver iterator : Mutable.Value) (base acquired middle : Mutable.Heap)
    (acquiredUsed nextUsed : Nat) (closed : Closed base)
    (acquisition : AllocatingAcquireRun context choices receiver base 0 iterator acquired acquiredUsed)
    (attempts : List String)
    (trace : AllocatingYieldPrefix context choices iterator graph.text acquired acquiredUsed attempts middle nextUsed)
    (valid : ∀ value ∈ attempts, ValidAttempt value)
    (yielded : Mutable.Heap) (finalUsed : Nat) (s : String)
    (ran : NextRun context choices iterator middle nextUsed (.ok (graph.text s)) yielded finalUsed)
    (invalid : ¬ ValidAttempt s) :
    ∃ after rename, ExecutionLayout yielded after rename ∧
      (∀ index < base.length, rename index = index) ∧
      ((IteratorCalls.call context "solve" [receiver] base).run choices).Finishes
        ⟨.error (invalidException s), [], after, base.length⟩ := by
  obtain ⟨physical, env, entryMap, layout, entryFixed, entered⟩ := solve_allocating_prefix_entry context extension source choices receiver iterator
    base acquired middle acquiredUsed nextUsed closed acquisition attempts trace valid
  obtain ⟨acquiredClosed, acquisitionGrows, _⟩ := acquisition.acquireRun.effects source closed
  have prefixGrows := (trace.yieldPrefix.effects source acquiredClosed).2
  obtain ⟨after, rename, extended, rejected⟩ := terminal_rejection_relocates context extension source choices iterator
    middle yielded physical nextUsed finalUsed base.length s invalid ran entryMap layout env returnGraph (iteratorBuildStack receiver []) []
  refine ⟨after, rename, extended.layout, fun index bounded => (extended.agrees index (by omega)).trans (entryFixed index bounded), ?_⟩
  exact (entered.trans (by simpa only [relocateValue_id] using rejected)).finishes (.done rfl)

end solvers.«79».source

namespace solvers.«79».source
open FormalSW.PyAst HeapCalls Mutable.Relocation IteratorCalls
open solvers.«79».graph
open data.«project-euler-statements».data.py.project_euler_statements
set_option Elab.async false

theorem allocating_terminal_failure_equiv (context : Module)
    (solverExtension : Extends context graph.program) (referenceExtension : Extends context p79.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (receiver iterator : Mutable.Value) (base acquired middle failed : Mutable.Heap)
    (acquiredUsed nextUsed finalUsed : Nat) (closed : Closed base)
    (acquisition : AllocatingAcquireRun context choices receiver base 0 iterator acquired acquiredUsed)
    (attempts : List String)
    (trace : AllocatingYieldPrefix context choices iterator graph.text acquired acquiredUsed attempts middle nextUsed)
    (valid : ∀ value ∈ attempts, ValidAttempt value)
    (error : ExceptionObject) (ran : NextRun context choices iterator middle nextUsed (.error error) failed finalUsed)
    (notExhausted : error.typeName ≠ "StopIteration") :
    ((IteratorCalls.call context "solve" [receiver] base).run choices).Equiv
      ((IteratorCalls.call context "naive" [receiver] base).run choices) := by
  obtain ⟨left, leftMap, leftLayout, leftFixed, optimized⟩ := solve_allocating_terminal_failure
    context solverExtension source choices receiver iterator base acquired middle acquiredUsed nextUsed closed
      acquisition attempts trace valid failed finalUsed error ran notExhausted
  obtain ⟨right, rightMap, rightLayout, rightFixed, reference⟩ := p79.naive_allocating_terminal_failure
    context referenceExtension source choices receiver iterator base acquired middle acquiredUsed nextUsed closed
      acquisition attempts trace valid failed finalUsed error ran notExhausted
  obtain ⟨acquiredClosed, acquisitionGrows, _⟩ := acquisition.acquireRun.effects source closed
  obtain ⟨middleClosed, prefixGrows⟩ := trace.yieldPrefix.effects source acquiredClosed
  have grows := (ran.effects source middleClosed).2
  exact Mutable.Call.equiv_of_finishes optimized reference
    (leftLayout.equivalent_exceptions rightLayout base.length (by omega) leftFixed rightFixed error)

theorem allocating_terminal_rejection_equiv (context : Module)
    (solverExtension : Extends context graph.program) (referenceExtension : Extends context p79.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (receiver iterator : Mutable.Value) (base acquired middle yielded : Mutable.Heap)
    (acquiredUsed nextUsed finalUsed : Nat) (closed : Closed base)
    (acquisition : AllocatingAcquireRun context choices receiver base 0 iterator acquired acquiredUsed)
    (attempts : List String)
    (trace : AllocatingYieldPrefix context choices iterator graph.text acquired acquiredUsed attempts middle nextUsed)
    (valid : ∀ value ∈ attempts, ValidAttempt value)
    (s : String) (ran : NextRun context choices iterator middle nextUsed (.ok (graph.text s)) yielded finalUsed)
    (invalid : ¬ ValidAttempt s) :
    ((IteratorCalls.call context "solve" [receiver] base).run choices).Equiv
      ((IteratorCalls.call context "naive" [receiver] base).run choices) := by
  obtain ⟨left, leftMap, leftLayout, leftFixed, optimized⟩ := solve_allocating_terminal_rejection
    context solverExtension source choices receiver iterator base acquired middle acquiredUsed nextUsed closed
      acquisition attempts trace valid yielded finalUsed s ran invalid
  have rejected : s.length ≠ 3 ∨ Strings.isDigit s = false := by
    by_cases length : s.length = 3
    · exact Or.inr (by simpa [ValidAttempt, length] using invalid)
    · exact Or.inl length
  obtain ⟨right, rightMap, rightLayout, rightFixed, reference⟩ := p79.naive_allocating_terminal_rejection
    context referenceExtension source choices receiver iterator base acquired middle acquiredUsed nextUsed closed
      acquisition attempts trace valid yielded finalUsed s ran rejected
  obtain ⟨acquiredClosed, acquisitionGrows, _⟩ := acquisition.acquireRun.effects source closed
  obtain ⟨middleClosed, prefixGrows⟩ := trace.yieldPrefix.effects source acquiredClosed
  have grows := (ran.effects source middleClosed).2
  exact Mutable.Call.equiv_of_finishes optimized reference
    (leftLayout.equivalent_exceptions rightLayout base.length (by omega) leftFixed rightFixed (invalidException s))

end solvers.«79».source
