import solvers.«79.py.lean.d».ConsumerAgreement
import solvers.«79.py.lean.d».Allocation
import data.«project-euler-statements».data.py.project_euler_statements.«p79.py.lean.d».AllocationResult
import lang.py.FormalSW.RuntimeExceptionObservation
import lang.py.FormalSW.SetExecutionExtension

namespace solvers.«79».source
open FormalSW.PyAst HeapCalls Mutable.Relocation IteratorCalls
open solvers.«79».graph solvers.«79».topo
open ProjectEulerStatements.TopologicalOrder
open data.«project-euler-statements».data.py.project_euler_statements
set_option Elab.async false

theorem solve_allocating_result (context : Module) (extension : Extends context graph.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (receiver iterator : Mutable.Value) (base acquired middle yielded : Mutable.Heap)
    (acquiredUsed nextUsed finalUsed : Nat) (closed : Closed base)
    (acquisition : AcquireRun context choices receiver base 0 iterator acquired acquiredUsed)
    (attempts : List String)
    (trace : AllocatingYieldPrefix context choices iterator graph.text acquired acquiredUsed attempts middle nextUsed)
    (args : List Constant)
    (last : AllocatingNextRun context choices iterator middle nextUsed (.error ⟨"StopIteration", args⟩) yielded finalUsed)
    (acquisitionSize : acquired.length = base.length) (valid : ∀ value ∈ attempts, ValidAttempt value) :
    ∃ after finalMap, RuntimeLayout yielded after finalMap ∧
      (∀ index < base.length, finalMap index = index) ∧
      ((IteratorCalls.call context "solve" [receiver] base).run choices).Finishes
        ⟨p79.checkedOutcome attempts, [], after, base.length⟩ := by
  obtain ⟨acquiredClosed, _, iteratorInside⟩ := acquisition.effects source closed
  have prefixGrows := (trace.yieldPrefix.effects source acquiredClosed).2
  obtain ⟨initialized, layout, represented, saved, entered⟩ := solve_iterator_entry context extension source choices
    receiver iterator base acquired 0 acquiredUsed base.length closed acquisition acquisitionSize [] []
  obtain ⟨prefixHeap, next, finalEnv, prefixMap, agrees, prefixRep, prefixSaved, bindings, prefixLayout, data, consumed⟩ :=
    process_allocating_prefix context extension source choices iterator acquired middle acquiredUsed nextUsed attempts trace valid
      initialized id (initialRoots base) emptyStorage (initialEnv (buildEnv receiver) base) layout represented saved
      (initial_bindings (buildEnv receiver) base) returnGraph base.length (iteratorBuildStack receiver []) []
  obtain ⟨middleHeap, finalMap, extended, kept, exited⟩ := last.exit_isolated source prefixHeap prefixMap prefixLayout base.length
  have outside (address : Nat) (separate : ¬ callerImage middle prefixMap address) :
      ∀ index < middle.length, prefixMap index ≠ address := by
    intro index bounded same
    exact separate ⟨index, bounded, same⟩
  have nextRep : Represents (initialRoots base) next middleHeap :=
    { prefixRep with
      nodesFound := (kept (initialRoots base).nodes _ (by simp)
        ⟨prefixLayout, prefixRep.nodesFound, outside _ prefixSaved.nodes⟩).saved
      adjacencyFound := (kept (initialRoots base).adjacency _ (by simp)
        ⟨prefixLayout, prefixRep.adjacencyFound, outside _ prefixSaved.adjacency⟩).saved
      degreesFound := (kept (initialRoots base).degrees _ (by simp)
        ⟨prefixLayout, prefixRep.degreesFound, outside _ prefixSaved.degrees⟩).saved
      neighborsFound := fun vertex member => (kept (next.address vertex) _ (by simp)
        ⟨prefixLayout, prefixRep.neighborsFound vertex member, outside _ (prefixSaved.neighbors vertex member)⟩).saved }
  have sameIterator : relocateValue prefixMap iterator = iterator := by
    simpa only [relocateValue_id] using relocateValue_congr prefixMap id acquired.length iterator agrees iteratorInside
  have exhausted := consumed.trans (by simpa only [ExitResult, exitOutcome, if_pos rfl, sameIterator,
    relocateValue_id, inputIteratorLoop, loopCode, List.nil_append] using
      (exited (inputIteratorLoop finalEnv returnGraph (iteratorBuildStack receiver [])) []))
  let roots := initialRoots base
  let tupleHeap := middleHeap ++ [.tuple (graphValues roots)]
  let bound := assignNames (iteratorSolveEnv receiver) ["nodes", "adj", "indeg"] (graphValues roots)
  have returned := (return_graph roots finalEnv middleHeap bindings base.length (iteratorBuildStack receiver [])).extend extension
  have unpacked : HeapCalls.Reaches context base.length
      ⟨.value (.ref middleHeap.length), iteratorBuildStack receiver [], tupleHeap⟩
      ⟨.block returnTopo bound, [], tupleHeap⟩ := by
    apply HeapCalls.Reaches.next _ .refl
    simp [HeapCalls.step, iteratorBuildStack, tupleHeap, Mutable.sequence, HeapCalls.run,
      graphValues, assignNames, bound]
  obtain ⟨after, sorted, retained⟩ := return_topo choices tupleHeap roots next
    (nextRep.append roots next middleHeap [.tuple (graphValues roots)]) bound
      (by simp [bound, assignNames, graphValues, setLocal])
      (by simp [bound, assignNames, graphValues, setLocal])
      (by simp [bound, assignNames, graphValues, setLocal]) base.length finalUsed []
  have agree := ConsumersAgree.traversal emptyStorage next [] attempts ConsumersAgree.empty valid data
  have same : resultOutcome (order next.vertices next.edges) = p79.checkedOutcome attempts := by
    simpa only [ConsumersAgree, processedOutcome, GraphData.process, Storage.graphData,
      p79.validatedOutcome, List.nil_append] using agree []
  have finished := SetCalls.finishes_extend (ContextResult.finishes _ _ _ _ _ _ _ sorted) extension
  rw [same] at finished
  have continued := IteratorCalls.finishes_of_set finished (by
    intro completion equal
    cases equal
    unfold p79.checkedOutcome
    split <;> simp [p79.cycleException])
  refine ⟨after, finalMap, ?_, ?_, ?_⟩
  · obtain ⟨extra, rfl⟩ := retained
    exact (extended.layout.runtime.append_private [.tuple (graphValues roots)]).append_private extra
  · intro index bounded
    exact (extended.agrees index (by omega)).trans (agrees index (by omega))
  · have reached := entered.trans (by simpa only [ExitResult, relocateValue_id, inputIteratorLoop,
      List.nil_append] using exhausted)
    exact (reached.trans (IteratorCalls.Reaches.of_heap (returned.trans unpacked) [])).finishes continued


theorem allocating_result_equiv (context : Module)
    (solverExtension : Extends context graph.program) (referenceExtension : Extends context p79.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (receiver iterator : Mutable.Value) (base acquired middle yielded : Mutable.Heap)
    (acquiredUsed nextUsed finalUsed : Nat) (closed : Closed base)
    (acquisition : AcquireRun context choices receiver base 0 iterator acquired acquiredUsed)
    (attempts : List String)
    (trace : AllocatingYieldPrefix context choices iterator graph.text acquired acquiredUsed attempts middle nextUsed)
    (args : List Constant)
    (last : AllocatingNextRun context choices iterator middle nextUsed (.error ⟨"StopIteration", args⟩) yielded finalUsed)
    (acquisitionSize : acquired.length = base.length) (valid : ∀ value ∈ attempts, ValidAttempt value) :
    ((IteratorCalls.call context "solve" [receiver] base).run choices).Equiv
      ((IteratorCalls.call context "naive" [receiver] base).run choices) := by
  obtain ⟨left, leftMap, leftLayout, leftFixed, optimized⟩ := solve_allocating_result context solverExtension source choices
    receiver iterator base acquired middle yielded acquiredUsed nextUsed finalUsed closed acquisition attempts trace args last acquisitionSize valid
  obtain ⟨right, rightMap, rightLayout, rightFixed, reference⟩ := p79.naive_allocating_result context referenceExtension source choices
    receiver iterator base acquired middle yielded acquiredUsed nextUsed finalUsed closed acquisition attempts trace args last acquisitionSize valid
  have acquiredClosed := (acquisition.effects source closed).1
  obtain ⟨middleClosed, prefixGrows⟩ := trace.yieldPrefix.effects source acquiredClosed
  obtain ⟨yieldedClosed, lastGrows⟩ := last.nextRun.effects source middleClosed
  apply Mutable.Call.equiv_of_finishes optimized reference
  cases ordered : order (p79.nodeValues attempts) (p79.attemptEdges attempts) with
  | none =>
    simpa only [p79.checkedOutcome, ordered, Except.map] using
      leftLayout.equivalent_scalar_completions rightLayout yieldedClosed base.length (by omega)
        leftFixed rightFixed (.error p79.cycleException)
  | some result =>
    simpa only [p79.checkedOutcome, ordered, Except.map, p79.text] using
      leftLayout.equivalent_scalar_completions rightLayout yieldedClosed base.length (by omega)
        leftFixed rightFixed (.ok (.string (String.intercalate "" result)))

end solvers.«79».source
