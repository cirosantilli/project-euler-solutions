import solvers.«79.iterator.prefix.source»
import solvers.«79.iterator.traversal.source»
import solvers.«79.solve.source»
import lang.py.FormalSW.IteratorAcquisitionIsolation

/-! The solver call initializes its graph and acquires the input before consuming actual iterator prefixes. -/
namespace solvers.«79».source
open FormalSW.PyAst HeapCalls Mutable.Relocation IteratorCalls
open solvers.«79».graph
set_option Elab.async false

def iteratorSolveEnv (receiver : Mutable.Value) : HeapCalls.Env := functionEnv solveArgs solveBody [receiver] []
def iteratorBuildStack (receiver : Mutable.Value) (stack : List HeapCalls.Frame) : List HeapCalls.Frame :=
  .unpack ["nodes", "adj", "indeg"] returnTopo (iteratorSolveEnv receiver) :: stack

theorem solve_iterator_initialize (base : Mutable.Heap) (receiver : Mutable.Value) (anchors : Nat)
    (stack : List HeapCalls.Frame) :
    HeapCalls.Reaches graph.program anchors ⟨.operation (.call "solve") [receiver], stack, base⟩
      ⟨.block (inputLoop :: returnGraph) (initialEnv (buildEnv receiver) base),
        iteratorBuildStack receiver stack, initialHeap base⟩ := by
  apply HeapCalls.Reaches.next (following := ⟨.block solveCode (iteratorSolveEnv receiver), stack, base⟩) rfl
  rw [solve_source]
  refine .next rfl (.next rfl ?_)
  apply (Evaluates.arguments_reach (.cons (Evaluates.lookup
    (show iteratorSolveEnv receiver "attempts" = some receiver from rfl)) .nil)
      (.call "build_graph") [] anchors (iteratorBuildStack receiver stack)).trans
  apply HeapCalls.Reaches.next (following :=
    ⟨.block buildCode (buildEnv receiver), iteratorBuildStack receiver stack, base⟩) rfl
  simpa only [build_parts, List.singleton_append] using
    initial_block (buildEnv receiver) base (inputLoop :: returnGraph) anchors (iteratorBuildStack receiver stack)

theorem initial_execution_layout (base : Mutable.Heap) (closed : Closed base) :
    ExecutionLayout base (initialHeap base) id := by
  have nodes := (ExecutionLayout.refl base closed).append_private_nonempty (.set []) (by simp)
  have adjacency := nodes.append_private_nonempty (.dict []) (by simp)
  have degrees := adjacency.append_private_nonempty (.dict []) (by simp)
  simpa only [initialHeap, List.append_assoc, List.cons_append, List.nil_append] using degrees

theorem initial_protected_cells (base : Mutable.Heap) :
    ProtectedCells (callerImage base id) (initialRoots base) emptyStorage (initialHeap base) := by
  constructor
  · intro address selected
    obtain ⟨index, bounded, same⟩ := selected
    simp only [id_eq] at same
    simp only [initialHeap, List.length_append, List.length_cons, List.length_nil]
    omega
  · simp [callerImage, initialRoots]
  · simp [callerImage, initialRoots]
  · simp [callerImage, initialRoots]
  · simp [emptyStorage]

theorem solve_iterator_entry (context : Module) (extension : Extends context graph.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (receiver iterator : Mutable.Value) (base acquired : Mutable.Heap) (used finalUsed anchors : Nat)
    (closed : Closed base) (ran : AcquireRun context choices receiver base used iterator acquired finalUsed)
    (noGrowth : acquired.length = base.length) (stack : List HeapCalls.Frame) (resumes : List Resume) :
    ∃ after, ExecutionLayout acquired after id ∧ Represents (initialRoots base) emptyStorage after ∧
      ProtectedCells (callerImage acquired id) (initialRoots base) emptyStorage after ∧
      IteratorCalls.Reaches context anchors choices []
        (.running ⟨⟨.operation (.call "solve") [receiver], stack, base⟩, used⟩ resumes)
        (.running ⟨⟨.block (sequenceLoop (sequenceSource iterator) 0 :: returnGraph)
          (initialEnv (buildEnv receiver) base), iteratorBuildStack receiver stack, after⟩, finalUsed⟩ resumes) := by
  have layout := initial_execution_layout base closed
  have represented := initial_represents base
  have saved := initial_protected_cells base
  obtain ⟨after, afterLayout, kept, entered⟩ := ran.for_name_isolated source (initialHeap base) id layout noGrowth anchors
  have outside (address : Nat) (separate : ¬ callerImage base id address) :
      ∀ index < base.length, id index ≠ address := by
    intro index bounded equal
    exact separate ⟨index, bounded, equal⟩
  have afterRep : Represents (initialRoots base) emptyStorage after :=
    { represented with
      nodesFound := (kept (initialRoots base).nodes _ ⟨layout, represented.nodesFound, outside _ saved.nodes⟩).saved
      adjacencyFound := (kept (initialRoots base).adjacency _ ⟨layout, represented.adjacencyFound, outside _ saved.adjacency⟩).saved
      degreesFound := (kept (initialRoots base).degrees _ ⟨layout, represented.degreesFound, outside _ saved.degrees⟩).saved
      neighborsFound := by simp [emptyStorage] }
  have afterSaved : ProtectedCells (callerImage acquired id) (initialRoots base) emptyStorage after := by
    have images : callerImage acquired id = callerImage base id := by
      funext address
      simp only [callerImage, noGrowth]
    rw [images]
    refine { saved with inside := ?_ }
    intro address selected
    obtain ⟨index, bounded, rfl⟩ := selected
    exact afterLayout.runtime.bound index (by omega)
  refine ⟨after, afterLayout, afterRep, afterSaved, ?_⟩
  apply (IteratorCalls.Reaches.of_heap ((solve_iterator_initialize base receiver anchors stack).extend extension) resumes).trans
  simpa only [relocateValue_id, loopCode, inputIteratorLoop, inputLoop] using entered
    (inputIteratorLoop (initialEnv (buildEnv receiver) base) returnGraph (iteratorBuildStack receiver stack)) resumes "attempts"
    (by simpa only [relocateValue_id] using
      (show initialEnv (buildEnv receiver) base "attempts" = some receiver from rfl))

theorem solve_iterator_rejection (context : Module) (extension : Extends context graph.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (receiver iterator : Mutable.Value) (base acquired middle yielded : Mutable.Heap)
    (acquiredUsed nextUsed finalUsed : Nat) (closed : Closed base)
    (acquisition : AcquireRun context choices receiver base 0 iterator acquired acquiredUsed)
    (attempts : List String) (s : String)
    (trace : YieldPrefix context choices iterator graph.text acquired acquiredUsed attempts middle nextUsed)
    (ran : NextRun context choices iterator middle nextUsed (.ok (graph.text s)) yielded finalUsed)
    (noGrowth : yielded.length = base.length) (valid : ∀ value ∈ attempts, ValidAttempt value)
    (invalid : ¬ ValidAttempt s) :
    ∃ after, ExecutionLayout yielded after id ∧
      ((IteratorCalls.call context "solve" [receiver] base).run choices).Finishes
        ⟨.error (invalidException s), [], after, base.length⟩ := by
  obtain ⟨acquiredClosed, acquisitionGrows, _⟩ := acquisition.effects source closed
  obtain ⟨middleClosed, prefixGrows⟩ := trace.effects source acquiredClosed
  have lastGrows := (ran.effects source middleClosed).2
  obtain ⟨initialized, layout, represented, saved, entered⟩ := solve_iterator_entry context extension source choices
    receiver iterator base acquired 0 acquiredUsed base.length closed acquisition (by omega) [] []
  obtain ⟨after, finalLayout, failed⟩ := graph.reject_after_iterator_prefix context extension source choices iterator
    acquired middle yielded acquiredUsed nextUsed finalUsed attempts s trace ran (by omega) valid invalid
      initialized id (initialRoots base) emptyStorage (initialEnv (buildEnv receiver) base) layout represented saved
      (initial_bindings (buildEnv receiver) base) returnGraph base.length (iteratorBuildStack receiver []) []
  refine ⟨after, finalLayout, ?_⟩
  exact (entered.trans (by simpa only [relocateValue_id] using failed)).finishes (.done rfl)

theorem solve_iterator_failure (context : Module) (extension : Extends context graph.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (receiver iterator : Mutable.Value) (base acquired yielded : Mutable.Heap)
    (acquiredUsed finalUsed : Nat) (closed : Closed base)
    (acquisition : AcquireRun context choices receiver base 0 iterator acquired acquiredUsed)
    (attempts : List String) (error : ExceptionObject)
    (trace : Traversal context choices iterator graph.text acquired acquiredUsed attempts (.error error) yielded finalUsed)
    (noGrowth : yielded.length = base.length) (valid : ∀ value ∈ attempts, ValidAttempt value) :
    ∃ after, ExecutionLayout yielded after id ∧
      ((IteratorCalls.call context "solve" [receiver] base).run choices).Finishes
        ⟨.error error, [], after, base.length⟩ := by
  obtain ⟨acquiredClosed, acquisitionGrows, _⟩ := acquisition.effects source closed
  have traversalGrows := (trace.effects source acquiredClosed).2
  obtain ⟨initialized, layout, represented, saved, entered⟩ := solve_iterator_entry context extension source choices
    receiver iterator base acquired 0 acquiredUsed base.length closed acquisition (by omega) [] []
  obtain ⟨after, _, _, _, _, _, finalLayout, _, failed⟩ := process_iterator_traversal context extension source choices iterator
    acquired yielded acquiredUsed finalUsed attempts (.error error) trace (by omega) valid
      initialized id (initialRoots base) emptyStorage (initialEnv (buildEnv receiver) base) layout represented saved
      (initial_bindings (buildEnv receiver) base) returnGraph base.length (iteratorBuildStack receiver []) []
  refine ⟨after, finalLayout, ?_⟩
  exact (entered.trans (by simpa only [ExitResult, relocateValue_id] using failed)).finishes (.done rfl)

end solvers.«79».source
