import solvers.«79.py.lean.d».Execution
import lang.py.FormalSW.IteratorAllocatingAcquisition

namespace solvers.«79».source
open FormalSW.PyAst HeapCalls Mutable.Relocation IteratorCalls
open solvers.«79».graph
set_option Elab.async false

theorem solve_allocating_entry (context : Module) (extension : Extends context graph.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (receiver iterator : Mutable.Value) (base acquired : Mutable.Heap) (used finalUsed anchors : Nat)
    (closed : Closed base) (ran : AllocatingAcquireRun context choices receiver base used iterator acquired finalUsed)
    (stack : List HeapCalls.Frame) (resumes : List Resume) :
    ∃ after rename, ExecutionLayout acquired after rename ∧
      (∀ index < base.length, rename index = index) ∧ Represents (initialRoots base) emptyStorage after ∧
      ProtectedCells (callerImage acquired rename) (initialRoots base) emptyStorage after ∧
      IteratorCalls.Reaches context anchors choices []
        (.running ⟨⟨.operation (.call "solve") [receiver], stack, base⟩, used⟩ resumes)
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: returnGraph)
          (initialEnv (buildEnv receiver) base), iteratorBuildStack receiver stack, after⟩, finalUsed⟩ resumes) := by
  have layout := initial_execution_layout base closed
  have represented := initial_represents base
  have saved := initial_protected_cells base
  obtain ⟨after, rename, extended, kept, entered⟩ := ran.for_name_isolated source (initialHeap base) id layout anchors
  have outside (address : Nat) (separate : ¬ callerImage base id address) :
      ∀ index < base.length, id index ≠ address := by
    intro index bounded equal
    exact separate ⟨index, bounded, equal⟩
  have nodes := kept (initialRoots base).nodes _ (by simp) ⟨layout, represented.nodesFound, outside _ saved.nodes⟩
  have adjacency := kept (initialRoots base).adjacency _ (by simp) ⟨layout, represented.adjacencyFound, outside _ saved.adjacency⟩
  have degrees := kept (initialRoots base).degrees _ (by simp) ⟨layout, represented.degreesFound, outside _ saved.degrees⟩
  have afterRep : Represents (initialRoots base) emptyStorage after :=
    { represented with
      nodesFound := nodes.saved
      adjacencyFound := adjacency.saved
      degreesFound := degrees.saved
      neighborsFound := by simp [emptyStorage] }
  have excluded (address : Nat) (object : Mutable.Object) (cell : PrivateLayout acquired after rename address object) :
      ¬ callerImage acquired rename address := by
    rintro ⟨index, bounded, equal⟩
    exact cell.excludes index bounded equal
  have afterSaved : ProtectedCells (callerImage acquired rename) (initialRoots base) emptyStorage after :=
    { inside := by
        rintro address ⟨index, bounded, rfl⟩
        exact extended.layout.runtime.bound index bounded
      nodes := excluded _ _ nodes
      adjacency := excluded _ _ adjacency
      degrees := excluded _ _ degrees
      neighbors := by simp [emptyStorage] }
  refine ⟨after, rename, extended.layout, extended.agrees, afterRep, afterSaved, ?_⟩
  apply (IteratorCalls.Reaches.of_heap ((solve_iterator_initialize base receiver anchors stack).extend extension) resumes).trans
  simpa only [loopCode, inputIteratorLoop, inputLoop] using entered
    (inputIteratorLoop (initialEnv (buildEnv receiver) base) returnGraph (iteratorBuildStack receiver stack)) resumes "attempts"
    (by simpa only [relocateValue_id] using
      (show initialEnv (buildEnv receiver) base "attempts" = some receiver from rfl))

end solvers.«79».source
