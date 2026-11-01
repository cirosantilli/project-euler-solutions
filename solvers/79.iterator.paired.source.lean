import solvers.«79.iterator.isolation.source»
import solvers.«79.outcome.source»
import data.«project-euler-statements».data.py.project_euler_statements.«p79.iterator.isolation.source»

/-! Both consumers advance from one actual caller-method trace and retain their ordering invariant. -/
namespace solvers.«79».source
open FormalSW.PyAst HeapCalls Mutable.Relocation
open solvers.«79».graph
open data.«project-euler-statements».data.py.project_euler_statements
set_option Elab.async false

def ConsumersAgree (state : Storage) (checked : List String) : Prop :=
  ∀ rest, processedOutcome state.graphData rest = p79.validatedOutcome checked rest

theorem ConsumersAgree.empty : ConsumersAgree emptyStorage [] :=
  processed_outcome emptyStorage.graphData [] RepresentsTriples.empty

theorem ConsumersAgree.extend (state next : Storage) (checked : List String) (s : String)
    (agree : ConsumersAgree state checked) (valid : ValidAttempt s)
    (data : next.graphData = state.graphData.extend s) : ConsumersAgree next (checked ++ [s]) := by
  intro rest
  have matched := agree (s :: rest)
  have accepted : s.length = 3 ∧ Strings.isDigit s = true := valid
  simpa only [processedOutcome, GraphData.process, if_pos valid,
    p79.validatedOutcome, if_pos accepted, data] using matched

/-- The consumers execute their own source bodies; the method trace fixes their shared choice counter. -/
theorem paired_valid_iterator_step (context : Module)
    (solverExtension : Extends context graph.program) (referenceExtension : Extends context p79.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (leftAnchors rightAnchors used finalUsed : Nat) (logical yielded left right : Mutable.Heap)
    (leftMap rightMap : Nat → Nat) (iterator : Mutable.Value) (operation : Operation)
    (roots : Roots) (state : Storage) (checked : List String) (checkedAddress : Nat)
    (leftEnv rightEnv : HeapCalls.Env) (s : String) (valid : ValidAttempt s)
    (leftLayout : ExecutionLayout logical left leftMap)
    (rightLayout : PrivateLayout logical right rightMap checkedAddress (.list (checked.map p79.text)))
    (inside : iterator.edge.Within logical.length)
    (isObject : IteratorCalls.isInstance logical iterator = true)
    (bound : ProtocolCalls.nextMethod context iterator logical = some (.ok (some operation), logical))
    (executed : IteratorCalls.Reaches context logical.length choices []
      (.running ⟨⟨.operation operation [], [], logical⟩, used⟩ [])
      (.completed (.ok ⟨.ok (graph.text s), [], yielded, logical.length⟩) finalUsed []))
    (noGrowth : yielded.length = logical.length)
    (represented : Represents roots state left)
    (saved : ProtectedCells (callerImage logical leftMap) roots state left)
    (bindings : Bindings roots leftEnv) (getChecked : rightEnv "checked" = some (.ref checkedAddress))
    (agree : ConsumersAgree state checked)
    (leftFollowing rightFollowing : List Stmt) (leftStack rightStack : List HeapCalls.Frame)
    (leftResumes rightResumes : List IteratorCalls.Resume) :
    ∃ (leftAfter rightYielded : Mutable.Heap) (next : Storage) (nextEnv : HeapCalls.Env)
      (nextLeftMap nextRightMap : Nat → Nat),
      IteratorCalls.Reaches context leftAnchors choices []
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue leftMap iterator)) 0 :: leftFollowing)
          leftEnv, leftStack, left⟩, used⟩ leftResumes)
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue leftMap iterator)) 0 :: leftFollowing)
          nextEnv, leftStack, leftAfter⟩, finalUsed⟩ leftResumes) ∧
      IteratorCalls.Reaches context rightAnchors choices []
        (.running ⟨⟨.block (p79.validationLoop (sequenceSource (relocateValue rightMap iterator)) 0 :: rightFollowing)
          rightEnv, rightStack, right⟩, used⟩ rightResumes)
        (.running ⟨⟨.block (p79.validationLoop (sequenceSource (relocateValue rightMap iterator)) 0 :: rightFollowing)
          (setLocal rightEnv "attempt" (p79.text s)), rightStack,
          rightYielded.set checkedAddress (.list ((checked ++ [s]).map p79.text))⟩, finalUsed⟩ rightResumes) ∧
      (∀ index < logical.length, nextLeftMap index = leftMap index) ∧
      (∀ index < logical.length, nextRightMap index = rightMap index) ∧
      Represents roots next leftAfter ∧ ProtectedCells (callerImage logical leftMap) roots next leftAfter ∧
      Bindings roots nextEnv ∧ ExecutionLayout yielded leftAfter nextLeftMap ∧
      PrivateLayout yielded (rightYielded.set checkedAddress (.list ((checked ++ [s]).map p79.text)))
        nextRightMap checkedAddress (.list ((checked ++ [s]).map p79.text)) ∧
      ConsumersAgree next (checked ++ [s]) := by
  obtain ⟨leftAfter, next, nextEnv, nextLeftMap, leftAgrees, leftRun,
    nextRep, nextSaved, nextBindings, nextLayout, data⟩ :=
    iterator_valid_from_logical context solverExtension source choices leftAnchors used finalUsed
      logical left yielded leftMap iterator operation roots state leftEnv s valid leftLayout inside
      isObject bound executed noGrowth represented saved bindings leftFollowing leftStack leftResumes
  obtain ⟨rightYielded, nextRightMap, rightAgrees, rightRun, nextPrivate⟩ :=
    p79.iterator_accepted_from_logical context referenceExtension source choices rightAnchors used finalUsed
      logical right yielded rightMap iterator operation checkedAddress checked rightEnv s valid.1 valid.2
      rightLayout inside isObject bound executed noGrowth getChecked rightFollowing rightStack rightResumes
  exact ⟨leftAfter, rightYielded, next, nextEnv, nextLeftMap, nextRightMap,
    leftRun, rightRun, leftAgrees, rightAgrees, nextRep, nextSaved, nextBindings, nextLayout, nextPrivate,
    agree.extend state next checked s valid data⟩

end solvers.«79».source
