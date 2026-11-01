import solvers.«79.iterator.traversal.source»
import solvers.«79.iterator.paired.source»
import data.«project-euler-statements».data.py.project_euler_statements.«p79.iterator.traversal.source»

/-! Both consumers traverse the same actual finite iterator execution and retain their ordering invariant. -/
namespace solvers.«79».source
open FormalSW.PyAst HeapCalls Mutable.Relocation IteratorCalls
open solvers.«79».graph
open data.«project-euler-statements».data.py.project_euler_statements
set_option Elab.async false

theorem processedOutcome_prefix (data : GraphData) (attempts rest : List String)
    (valid : ∀ s ∈ attempts, ValidAttempt s) :
    processedOutcome data (attempts ++ rest) =
      processedOutcome (attempts.foldl GraphData.extend data) rest := by
  induction attempts generalizing data with
  | nil => rfl
  | cons s tail ih =>
      have head := valid s (by simp)
      simpa only [List.cons_append, List.foldl_cons, processedOutcome, GraphData.process, if_pos head]
        using ih (data.extend s) (fun value member => valid value (List.mem_cons_of_mem _ member))

theorem ConsumersAgree.traversal (state next : Storage) (checked attempts : List String)
    (agree : ConsumersAgree state checked) (valid : ∀ s ∈ attempts, ValidAttempt s)
    (data : next.graphData = attempts.foldl GraphData.extend state.graphData) :
    ConsumersAgree next (checked ++ attempts) := by
  intro rest
  have matched := agree (attempts ++ rest)
  rw [processedOutcome_prefix _ _ _ valid, p79.validatedOutcome_prefix _ _ _ valid] at matched
  simpa only [data] using matched

theorem paired_iterator_traversal (context : Module)
    (solverExtension : Extends context graph.program) (referenceExtension : Extends context p79.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (iterator : Mutable.Value) (logical finalLogical : Mutable.Heap) (used finalUsed : Nat)
    (attempts : List String) (outcome : Except ExceptionObject Unit)
    (trace : Traversal context choices iterator graph.text logical used attempts outcome finalLogical finalUsed)
    (noGrowth : finalLogical.length = logical.length) (valid : ∀ s ∈ attempts, ValidAttempt s)
    (left right : Mutable.Heap) (leftMap rightMap : Nat → Nat)
    (roots : Roots) (state : Storage) (checked : List String) (checkedAddress : Nat)
    (leftEnv rightEnv : HeapCalls.Env)
    (leftLayout : ExecutionLayout logical left leftMap)
    (rightLayout : PrivateLayout logical right rightMap checkedAddress (.list (checked.map p79.text)))
    (represented : Represents roots state left)
    (saved : ProtectedCells (callerImage logical leftMap) roots state left)
    (bindings : Bindings roots leftEnv) (getChecked : rightEnv "checked" = some (.ref checkedAddress))
    (agree : ConsumersAgree state checked)
    (leftFollowing rightFollowing : List Stmt) (leftAnchors rightAnchors : Nat)
    (leftStack rightStack : List HeapCalls.Frame) (leftResumes rightResumes : List Resume) :
    ∃ leftAfter rightAfter next nextLeftEnv nextRightEnv,
      Represents roots next leftAfter ∧ ProtectedCells (callerImage logical leftMap) roots next leftAfter ∧
      Bindings roots nextLeftEnv ∧ ExecutionLayout finalLogical leftAfter leftMap ∧
      PrivateLayout finalLogical rightAfter rightMap checkedAddress (.list ((checked ++ attempts).map p79.text)) ∧
      nextRightEnv "checked" = some (.ref checkedAddress) ∧
      ConsumersAgree next (checked ++ attempts) ∧
      ExitResult context choices leftAnchors
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue leftMap iterator)) 0 :: leftFollowing)
          leftEnv, leftStack, left⟩, used⟩ leftResumes)
        (inputIteratorLoop nextLeftEnv leftFollowing leftStack) leftResumes leftAfter finalUsed outcome ∧
      ExitResult context choices rightAnchors
        (.running ⟨⟨.block (p79.validationLoop (sequenceSource (relocateValue rightMap iterator)) 0 :: rightFollowing)
          rightEnv, rightStack, right⟩, used⟩ rightResumes)
        (p79.validationIteratorLoop nextRightEnv rightFollowing rightStack) rightResumes rightAfter finalUsed outcome := by
  obtain ⟨leftAfter, next, nextLeftEnv, nextRep, nextSaved, nextBindings, nextLayout, data, leftRun⟩ :=
    process_iterator_traversal context solverExtension source choices iterator logical finalLogical used finalUsed
      attempts outcome trace noGrowth valid left leftMap roots state leftEnv leftLayout represented saved bindings
      leftFollowing leftAnchors leftStack leftResumes
  obtain ⟨rightAfter, nextRightEnv, nextPrivate, nextChecked, rightRun⟩ :=
    p79.validated_iterator_traversal context referenceExtension source choices iterator logical finalLogical used finalUsed
      attempts outcome trace noGrowth valid right rightMap checkedAddress checked rightEnv rightLayout getChecked
      rightFollowing rightAnchors rightStack rightResumes
  exact ⟨leftAfter, rightAfter, next, nextLeftEnv, nextRightEnv, nextRep, nextSaved, nextBindings,
    nextLayout, nextPrivate, nextChecked, agree.traversal state next checked attempts valid data, leftRun, rightRun⟩

end solvers.«79».source
