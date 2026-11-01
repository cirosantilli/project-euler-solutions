import solvers.«79.iterator.prefix.source»
import data.«project-euler-statements».data.py.project_euler_statements.«p79.iterator.prefix.source»

/-! The first invalid attempt produces the same exception and caller mutations in both consumers. -/
namespace solvers.«79».source
open FormalSW.PyAst HeapCalls Mutable.Relocation IteratorCalls
open solvers.«79».graph
open data.«project-euler-statements».data.py.project_euler_statements
set_option Elab.async false

theorem paired_iterator_rejection (context : Module)
    (solverExtension : Extends context graph.program) (referenceExtension : Extends context p79.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (iterator : Mutable.Value) (logical middle yielded : Mutable.Heap) (used nextUsed finalUsed : Nat)
    (attempts : List String) (s : String)
    (trace : YieldPrefix context choices iterator graph.text logical used attempts middle nextUsed)
    (ran : NextRun context choices iterator middle nextUsed (.ok (graph.text s)) yielded finalUsed)
    (noGrowth : yielded.length = logical.length) (valid : ∀ value ∈ attempts, ValidAttempt value)
    (invalid : ¬ ValidAttempt s)
    (left right : Mutable.Heap) (leftMap rightMap : Nat → Nat)
    (roots : Roots) (state : Storage) (checked : List String) (checkedAddress : Nat)
    (leftEnv rightEnv : HeapCalls.Env)
    (leftLayout : ExecutionLayout logical left leftMap)
    (rightLayout : PrivateLayout logical right rightMap checkedAddress (.list (checked.map p79.text)))
    (represented : Represents roots state left)
    (saved : ProtectedCells (callerImage logical leftMap) roots state left)
    (bindings : Bindings roots leftEnv) (getChecked : rightEnv "checked" = some (.ref checkedAddress))
    (leftFollowing rightFollowing : List Stmt) (leftAnchors rightAnchors : Nat)
    (leftStack rightStack : List HeapCalls.Frame) (leftResumes rightResumes : List Resume) :
    ∃ leftAfter rightAfter,
      ExecutionLayout yielded leftAfter leftMap ∧
      PrivateLayout yielded rightAfter rightMap checkedAddress (.list ((checked ++ attempts).map p79.text)) ∧
      IteratorCalls.Reaches context leftAnchors choices []
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue leftMap iterator)) 0 :: leftFollowing)
          leftEnv, leftStack, left⟩, used⟩ leftResumes)
        (.completed (.ok ⟨.error (graph.invalidException s), [], leftAfter, leftAnchors⟩) finalUsed leftResumes) ∧
      IteratorCalls.Reaches context rightAnchors choices []
        (.running ⟨⟨.block (p79.validationLoop (sequenceSource (relocateValue rightMap iterator)) 0 :: rightFollowing)
          rightEnv, rightStack, right⟩, used⟩ rightResumes)
        (.completed (.ok ⟨.error (graph.invalidException s), [], rightAfter, rightAnchors⟩) finalUsed rightResumes) := by
  obtain ⟨leftAfter, finalLayout, leftRun⟩ := graph.reject_after_iterator_prefix context solverExtension source choices
    iterator logical middle yielded used nextUsed finalUsed attempts s trace ran noGrowth valid invalid
      left leftMap roots state leftEnv leftLayout represented saved bindings leftFollowing leftAnchors leftStack leftResumes
  have rejected : s.length ≠ 3 ∨ Strings.isDigit s = false := by
    by_cases length : s.length = 3
    · exact Or.inr (by simpa [ValidAttempt, length] using invalid)
    · exact Or.inl length
  obtain ⟨rightAfter, finalPrivate, rightRun⟩ := p79.reject_after_iterator_prefix context referenceExtension source choices
    iterator logical middle yielded used nextUsed finalUsed attempts s trace ran noGrowth valid rejected
      right rightMap checkedAddress checked rightEnv rightLayout getChecked rightFollowing rightAnchors rightStack rightResumes
  exact ⟨leftAfter, rightAfter, finalLayout, finalPrivate, leftRun, rightRun⟩

end solvers.«79».source
