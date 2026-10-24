import «.formalsw».solvers.«61.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p61.py»
import FormalSW

namespace solvers.«61».polygonal
open FormalSW.PyAst

private abbrev answer :=
  data.«project-euler-statements».data.py.project_euler_statements.p61.polygonal.answer

/-- Source-level arithmetic, including Python's signed floor division. -/
theorem result (s n : Int) :
    Finishes (scalarFunction Solvers.P61Ast.module "polygonal" [.int s, .int n]).step
      (scalarFunction Solvers.P61Ast.module "polygonal" [.int s, .int n]).initial
      (.ok (.int (answer s n))) := by
  apply Execution.finishes_of_observe (fuel := 1)
  rfl

theorem «Returns» (s n : Int) :
    (scalarFunction Solvers.P61Ast.module "polygonal" [.int s, .int n]).Returns :=
  ⟨.int (answer s n), result s n⟩

theorem «Equiv(reference_polygonal)» (s n : Int) :
    (scalarFunction Solvers.P61Ast.module "polygonal" [.int s, .int n]).Equiv
      (scalarFunction P61Ast.module "polygonal" [.int s, .int n]) := by
  have reference :=
    data.«project-euler-statements».data.py.project_euler_statements.p61.polygonal.result s n
  have optimized := result s n
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact reference
  · intro finished
    rw [finished.unique reference]
    exact optimized

theorem «Supported» :
    (∀ s n : Int, (scalarFunction Solvers.P61Ast.module "polygonal" [.int s, .int n]).Supported) ∧
      (∀ s n : Int, (scalarFunction P61Ast.module "polygonal" [.int s, .int n]).Supported) :=
  ⟨fun s n => («Returns» s n).supported,
    data.«project-euler-statements».data.py.project_euler_statements.p61.polygonal.«Supported»⟩

end solvers.«61».polygonal
