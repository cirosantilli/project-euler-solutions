import «.formalsw».solvers.«61.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p61.py»
import FormalSW

namespace solvers.«61».polygonal
open FormalSW.PyAst

private abbrev answer :=
  data.«project-euler-statements».data.py.project_euler_statements.p61.polygonal.answer

/-- The actual helper preserves any surrounding loop's heap and global context. -/
theorem heap_result (s n : Int) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    HeapCalls.ReturnsValue Solvers.P61Ast.module "polygonal" [.scalar (.int s), .scalar (.int n)] heap
      (.scalar (.int (answer s n))) heap globals := by
  let params := ((findFunctionIn Solvers.P61Ast.module.body "polygonal").getD ([], [])).1
  let body := ((findFunctionIn Solvers.P61Ast.module.body "polygonal").getD ([], [])).2
  let env := HeapCalls.functionEnv params body [.scalar (.int s), .scalar (.int n)] globals
  apply HeapCalls.BlockReturns.function rfl rfl
  change HeapCalls.BlockReturns Solvers.P61Ast.module env
    [.return (data.«project-euler-statements».data.py.project_euler_statements.p61.polygonal.expression "s" "n")]
    heap (.scalar (.int (answer s n))) heap globals
  apply HeapCalls.BlockReturns.return_value
  exact data.«project-euler-statements».data.py.project_euler_statements.p61.polygonal.expression_evaluates
    _ _ _ _ _ _ _ _ (by cases globals <;> rfl) (by cases globals <;> rfl)

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
