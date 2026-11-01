import solvers.«62.py.lean.d».Equivalence
import ProjectEulerSolutions.Lib.CubicFiveClass.Certificate

/-! The standard class-size request inherits termination from a checked existence witness. -/
namespace solvers.«62».solve
open FormalSW.PyAst

/-- The usual class size has a witness; every signed target retains the general equivalence. -/
theorem target_five_returns :
    (HeapCalls.scalarFunction Solvers.P62Ast.module "solve" [.int 5]).Returns := by
  obtain ⟨_, returned, _⟩ := search.result 5 _ []
    ProjectEulerSolutions.CubicFiveClass.model_returns
  exact returned.scalar_returns (values := [.int 5])

theorem reference_target_five_returns :
    (HeapCalls.scalarFunction P62Ast.module "naive" [.int 5]).Returns :=
  (Execution.Equiv.returns_iff («Equiv(naive)» 5)).mp target_five_returns

end solvers.«62».solve
