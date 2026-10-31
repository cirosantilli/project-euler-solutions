import solvers.«76.source»
import data.«project-euler-statements».data.py.project_euler_statements.«p76.py»

/-! The mutable DP and cached recursive reference agree at every signed input. -/
namespace solvers.«76».solve
open FormalSW.PyAst
set_option Elab.async false

theorem «Returns» (n : Int) :
    (HeapCalls.scalarFunction program "solve" [.int n]).Returns := by
  obtain ⟨_, returned⟩ := result n []
  exact returned.scalar_returns (values := [.int n])

theorem «Equiv(naive)» (n : Int) :
    (HeapCalls.scalarFunction program "solve" [.int n]).Equiv
      (MemoCalls.sourceFunction P76Ast.module "naive" [.int n]) := by
  obtain ⟨_, returned⟩ := result n []
  have optimized := returned.scalar_finishes (values := [.int n])
  have reference :=
    (data.«project-euler-statements».data.py.project_euler_statements.p76.naive.result n).finishes
      (MemoCalls.Cache.valid_empty _)
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact reference
  · intro finished
    rw [finished.unique reference]
    exact optimized

theorem «Supported» :
    (∀ n : Int, (HeapCalls.scalarFunction program "solve" [.int n]).Supported) ∧
    (∀ n : Int, (MemoCalls.sourceFunction P76Ast.module "naive" [.int n]).Supported) := by
  constructor
  · intro n
    exact («Returns» n).supported
  · intro n
    exact (data.«project-euler-statements».data.py.project_euler_statements.p76.naive.«Returns»
      n).supported

end solvers.«76».solve
