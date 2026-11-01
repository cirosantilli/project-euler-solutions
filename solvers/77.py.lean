import solvers.«77.py.lean.d».Search
import data.«project-euler-statements».data.py.project_euler_statements.«p77.py»

/-! The actual bytearray/DP search and cached tuple reference agree for every signed threshold. -/
namespace solvers.«77».solve
open FormalSW.PyAst
set_option Elab.async false

theorem «Returns» (target : Int) :
    (HeapCalls.scalarFunction program "solve" [.int target]).Returns := by
  obtain ⟨_, returned⟩ := result target []
  exact returned.scalar_returns (values := [.int target])

theorem «Equiv(naive)» (target : Int) :
    (HeapCalls.scalarFunction program "solve" [.int target]).Equiv
      (HeapMemoCalls.scalarFunction P77Ast.module "naive" [.int target]) := by
  obtain ⟨_, returned⟩ := result target []
  have optimized := returned.scalar_finishes (values := [.int target])
  have reference := data.«project-euler-statements».data.py.project_euler_statements.p77.naive.scalar_result target
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact reference
  · intro finished
    rw [finished.unique reference]
    exact optimized

theorem «Supported» :
    (∀ target : Int, (HeapCalls.scalarFunction program "solve" [.int target]).Supported) ∧
    (∀ target : Int, (HeapMemoCalls.scalarFunction P77Ast.module "naive" [.int target]).Supported) := by
  constructor
  · intro target
    exact («Returns» target).supported
  · intro target
    exact (data.«project-euler-statements».data.py.project_euler_statements.p77.naive.«Returns» target).supported

end solvers.«77».solve
