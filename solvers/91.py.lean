import solvers.«91.py.lean.d».Execution
import data.«project-euler-statements».data.py.project_euler_statements.«p91.py»

/-! Termination and equivalence of the actual Python grid counts for every signed bound. -/
namespace solvers.«91».solve
open FormalSW.PyAst
set_option Elab.async false

theorem «Returns» (n : Int) :
    (Calls.function program "solve" [.int n]).Returns :=
  (result n).returns

theorem «Equiv(naive)» (n : Int) :
    (Calls.function program "solve" [.int n]).Equiv
      (Calls.function P91Ast.module "naive" [.int n]) :=
  (result n).equiv
    (data.«project-euler-statements».data.py.project_euler_statements.p91.naive.result n)

theorem «Supported» :
    (∀ n : Int, (Calls.function program "solve" [.int n]).Supported) ∧
    (∀ n : Int, (Calls.function P91Ast.module "naive" [.int n]).Supported) := by
  constructor
  · intro n
    exact («Returns» n).supported
  · exact data.«project-euler-statements».data.py.project_euler_statements.p91.naive.«Supported»

end solvers.«91».solve
