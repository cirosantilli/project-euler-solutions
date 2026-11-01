import solvers.«86.py.lean.d».Search
import data.«project-euler-statements».data.py.project_euler_statements.«p86.py»

/-! Termination and equivalence of the actual Python searches for every integer threshold. -/
namespace solvers.«86».solve
open FormalSW.PyAst
set_option Elab.async false

theorem «Returns» (target : Int) :
    (Calls.function program "solve" [.int target]).Returns :=
  (result target).returns

theorem «Equiv(naive)» (target : Int) :
    (Calls.function program "solve" [.int target]).Equiv
      (Calls.function P86Ast.module "naive" [.int target]) :=
  (result target).equiv
    (data.«project-euler-statements».data.py.project_euler_statements.p86.naive.result target)

theorem «Supported» :
    (∀ target : Int, (Calls.function program "solve" [.int target]).Supported) ∧
    (∀ target : Int, (Calls.function P86Ast.module "naive" [.int target]).Supported) := by
  constructor
  · intro target
    exact («Returns» target).supported
  · exact data.«project-euler-statements».data.py.project_euler_statements.p86.naive.«Supported»

end solvers.«86».solve
