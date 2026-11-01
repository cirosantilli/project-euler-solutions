import solvers.«66.py.lean.d».Execution
import data.«project-euler-statements».data.py.project_euler_statements.«p66.py»

namespace solvers.«66».solve
open FormalSW.PyAst
set_option Elab.async false
private abbrev program := Solvers.P66Ast.module

theorem «Returns» (limit : Int) : (Calls.function program "solve" [.int limit]).Returns :=
  (result limit).returns

theorem «Equiv(naive)» (limit : Int) :
    (Calls.function program "solve" [.int limit]).Equiv
      (Calls.function P66Ast.module "naive" [.int limit]) :=
  (result limit).equiv
    (data.«project-euler-statements».data.py.project_euler_statements.p66.naive.result limit)

theorem «Supported» :
    (∀ limit : Int, (Calls.function program "solve" [.int limit]).Supported) ∧
    (∀ limit : Int, (Calls.function P66Ast.module "naive" [.int limit]).Supported) := by
  constructor
  · intro limit
    exact («Returns» limit).supported
  · intro limit
    exact (data.«project-euler-statements».data.py.project_euler_statements.p66.naive.«Returns» limit).supported

theorem default_returns : (Calls.function program "solve" []).Returns := default_result.returns

end solvers.«66».solve
