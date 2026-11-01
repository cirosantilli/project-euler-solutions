import solvers.«64.py.lean.d».Execution
import data.«project-euler-statements».data.py.project_euler_statements.«p64.py»

namespace solvers.«64».solve
open FormalSW.PyAst
set_option Elab.async false

theorem «Returns» (limit : Int) : (Calls.function Solvers.P64Ast.module "solve" [.int limit]).Returns :=
  (result limit).returns

theorem «Equiv(naive)» (limit : Int) :
    (Calls.function Solvers.P64Ast.module "solve" [.int limit]).Equiv
      (HeapCalls.scalarFunction P64Ast.module "naive" [.int limit]) := by
  obtain ⟨_, reference⟩ := data.«project-euler-statements».data.py.project_euler_statements.p64.naive.result limit []
  have left := (result limit).finishes
  have right := reference.scalar_finishes (values := [.int limit])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique left]
    exact right
  · intro finished
    rw [finished.unique right]
    exact left

theorem «Supported» :
    (∀ limit : Int, (Calls.function Solvers.P64Ast.module "solve" [.int limit]).Supported) ∧
    (∀ limit : Int, (HeapCalls.scalarFunction P64Ast.module "naive" [.int limit]).Supported) :=
  ⟨fun limit => («Returns» limit).supported,
    data.«project-euler-statements».data.py.project_euler_statements.p64.naive.«Supported»⟩

end solvers.«64».solve
