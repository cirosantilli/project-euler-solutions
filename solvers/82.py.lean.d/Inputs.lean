import «.formalsw».solvers.«82.py.ast»
import ProjectEulerSolutions.Lib.NonnegativeMatrixCalls

namespace solvers.«82»
open FormalSW.PyAst ProjectEulerSolutions.RectangularMatrixCalls
set_option Elab.async false
abbrev program := Solvers.P82Ast.module

theorem shape_source : findFunctionIn program.body "is_rectangular" = some (arguments, shapeCode) := rfl

theorem requires_solve_0_supported (heap : Mutable.Heap) (address : Nat) (_valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]") :
    (HeapCalls.evaluate program (matrixEnv address) Solvers.P82Ast.requires_solve_0 heap).Supported := by
  obtain ⟨_, evaluated, _⟩ := condition_result shape_source heap address typed
  exact Execution.supported_of_finishes evaluated.finishes

theorem nonnegative_source : findFunctionIn program.body "has_nonnegative_entries" =
    some (arguments, ProjectEulerSolutions.NonnegativeMatrixCalls.checkCode) := rfl

theorem requires_solve_1_supported (heap : Mutable.Heap) (address : Nat) (_valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]")
    (_shape : HeapCalls.Condition program (matrixEnv address) Solvers.P82Ast.requires_solve_0 heap) :
    (HeapCalls.evaluate program (matrixEnv address) Solvers.P82Ast.requires_solve_1 heap).Supported :=
  ProjectEulerSolutions.NonnegativeMatrixCalls.condition_supported nonnegative_source heap address typed

end solvers.«82»
