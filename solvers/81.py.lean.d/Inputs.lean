import «.formalsw».solvers.«81.py.ast»
import ProjectEulerSolutions.Lib.RectangularMatrixCalls

namespace solvers.«81»
open FormalSW.PyAst ProjectEulerSolutions.RectangularMatrixCalls
set_option Elab.async false
abbrev program := Solvers.P81Ast.module

theorem shape_source : findFunctionIn program.body "is_rectangular" = some (arguments, shapeCode) := rfl

theorem requires_solve_0_supported (heap : Mutable.Heap) (address : Nat) (_valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]") :
    (HeapCalls.evaluate program (matrixEnv address) Solvers.P81Ast.requires_solve_0 heap).Supported := by
  obtain ⟨_, evaluated, _⟩ := condition_result shape_source heap address typed
  exact Execution.supported_of_finishes evaluated.finishes

end solvers.«81»
