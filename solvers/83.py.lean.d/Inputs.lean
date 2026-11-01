import «.formalsw».solvers.«83.py.ast»
import ProjectEulerSolutions.Lib.NonnegativeMatrixCorrectness

namespace solvers.«83»
open FormalSW.PyAst ProjectEulerSolutions.RectangularMatrixCalls
set_option Elab.async false
abbrev program := Solvers.P83Ast.module

theorem shape_source : findFunctionIn program.body "is_rectangular" = some (arguments, shapeCode) := rfl

theorem requires_solve_0_supported (heap : Mutable.Heap) (address : Nat) (_valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]") :
    (HeapCalls.evaluate program (matrixEnv address) Solvers.P83Ast.requires_solve_0 heap).Supported := by
  obtain ⟨_, evaluated, _⟩ := condition_result shape_source heap address typed
  exact Execution.supported_of_finishes evaluated.finishes

theorem nonnegative_source : findFunctionIn program.body "has_nonnegative_entries" =
    some (arguments, ProjectEulerSolutions.NonnegativeMatrixCalls.checkCode) := rfl

theorem requires_solve_1_supported (heap : Mutable.Heap) (address : Nat) (_valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]")
    (_shape : HeapCalls.Condition program (matrixEnv address) Solvers.P83Ast.requires_solve_0 heap) :
    (HeapCalls.evaluate program (matrixEnv address) Solvers.P83Ast.requires_solve_1 heap).Supported :=
  ProjectEulerSolutions.NonnegativeMatrixCalls.condition_supported nonnegative_source heap address typed

theorem nonnegative_domain (heap : Mutable.Heap) (address : Nat) (rows : MatrixRows)
    (found : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (reads : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2)))
    (requires : HeapCalls.Condition program (matrixEnv address) Solvers.P83Ast.requires_solve_1 heap) :
    ∀ pair ∈ rows, ∀ value ∈ pair.2, 0 ≤ value :=
  ProjectEulerSolutions.NonnegativeMatrixCalls.domain_of_condition nonnegative_source heap address rows found reads requires

end solvers.«83»
