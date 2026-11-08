import solvers.«84.py.lean.d».StationaryBackSum
import ProjectEulerSolutions.Lib.MatrixBackRowCalls

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

export ProjectEulerSolutions.MatrixBackRowCalls (backExpression back_numerator)
abbrev back_expression_result := ProjectEulerSolutions.MatrixBackRowCalls.back_expression_result program

def backBody : List Stmt :=
  match (executableBody definition.2)[(8 : Nat)]? with
  | some (.forIn _ _ body _) => body
  | _ => []

theorem back_body_source : backBody = [.assign [.subscript (.name "weights") (.name "row")] backExpression] := rfl

theorem back_row_result {sides : Nat} (state : Bareiss.State (Monopoly.augmentedMatrix sides) 119) (values : Fin 120 → Int)
    (heap : Mutable.Heap) (root weights : Nat) (addresses : Fin 120 → Nat)
    (view : BackView state.matrix values heap root addresses weights) (row : Fin 120) (scale : Int)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root)) (getWeights : env "weights" = some (.ref weights))
    (getRow : env "row" = some (.scalar (.int row.val))) (getSize : env "size" = some (.scalar (.int 120)))
    (getScale : env "scale" = some (.scalar (.int scale))) :
    let value := (Bareiss.numerator state.matrix scale values row).fdiv (state.matrix row row.castSucc)
    BlockContinues program env backBody heap env (writeWeight heap weights values row value) globals ∧
      BackView state.matrix (Function.update values row value) (writeWeight heap weights values row value) root addresses weights := by
  exact ProjectEulerSolutions.MatrixBackRowCalls.back_row_result program state.matrix values heap root weights addresses
    view row scale (state.diagonal_nonzero row) env globals getMatrix getWeights getRow getSize getScale

end solvers.«84».stationary_weights
