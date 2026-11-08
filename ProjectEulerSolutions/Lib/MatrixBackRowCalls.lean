import ProjectEulerSolutions.Lib.MatrixBackSumCalls
import ProjectEulerSolutions.Lib.IntegerDivisionCalls
import ProjectEulerSolutions.Lib.BareissBackSubstitution

/-! One actual reverse-substitution expression and weight write. -/
namespace ProjectEulerSolutions.MatrixBackRowCalls
open FormalSW.PyAst HeapCalls MatrixBackBufferCalls MatrixBackSumCalls IntegerDivisionCalls
set_option Elab.async false

def backExpression : Expr := .binOp
  (.binOp (.binOp (.subscript (.subscript (.name "matrix") (.name "row")) (.unaryOp .uSub (.constant (.int 1))))
    .mult (.name "scale")) .sub backSum)
  .floorDiv (.subscript (.subscript (.name "matrix") (.name "row")) (.name "row"))

def backBody : List Stmt := [.assign [.subscript (.name "weights") (.name "row")] backExpression]

theorem back_numerator (matrix : Matrix (Fin 120) (Fin 121) Int) (values : Fin 120 → Int) (row : Fin 120) (scale : Int) :
    matrix row 120 * scale - upperProductSum matrix values row (row.val + 1) = Bareiss.numerator matrix scale values row := by
  simp only [Bareiss.numerator, Bareiss.Coefficients, upperProductSum, Nat.add_one_le_iff]
  rfl

theorem back_expression_result (module : Module) (matrix : Matrix (Fin 120) (Fin 121) Int) (values : Fin 120 → Int)
    (heap : Mutable.Heap) (root weights : Nat) (addresses : Fin 120 → Nat)
    (view : BackView matrix values heap root addresses weights) (row : Fin 120) (scale : Int)
    (nonzero : matrix row row.castSucc ≠ 0) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root)) (getWeights : env "weights" = some (.ref weights))
    (getRow : env "row" = some (.scalar (.int row.val))) (getSize : env "size" = some (.scalar (.int 120)))
    (getScale : env "scale" = some (.scalar (.int scale))) :
    Evaluates module env backExpression heap
      (.scalar (.int ((Bareiss.numerator matrix scale values row).fdiv (matrix row row.castSucc)))) heap globals := by
  have rowValue := view.matrixView.row_result module env globals _ _ row (.lookup getMatrix) (.lookup getRow) (by intros; simp)
  have constant : Evaluates module env (.subscript (.subscript (.name "matrix") (.name "row")) (.unaryOp .uSub (.constant (.int 1))))
      heap (.scalar (.int (matrix row 120))) heap globals :=
    Evaluates.subscript (by intros; simp) rowValue (Evaluates.unary Evaluates.constant rfl) (view.matrixView.entry_fromEnd row 120)
  have scaled := Evaluates.binary (op := .mult) constant (.lookup getScale) rfl
  have summed := back_sum_result module matrix values heap root weights addresses view row env globals getMatrix getWeights getRow getSize
  have numerator := Evaluates.binary (op := .sub) scaled summed rfl
  rw [back_numerator] at numerator
  have divisor := view.matrixView.entry_result module env globals _ _ row row.castSucc rowValue (.lookup getRow) (by intros; simp)
  exact Evaluates.binary numerator divisor (floor_divide _ _ heap nonzero)

theorem back_row_result (module : Module) (matrix : Matrix (Fin 120) (Fin 121) Int) (values : Fin 120 → Int)
    (heap : Mutable.Heap) (root weights : Nat) (addresses : Fin 120 → Nat)
    (view : BackView matrix values heap root addresses weights) (row : Fin 120) (scale : Int)
    (nonzero : matrix row row.castSucc ≠ 0) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root)) (getWeights : env "weights" = some (.ref weights))
    (getRow : env "row" = some (.scalar (.int row.val))) (getSize : env "size" = some (.scalar (.int 120)))
    (getScale : env "scale" = some (.scalar (.int scale))) :
    let value := (Bareiss.numerator matrix scale values row).fdiv (matrix row row.castSucc)
    BlockContinues module env backBody heap env (writeWeight heap weights values row value) globals ∧
      BackView matrix (Function.update values row value) (writeWeight heap weights values row value) root addresses weights := by
  dsimp only
  refine ⟨?_, view.write row _⟩
  exact view.write_result_in module row _ env globals backExpression
    (back_expression_result module matrix values heap root weights addresses view row scale nonzero
      env globals getMatrix getWeights getRow getSize getScale) getWeights getRow

end ProjectEulerSolutions.MatrixBackRowCalls
