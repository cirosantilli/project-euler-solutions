import solvers.«84.py.lean.d».StationaryPivot
import ProjectEulerSolutions.Lib.IntegerDivisionCalls

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

def eliminationRowBody : List Stmt :=
  match eliminationBody[(6 : Nat)]? with
  | some (.forIn _ _ body _) => body
  | _ => []

def cellExpression : Expr :=
  .binOp (.binOp
    (.binOp (.name "pivot") .mult (.subscript (.subscript (.name "matrix") (.name "row")) (.name "j"))) .sub
    (.binOp (.name "factor") .mult (.subscript (.name "pivot_values") (.name "j")))) .floorDiv (.name "previous")

def cellBody : List Stmt :=
  [.assign [.subscript (.subscript (.name "matrix") (.name "row")) (.name "j")] cellExpression]

theorem cell_source : eliminationRowBody[(1 : Nat)]? =
    some (.forIn (.name "j") (.call (.name "range")
      [.binOp (.name "column") .add (.constant (.int 1)), .binOp (.name "size") .add (.constant (.int 1))]) cellBody []) := rfl

export ProjectEulerSolutions.IntegerDivisionCalls (floor_divide)

theorem cell_value (matrix : Matrix (Fin 120) (Fin 121) Int) (heap : Mutable.Heap)
    (root : Nat) (addresses : Fin 120 → Nat) (view : MutableIntegerMatrix.View matrix heap root addresses)
    (pivotRow row : Fin 120) (column : Fin 121) (pivot factor previous : Int) (nonzero : previous ≠ 0)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root))
    (getRow : env "row" = some (.scalar (.int row.val)))
    (getColumn : env "j" = some (.scalar (.int column.val)))
    (getValues : env "pivot_values" = some (.ref (addresses pivotRow)))
    (getPivot : env "pivot" = some (.scalar (.int pivot)))
    (getFactor : env "factor" = some (.scalar (.int factor)))
    (getPrevious : env "previous" = some (.scalar (.int previous))) :
    Evaluates program env cellExpression heap
      (.scalar (.int ((pivot * matrix row column - factor * matrix pivotRow column).fdiv previous))) heap globals := by
  have current := view.entry_result program env globals _ _ row column
    (view.row_result program env globals _ _ row (.lookup getMatrix) (.lookup getRow) (by intros; simp))
    (.lookup getColumn) (by intros; simp)
  have pivotEntry := view.entry_result program env globals _ _ pivotRow column
    (.lookup getValues) (.lookup getColumn) (by intros; simp)
  have left := Evaluates.binary (op := .mult) (.lookup getPivot) current rfl
  have right := Evaluates.binary (op := .mult) (.lookup getFactor) pivotEntry rfl
  have numerator := Evaluates.binary (op := .sub) left right rfl
  exact Evaluates.binary numerator (.lookup getPrevious) (floor_divide _ _ heap nonzero)

/-- The unchanged source expression reads both operands before updating its single matrix entry. -/
theorem cell_result (matrix : Matrix (Fin 120) (Fin 121) Int) (heap : Mutable.Heap)
    (root : Nat) (addresses : Fin 120 → Nat) (view : MutableIntegerMatrix.View matrix heap root addresses)
    (pivotRow row : Fin 120) (column : Fin 121) (pivot factor previous : Int) (nonzero : previous ≠ 0)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root))
    (getRow : env "row" = some (.scalar (.int row.val)))
    (getColumn : env "j" = some (.scalar (.int column.val)))
    (getValues : env "pivot_values" = some (.ref (addresses pivotRow)))
    (getPivot : env "pivot" = some (.scalar (.int pivot)))
    (getFactor : env "factor" = some (.scalar (.int factor)))
    (getPrevious : env "previous" = some (.scalar (.int previous))) :
    let value := (pivot * matrix row column - factor * matrix pivotRow column).fdiv previous
    BlockContinues program env cellBody heap env (MutableIntegerMatrix.write heap addresses matrix row column value) globals ∧
      MutableIntegerMatrix.View (Function.update matrix row (Function.update (matrix row) column value))
        (MutableIntegerMatrix.write heap addresses matrix row column value) root addresses := by
  dsimp only
  refine ⟨?_, view.write row column _⟩
  apply BlockContinues.setItem (by intros; simp)
    (cell_value matrix heap root addresses view pivotRow row column pivot factor previous nonzero env globals
      getMatrix getRow getColumn getValues getPivot getFactor getPrevious)
    (.cons (view.row_result program env globals _ _ row (.lookup getMatrix) (.lookup getRow) (by intros; simp))
      (.cons (.lookup getColumn) .nil))
  exact view.write_operation row column _

end solvers.«84».stationary_weights
